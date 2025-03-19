from pathlib import Path
from typing import List
import json, random, time, os
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from extra.models.llama import Transformer, convert_from_huggingface, convert_from_gguf, fix_bf16
from tinygrad.nn.state import safe_load, torch_load, load_state_dict, get_parameters, gguf_load
from tinygrad import Tensor, dtypes, nn, Context, Device, GlobalCounters
from tinygrad.helpers import Profiling, Timing, DEBUG, colored, fetch, tqdm

# **** helper functions ****
def concat_weights(models, device=None):
  def convert(name) -> Tensor:
    disk_tensors: List[Tensor] = [model[name] for model in models]
    if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
      return disk_tensors[0].to(device=device)
    axis = 1 if name.endswith((".attention.wo.weight", ".feed_forward.w2.weight")) else 0
    lazy_tensors = [data.to(device=device) for data in disk_tensors]
    return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
  return {name: convert(name) for name in {name: None for model in models for name in model}}

def load(fn:str):
  if fn.endswith('.index.json'):
    with open(fn) as fp: weight_map = json.load(fp)['weight_map']
    parts = {n: load(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
    return {k: parts[n][k] for k, n in weight_map.items()}
  elif fn.endswith(".gguf"):
    gguf_tensor = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}").to(Device.DEFAULT)
    return gguf_load(gguf_tensor)[1]
  elif fn.endswith(".safetensors"):
    return safe_load(fn)
  else:
    return torch_load(fn)

# **** quantized linears ****
class Int8Linear:
  def __init__(self, in_features, out_features, bias=False):
    assert bias == False
    self.weight = Tensor.ones(out_features, in_features, dtype=dtypes.int8)
    self.scale = Tensor.ones(out_features, dtype=dtypes.half)

  def __call__(self, x):
    return x.dot(self.weight.cast(self.scale.dtype).T*self.scale)

  @staticmethod
  def quantize(tensors, device, scale_dtype=dtypes.float16, quantize_embeds=False):
    new_tensors = {}
    for name,v in tensors.items():
      if "feed_forward" in name or "attention.w" in name or (quantize_embeds and "tok_embeddings.weight" in name):
        assert "weight" in name, name
        v = v.cast(scale_dtype)
        scale = v.abs().max(axis=1) / 127.0
        int8_weight = (v.T/scale).T.round().cast(dtype=dtypes.int8) # without round(), cast truncates -34.9 to -34
        new_tensors[name] = int8_weight
        new_tensors[name.replace('weight', 'scale')] = scale
        if isinstance(device, tuple):
          new_tensors[name].shard_(device, axis=-1)
          new_tensors[name.replace('weight', 'scale')].shard_(device, axis=None)
      else:
        new_tensors[name] = v
    if quantize_embeds: new_tensors.update({"output.weight": new_tensors["tok_embeddings.weight"], "output.scale": new_tensors["tok_embeddings.scale"]})
    return new_tensors

class Int8Embedding:
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_sz, self.embed_sz = vocab_size, embed_size
    self.weight, self.scale = Tensor.ones(vocab_size, embed_size, dtype=dtypes.int8), Tensor.ones(vocab_size, dtype=dtypes.half)

  def __call__(self, idx:Tensor) -> Tensor:
    if not hasattr(self, 'arange'): self.arange = Tensor.arange(self.vocab_sz, requires_grad=False, device=self.weight.device).unsqueeze(-1)
    big_shp = idx.shape+(self.vocab_sz, self.embed_sz)
    arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1)).expand(big_shp), (self.weight.cast(self.scale.dtype).T*self.scale).T
    return (arange == idx).mul(vals).sum(-2, dtype=vals.dtype)

def NF4Linear(block_size):
  _CODE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
  ]
  CODE = Tensor.stack(*[Tensor(c, dtype=dtypes.float16) for c in _CODE])
  class _NF4Linear:
    def __init__(self, in_features, out_features, bias=False):
      assert not bias, "bias not supported"
      self.in_features, self.out_features = in_features, out_features
      self.weight = Tensor.empty(int(out_features * in_features / 2), dtype=dtypes.uint8)
      self.scale = Tensor.empty(int(out_features * in_features / block_size), 1, dtype=dtypes.float16)

    def __call__(self, x: Tensor) -> Tensor:
      high_bits = self.weight
      low_bits = (self.weight * 2 ** 4).contiguous()
      unpacked = Tensor.stack(high_bits, low_bits, dim=-1).idiv(2 ** 4)
      unscaled = CODE[unpacked].to(x.device).reshape(-1, block_size) * self.scale
      return x.linear(unscaled.reshape(self.out_features, self.in_features).T)

    @staticmethod
    def quantize(state_dict: dict[str, Tensor], device, scale_dtype=dtypes.float16) -> dict[str, Tensor]:
      new_state_dict = {}
      for k, v in state_dict.items():
        if "feed_forward" in k or "attention.w" in k:
          grouped = v.reshape(-1, block_size)
          scale = (grouped.abs().max(axis=1, keepdim=True))
          coded = ((grouped / scale).unsqueeze(-1) - CODE.to(v.device)).abs().argmin(axis=-1).cast(dtypes.uint8).flatten()
          new_state_dict[k] = coded[::2] * 2 ** 4 + coded[1::2]
          new_state_dict[k.replace(".weight", ".scale")] = scale.cast(scale_dtype)
          if isinstance(device, tuple):
            new_state_dict[k].shard_(device, axis=-1)
            new_state_dict[k.replace('weight', 'scale')].shard_(device, axis=None)
        else:
          new_state_dict[k] = v
      return new_state_dict
  return _NF4Linear

MODEL_PARAMS = {
  "1B": {
    "args": {"dim": 2048, "n_heads": 32, "n_kv_heads": 8, "n_layers": 16, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192},
    "files": 1
  },
  "8B": {
    "args": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 14336},
    "files": 1
  },
  "70B": {
    "args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256,  "hidden_dim": 28672},
    "files": 8
  }
}

def build_transformer(model_path: Path, model_size="8B", quantize=None, scale_dtype=dtypes.float16, device=None, max_context=8192, load_weights=True):
  # build model
  if quantize == "int8": linear, embedding, quantize_embeds = Int8Linear, Int8Embedding, True
  elif quantize == "nf4": linear, embedding, quantize_embeds = NF4Linear(64), nn.Embedding, False
  else: linear, embedding, quantize_embeds = nn.Linear, nn.Embedding, False
  model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=linear, embedding=embedding, max_context=max_context, jit=True)

  if not load_weights: return model
  # load weights
  if model_path.is_dir():
    if (model_path / "model.safetensors.index.json").exists(): weights = load(str(model_path / "model.safetensors.index.json"))
    elif (model_path / "model.safetensors").exists(): weights = load(str(model_path / "model.safetensors"))
    else: weights = concat_weights([load(str(model_path / f"consolidated.{i:02d}.pth")) for i in range(MODEL_PARAMS[model_size]["files"])], device[0] if isinstance(device, tuple) else device)
  else:
    weights = load(str(model_path))
  if "model.embed_tokens.weight" in weights:
    weights = convert_from_huggingface(weights, model, MODEL_PARAMS[model_size]["args"]["n_heads"], MODEL_PARAMS[model_size]["args"]["n_kv_heads"])
  elif "token_embd.weight" in weights:
    weights = convert_from_gguf(weights, model)
  weights = fix_bf16(weights)

  with Context(BEAM=0):
    # quantize
    if quantize == "float16": weights = {k:v.cast(quantize).contiguous() for k,v in weights.items()}
    elif quantize is not None:
      weights = linear.quantize(weights, device, scale_dtype, quantize_embeds)
      for _,v in weights.items(): v.realize()

    # shard
    if isinstance(device, tuple):
      for k,v in nn.state.get_state_dict(model).items():
        if 'scale' in k: v.shard_(device, axis=None)  # from quantized
        elif '.attention.' in k: v.shard_(device, axis=-1)
        elif '.feed_forward.w1.' in k: v.shard_(device, axis=0)
        elif '.feed_forward.w3.' in k: v.shard_(device, axis=0)
        elif '.feed_forward.' in k: v.shard_(device, axis=-1)
        elif 'tok_embeddings.weight' in k: v.shard_(device, axis=0)
        elif 'output.weight' in k: v.shard_(device, axis=0)
        else: v.shard_(device, axis=None)

    # replace weights in model
    load_state_dict(model, weights, strict=False, consume=True)
  return model

class LlamaModel:
    # Default generation settings
    TEMPERATURE = 0.95
    TOP_K = 0
    TOP_P = 0.0
    ALPHA_F = 0.0
    ALPHA_P = 0.0
    
    def __init__(self, model_path, model_size="8B", quantize=None, shard=1, max_context=8192):
        self.model_path = Path(model_path)
        self.model_size = model_size
        self.device = tuple(f"{Device.DEFAULT}:{i}" for i in range(shard)) if shard > 1 else Device.DEFAULT
        
        # # Initialize tokenizer
        # tokenizer_path = str((self.model_path if self.model_path.is_dir() else self.model_path.parent) / "tokenizer.model")
        # self.tokenizer = Tokenizer(tokenizer_path)
        
        # Build model
        self.model = build_transformer(self.model_path, model_size=self.model_size, 
                                      quantize=quantize, device=self.device,
                                      max_context=max_context)
        
        # Get model memory size
        self.param_bytes = sum(x.lazydata.size * x.dtype.itemsize for x in get_parameters(self.model))
        
        # Track seen tokens for caching
        self.last_seen_toks = []
    
    def prefill(self, toks, start_pos=0):
        """Pre-fill the model's KV cache with the provided tokens"""
        # We can skip part of the prompt if it is the same as last and start_pos=0
        if start_pos == 0:
            for i, (a, b) in enumerate(zip(toks, self.last_seen_toks)):
                if a != b: break
            else: i = min(len(toks), len(self.last_seen_toks))
            start_pos += i
            self.last_seen_toks = toks
            toks = toks[i:]

        logits_mask = Tensor.ones(MODEL_PARAMS[self.model_size]["args"]["vocab_size"], dtype=dtypes.int32, device=self.device) * 200

        # Prefill the model
        for tok in tqdm(toks):
            GlobalCounters.reset()
            self.model(Tensor([[tok]], device=self.device), logits_mask.contiguous(), start_pos, 
                      self.TEMPERATURE, self.TOP_K, self.TOP_P, 
                      self.ALPHA_F, self.ALPHA_P).realize()
            start_pos += 1
        return start_pos

    def generate_next_token(self, last_tok, logits_mask, start_pos, temperature=None):
        """Generate the next token given the last token and position"""
        if temperature is None:
          temperature = self.TEMPERATURE

        GlobalCounters.reset()
        tok = self.model(Tensor([[last_tok]], device=self.device), logits_mask.contiguous(), start_pos, 
                        temperature, self.TOP_K, self.TOP_P, 
                        self.ALPHA_F, self.ALPHA_P).item()
        return tok

    def download_model(size="1B"):
        """Download model files"""
        if size == "1B":
            fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="llama3-1b-instruct")
            fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/tokenizer.json", "tokenizer.json", subdir="llama3-1b-instruct")
            fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/tokenizer_config.json", "tokenizer_config.json", subdir="llama3-1b-instruct")
            fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/special_tokens_map.json", "special_tokens_map.json", subdir="llama3-1b-instruct")
            return fetch("https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf", "Llama-3.2-1B-Instruct-Q6_K.gguf", subdir="llama3-1b-instruct")
        elif size == "8B":
            fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir="llama3-8b-sfr")
            fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00001-of-00004.safetensors", "model-00001-of-00004.safetensors", subdir="llama3-8b-sfr")
            fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00002-of-00004.safetensors", "model-00002-of-00004.safetensors", subdir="llama3-8b-sfr")
            fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00003-of-00004.safetensors", "model-00003-of-00004.safetensors", subdir="llama3-8b-sfr")
            fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00004-of-00004.safetensors", "model-00004-of-00004.safetensors", subdir="llama3-8b-sfr")
            return fetch("https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/raw/main/model.safetensors.index.json", "model.safetensors.index.json", subdir="llama3-8b-sfr")
        elif size == "70B":
            subdir = "DeepSeek-R1-Distill-Llama-70B"
            model_path = fetch("https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/model.safetensors.index.json?download=true", "model.safetensors.index.json", subdir=subdir)
            fetch("https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model", "tokenizer.model", subdir=subdir)
            for i in range(17):
                fetch(f"https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/model-{i+1:05d}-of-000017.safetensors?download=true", f"model-{i+1:05d}-of-000017.safetensors", subdir=subdir)
            return model_path 