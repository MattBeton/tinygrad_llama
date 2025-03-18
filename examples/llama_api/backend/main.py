import argparse
from pathlib import Path
from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import Profiling, Timing, DEBUG, colored
from .model import LlamaModel
from .inference import LlamaInferenceEngine
from .api import LlamaAPI

def main():
    Tensor.no_grad = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_model", action="store_true", help="Download a model")
    parser.add_argument("--model", type=Path, help="Model path")
    parser.add_argument("--size", choices=["1B", "8B", "70B"], default="1B", help="Model size")
    parser.add_argument("--shard", type=int, default=1, help="Shard the model across multiple devices")
    parser.add_argument("--quantize", choices=["int8", "nf4", "float16"], help="Quantization method")
    parser.add_argument("--no_api", action="store_true", help="Disable the api and run a cli test interface")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server bind address")
    parser.add_argument("--port", type=int, default=7776, help="Web server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--temperature", type=float, default=0.85, help="Temperature")
    parser.add_argument("--benchmark", action="store_true", help="Run a benchmark")
    parser.add_argument("--timing", action="store_true", help="Print timing per token")
    parser.add_argument("--profile", action="store_true", help="Output profile data")
    args = parser.parse_args()

    # Download model if requested or if no model provided
    if args.download_model or not args.model:
        args.model = LlamaModel.download_model(args.size)

    assert args.model is not None, "please provide --model option"

    if args.seed is not None: Tensor.manual_seed(args.seed)
    if args.benchmark: Tensor.manual_seed(42)
    print(f"seed = {Tensor._seed}")
    
    # Create model instance
    model = LlamaModel(args.model, model_size=args.size, quantize=args.quantize, shard=args.shard)
    model.TEMPERATURE = args.temperature  # Set the temperature
    
    if not args.no_api and not args.benchmark:
        # Start the API server
        inference_engine = LlamaInferenceEngine(model)
        api = LlamaAPI(inference_engine)
        api.run(host=args.host, port=args.port, debug=args.debug)
    
    elif args.benchmark:
        # Run a benchmark
        toks = [model.tokenizer.bos_id] + model.encode_message("user", "Hello.") + model.encode_role("assistant")

        start_pos = model.prefill(toks[:-1])
        last_tok = toks[-1]
        generated = ""
        for _ in range(20):
            GlobalCounters.reset()
            st = GlobalCounters.time_sum_s
            with Profiling(enabled=args.profile):
                with Timing("total ", on_exit=lambda x: f", {1e9/x:.2f} tok/s, {GlobalCounters.global_mem/x:.2f} GB/s, param {model.param_bytes/x:.2f} GB/s"):
                    with Timing("enqueue in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU" if DEBUG>=2 else "")+
                                f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                                (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s, param {model.param_bytes*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None):
                        tok = model.generate_next_token(last_tok, start_pos)
            start_pos += 1
            last_tok = tok
            generated += model.tokenizer.decode([tok])
            print(generated)
            
        # Validate output for certain models
        if "LLaMA-3/8B-SF-DPO" in args.model.as_posix() and (args.temperature == 0.85 or args.temperature == 0):
            if args.temperature == 0.85:
                EXPECTED_TEXT = {
                    1: "Hello! How can I help you today? If you have any questions or need assistance with anything,",
                    2: "Hello! How can I help you today? If you have any questions, need assistance or just want",
                    3: "Hello! How can I help you today? If you have any questions or need assistance, feel free",
                    4: "Hello! How can I assist you today? If you have any questions, need information, or require",
                    5: "Hello! How can I assist you today? If you have any questions or need help with something",
                    6: "Hello! How can I assist you today? If you have any questions, need information, or require",
                }
            else:
                EXPECTED_TEXT = {k: "Hello! How can I assist you today? If you have any questions or need help with something," for k in range(1, 7)}
            assert generated == EXPECTED_TEXT[args.shard], f"{generated=} {EXPECTED_TEXT[args.shard]}"
            print("\n" + colored("output validated", "green"))
    
    else:
        # Run CLI interface
        prompt = [model.tokenizer.bos_id] + model.encode_message("system", "You are an helpful assistant.")

        start_pos = model.prefill(prompt)
        while True:
            toks = model.encode_message("user", input("Q: ")) + model.encode_role("assistant")

            start_pos = model.prefill(toks[:-1], start_pos=start_pos)
            last_tok = toks[-1]
            while True:
                if args.timing or args.profile: print("")
                st = GlobalCounters.time_sum_s
                with Profiling(enabled=args.profile):
                    with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9/x:.2f} tok/s, {GlobalCounters.global_mem/x:.2f} GB/s, param {model.param_bytes/x:.2f} GB/s"):
                        with Timing("enqueue in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU" if DEBUG>=2 else "")+
                                    f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                                    (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s, param {model.param_bytes*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None, enabled=args.timing):

                            tok = model.generate_next_token(last_tok, start_pos)
                start_pos += 1
                last_tok = tok
                if tok in model.tokenizer.stop_tokens: break
                print(model.tokenizer.decode([tok]), end="", flush=True)
            print(flush=True)

if __name__ == "__main__":
    main() 