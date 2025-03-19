from enum import Enum
from typing import List, Dict, Optional
from llguidance import LLInterpreter
from llguidance.hf import from_tokenizer as llg_from_tokenizer
from transformers import AutoTokenizer
import numpy as np
from tinygrad import Tensor

class TokenProcessingStatus(Enum):
    CONTINUE = None  # Continue processing tokens
    MAX_TOKENS_REACHED = "length"
    STOP_TOKEN_REACHED = "stop"

class GenerationOptions:
    def __init__(self, temperature=None, max_tokens=None, grammar:Optional[str]=None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.grammar = grammar

class StructuredOutput:
    def __init__(self, tokenizer, stop_tokens, generation_options: GenerationOptions):
        self.is_finished = False
        
        self.tokenizer = tokenizer
        self.token_count = 0
        self.max_tokens = generation_options.max_tokens
        self.stop_tokens = stop_tokens

        if generation_options.grammar:
            self.guidance_interpreter = LLInterpreter(
                llg_from_tokenizer(self.tokenizer, n_vocab=self.tokenizer.vocab_size),
                generation_options.grammar,
                # Below args copied from josh coles.
                enable_ff_tokens=False,
                enable_backtrack=False,
                # log_level=2
            )
            self.guidance_interpreter.start_without_prompt()

    def push_token(self, tok: int) -> TokenProcessingStatus:
        '''
        Push a token. Returns a TokenProcessingStatus indicating the processing state.
        '''
        self.token_count += 1

        if self.guidance_interpreter:
            valid = self.guidance_interpreter.commit_token(tok)
            if not valid:
                raise ValueError(f"Schema violation at token {tok} ('{self.tokenizer.decode([tok])}')")

        if self.max_tokens and self.token_count >= self.max_tokens:
            self.is_finished = True
            return TokenProcessingStatus.MAX_TOKENS_REACHED
        if tok in self.stop_tokens:
            self.is_finished = True
            return TokenProcessingStatus.STOP_TOKEN_REACHED

        if self.guidance_interpreter and self.guidance_interpreter.has_pending_stop():
            self.is_finished = True
            grammar_stop_reason = self.guidance_interpreter.stop_reason()
            if grammar_stop_reason == "EndOfSentence" or grammar_stop_reason == "NoExtension":
                return TokenProcessingStatus.STOP_TOKEN_REACHED
            elif grammar_stop_reason == "MaxTokensTotal" or grammar_stop_reason == "MaxTokensParser":
                return TokenProcessingStatus.MAX_TOKENS_REACHED
            else:
                raise ValueError(f"Unknown stop reason: {grammar_stop_reason}")

        return TokenProcessingStatus.CONTINUE
    
    def get_token_mask(self) -> Optional[np.ndarray]:
        if self.guidance_interpreter:
            mask, _ = self.guidance_interpreter.compute_mask()
            if mask is not None:
                return np.array(list(mask), dtype="int32")

        return None


class LlamaInferenceEngine:
    def __init__(self, model):
        self.model = model

        tokenizer_path = str((self.model.model_path if self.model.model_path.is_dir() else self.model.model_path.parent))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


    def build_prompt(self, messages):
        def encode_role(role: str):
            return '<|start_header_id|>' + role + '<|end_header_id|>\n\n'

        def encode_message(role: str, content: str):
            return encode_role(role) + content + '<|eot_id|>'

        prompt = ''
        for message in messages:
            prompt += encode_message(message["role"], message["content"])

        prompt += encode_role("assistant")

        return prompt

    def run_inference(self, prompt: str, generation_options: GenerationOptions):
        # For some reason the wrong stop token is loaded. So we hardcode for now.
        stop_tokens = {self.tokenizer.eos_token_id, 128009}
        structured_output = StructuredOutput(self.tokenizer, stop_tokens, generation_options)

        toks = self.tokenizer.encode(prompt)

        start_pos = self.model.prefill(toks)
        last_tok = toks[-1]

        generated_tokens = []

        if generation_options.temperature is None:
            generation_options.temperature = self.model.TEMPERATURE

        while True:
            mask = Tensor(structured_output.get_token_mask())
            # if mask is not None:
            #     print([self.tokenizer.decode([tok]) for tok, val in enumerate(mask) if val == np.int32(200)])
            tok = self.model.generate_next_token(last_tok, mask, start_pos, generation_options.temperature)
            start_pos += 1
            last_tok = tok
            self.model.last_seen_toks.append(tok)

            finish_reason = structured_output.push_token(tok)
            if finish_reason != TokenProcessingStatus.CONTINUE:
                break

            generated_tokens.append(tok)

        return self.tokenizer.decode(generated_tokens), finish_reason

    def run_inference_stream(self, prompt: str, generation_options: GenerationOptions):
        # For some reason the wrong stop token is loaded. So we hardcode for now.
        stop_tokens = {self.tokenizer.eos_token_id, 128009}
        structured_output = StructuredOutput(self.model, stop_tokens, generation_options)
        
        toks = self.tokenizer.encode(prompt)

        start_pos = self.model.prefill(toks)
        last_tok = toks[-1]

        if generation_options.temperature is None:
            generation_options.temperature = self.model.TEMPERATURE

        while True:
            tok = self.model.generate_next_token(last_tok, start_pos, generation_options.temperature)
            start_pos += 1
            last_tok = tok
            self.model.last_seen_toks.append(tok)

            finish_reason = structured_output.push_token(tok)
            if finish_reason != TokenProcessingStatus.CONTINUE:
                break

            yield self.tokenizer.decode([tok])

        return finish_reason