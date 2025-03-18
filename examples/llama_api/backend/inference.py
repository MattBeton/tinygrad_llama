from enum import Enum
from typing import List, Dict
from llguidance import LLInterpreter
from llguidance.hf import from_tokenizer as llg_from_tokenizer

class TokenProcessingStatus(Enum):
    CONTINUE = None  # Continue processing tokens
    MAX_TOKENS_REACHED = "length"
    STOP_TOKEN_REACHED = "stop"

class GenerationOptions:
    def __init__(self, temperature=None, max_tokens=None, grammar=None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.grammar = grammar

class StructuredOutput:
    def __init__(self, model, stop_tokens, generation_options: GenerationOptions):
        self.model = model

        self.token_count = 0
        self.max_tokens = generation_options.max_tokens
        self.stop_tokens = stop_tokens
        self.grammar = generation_options.grammar

        # tokenizer = llg_from_tokenizer(self.model.tokenizer)

    def push_token(self, tok: int) -> TokenProcessingStatus:
        '''
        Push a token. Returns a TokenProcessingStatus indicating the processing state.
        '''
        self.token_count += 1

        if self.max_tokens and self.token_count >= self.max_tokens:
            return TokenProcessingStatus.MAX_TOKENS_REACHED

        if tok in self.stop_tokens:
            return TokenProcessingStatus.STOP_TOKEN_REACHED
        
        return TokenProcessingStatus.CONTINUE

class LlamaInferenceEngine:
    def __init__(self, model):
        self.model = model

    def run_inference(self, prompt: str, generation_options: GenerationOptions):
        structured_output = StructuredOutput(self.model, self.model.tokenizer.stop_tokens, generation_options)

        toks = self.model.tokenizer.encode(prompt, allow_special=True)

        start_pos = self.model.prefill(toks)
        last_tok = toks[-1]
        self.model.last_seen_toks.append(last_tok)

        generated_tokens = []

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

            generated_tokens.append(tok)

        return self.model.tokenizer.decode(generated_tokens), finish_reason

    def run_inference_stream(self, prompt: str, generation_options: GenerationOptions):
        structured_output = StructuredOutput(self.model, self.model.tokenizer.stop_tokens, generation_options)

        toks = self.model.tokenizer.encode(prompt, allow_special=True)

        start_pos = self.model.prefill(toks)
        last_tok = toks[-1]
        self.model.last_seen_toks.append(last_tok)

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

            yield self.model.tokenizer.decode([tok])

        return finish_reason