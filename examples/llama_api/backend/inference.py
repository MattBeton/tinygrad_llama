from enum import Enum
from typing import List
from transformers import AutoTokenizer


class TokenProcessingStatus(Enum):
    CONTINUE = None  # Continue processing tokens
    MAX_TOKENS_REACHED = "length"
    STOP_TOKEN_REACHED = "stop"

class StructuredOutput:
    def __init__(self, model, stop_tokens, max_tokens=None):
        self.model = model

        self.token_count = 0
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens

    def push_token(self, tok: int):
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

        tokenizer_path = str((self.model.model_path if self.model.model_path.is_dir() else self.model.model_path.parent))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def build_prompt(self, messages):
        def encode_role(role: str):
            return '<|start_header_id|>' + role + '<|end_header_id|>\n\n'

        def encode_message(role: str, content: str):
            return encode_role(role) + content + '<|eot_id|>'

        # prompt = '<|begin_of_text|>'
        prompt = ''
        for message in messages:
            prompt += encode_message(message["role"], message["content"])

        prompt += encode_role("assistant")

        return prompt

    def run_inference(self, prompt: str, temperature=None, max_tokens=None):
        # For some reason the wrong stop token is loaded. So we hardcode for now.
        stop_tokens = {self.tokenizer.eos_token_id, 128009}
        structured_output = StructuredOutput(self.model, stop_tokens, max_tokens)

        toks = self.tokenizer.encode(prompt)

        start_pos = self.model.prefill(toks)
        last_tok = toks[-1]

        generated_tokens = []

        if temperature is None:
            temperature = self.model.TEMPERATURE

        while True:
            tok = self.model.generate_next_token(last_tok, start_pos, temperature)
            start_pos += 1
            last_tok = tok
            self.model.last_seen_toks.append(tok)

            finish_reason = structured_output.push_token(tok)
            if finish_reason != TokenProcessingStatus.CONTINUE:
                break

            generated_tokens.append(tok)

        return self.tokenizer.decode(generated_tokens), finish_reason

    def run_inference_stream(self, prompt: str, temperature=None, max_tokens=None):
        structured_output = StructuredOutput(self.model, self.tokenizer.stop_tokens, max_tokens)

        toks = self.tokenizer.encode(prompt, allow_special=True)

        start_pos = self.model.prefill(toks)
        last_tok = toks[-1]
        self.model.last_seen_toks.append(last_tok)

        if temperature is None:
            temperature = self.model.TEMPERATURE

        while True:
            tok = self.model.generate_next_token(last_tok, start_pos, temperature)
            start_pos += 1
            last_tok = tok
            self.model.last_seen_toks.append(tok)

            finish_reason = structured_output.push_token(tok)
            if finish_reason != TokenProcessingStatus.CONTINUE:
                break

            yield self.tokenizer.decode([tok])

        return finish_reason