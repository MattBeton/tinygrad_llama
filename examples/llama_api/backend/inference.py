from typing import List

class StructuredOutput:
    def __init__(self, model, stop_tokens, max_tokens=None):
        self.model = model

        self.token_count = 0
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens

    def push_token(self, tok: int):
        '''
        Push a token. Currently returns a boolean indicating whether we are finished.
        '''
        self.token_count += 1

        if self.max_tokens and self.token_count > self.max_tokens:
            return False

        if tok in self.stop_tokens:
            return False
        
        return True

class LlamaInferenceEngine:
    def __init__(self, model):
        self.model = model

    def build_prompt(self, messages):
        def encode_role(role: str):
            return '<|start_header_id|>' + role + '<|end_header_id|>\n\n'

        def encode_message(role: str, content: str):
            return encode_role(role) + content + '<|eot_id|>'

        prompt = '<|begin_of_text|>'
        for message in messages:
            prompt += encode_message(message["role"], message["content"])

        prompt += encode_role("assistant")

        return prompt

    def run_inference(self, prompt: str, temperature=None):
        structured_output = StructuredOutput(self.model, self.model.tokenizer.stop_tokens, None)

        toks = self.model.tokenizer.encode(prompt, allow_special=True)

        start_pos = self.model.prefill(toks)
        last_tok = toks[-1]
        self.model.last_seen_toks.append(last_tok)

        generated_tokens = []

        if temperature is None:
            temperature = self.model.TEMPERATURE

        while True:
            tok = self.model.generate_next_token(last_tok, start_pos, temperature)
            start_pos += 1
            last_tok = tok
            self.model.last_seen_toks.append(tok)

            if structured_output.push_token(tok) == False:
                break

            generated_tokens.append(tok)

        return self.model.tokenizer.decode(generated_tokens)

    def run_inference_stream(self, prompt: str, temperature=None):
        structured_output = StructuredOutput(self.model, self.model.tokenizer.stop_tokens, None)

        toks = self.model.tokenizer.encode(prompt, allow_special=True)

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

            if structured_output.push_token(tok) == False:
                break

            yield self.model.tokenizer.decode([tok])