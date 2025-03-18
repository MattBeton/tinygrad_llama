from typing import List

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

    def get_tokens(self, messages):
        prompt = self.build_prompt(messages)

        return 

    def run_inference(self, prompt: str, temperature=None):
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

            if tok in self.model.tokenizer.stop_tokens:
                break

            generated_tokens.append(tok)

        return self.model.tokenizer.decode(generated_tokens)

