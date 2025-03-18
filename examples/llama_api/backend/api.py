from pathlib import Path
import json, random, time
from bottle import Bottle, request, response, HTTPResponse, abort, static_file

class LlamaAPI:
    def __init__(self, model):
        """Initialize the API server with a loaded model instance"""
        self.model = model
        self.app = Bottle()
        self.setup_routes()
        
    def setup_routes(self):
        """Set up all the API routes"""
        # CORS Handling
        cors_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Authorization",
            "Access-Control-Allow-Credentials": "true",
        }
        
        @self.app.hook("before_request")
        def handle_options():
            if request.method == "OPTIONS": 
                raise HTTPResponse(headers=cors_headers)
                
        @self.app.hook("after_request")
        def enable_cors():
            for key, value in cors_headers.items(): 
                response.set_header(key, value)
        
        # Static file serving (for web UI)
        @self.app.route("/<filename>")
        def server_static(filename): 
            return static_file(filename, root=(Path(__file__).parent.parent.parent / "tinychat").as_posix())
            
        @self.app.route("/assets/<filename:path>")
        def server_assets(filename): 
            return static_file(filename, root=(Path(__file__).parent.parent.parent / "tinychat" / "assets").as_posix())
            
        @self.app.route("/")
        def index():
            return static_file("index.html", root=(Path(__file__).parent.parent.parent / "tinychat").as_posix())
        
        # API Endpoints
        @self.app.get("/v1/models")
        def models():
            model_path = str(self.model.model_path)
            return json.dumps({
                "object": "list",
                "data": [
                    {
                        "id": model_path,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "user"
                    }
                ]
            })
        
        @self.app.post("/v1/internal/token-count")
        def token_count():
            rjson = json.loads(request.body.read())
            return json.dumps(len(self.model.tokenizer.encode(rjson.get("text", ""))))
            
        @self.app.post("/v1/token/encode")
        def token_encode():
            rjson = json.loads(request.body.read())
            return json.dumps(self.model.tokenizer.encode(rjson.get("text", "")))
        
        @self.app.post("/v1/completions")
        def completions():
            rjson = json.loads(request.body.read())

            # check if we are streaming
            if rjson.get("stream", False):
                response.content_type = "text/event-stream"
                response.set_header("Cache-Control", "no-cache")
            else: 
                abort(400, "streaming required")

            toks = [self.model.tokenizer.bos_id] + self.model.tokenizer.encode(rjson.get("prompt", ""), allow_special=True)

            start_pos = self.model.prefill(toks[:-1])
            last_tok = toks[-1]
            while True:
                tok = self.model.generate_next_token(last_tok, start_pos, rjson.get("temperature", self.model.TEMPERATURE))
                start_pos += 1
                last_tok = tok
                if tok in self.model.tokenizer.stop_tokens: 
                    break

                res = {
                    "choices": [{
                        "text": self.model.tokenizer.decode([tok]),
                    }]
                }
                yield f"data: {json.dumps(res)}\n\n"
        
        @self.app.post("/v1/chat/token/encode")
        def chat_token_encode():
            rjson = json.loads(request.body.read())
            if "messages" not in rjson: 
                abort(400, "messages required")
                
            toks = [self.model.tokenizer.bos_id]
            for message in rjson["messages"]:
                toks += self.model.encode_message(message["role"], message["content"])
                
            if len(rjson["messages"]) > 0 and message["role"] == "user":
                toks += self.model.encode_role("assistant")
                
            return json.dumps(toks)
        
        @self.app.post("/v1/chat/completions")
        def chat_completions():
            rjson = json.loads(request.body.read())
            if "messages" not in rjson: 
                abort(400, "messages required")

            # Setup for both streaming and non-streaming
            streaming = rjson.get("stream", False)
            if streaming:
                response.content_type = "text/event-stream"
                response.set_header("Cache-Control", "no-cache")

            toks = [self.model.tokenizer.bos_id]
            for message in rjson["messages"]:
                toks += self.model.encode_message(message["role"], message["content"])
                
            # ensure that the last message was a user message
            if message["role"] != "user": 
                abort(400, "last message must be a user message")
                
            toks += self.model.encode_role("assistant")

            random_id = random.randbytes(16).hex()

            start_pos = self.model.prefill(toks[:-1])
            last_tok = toks[-1]
            self.model.last_seen_toks.append(last_tok)
            
            # For non-streaming, collect all generated tokens
            if not streaming:
                generated_tokens = []
                
                while True:
                    tok = self.model.generate_next_token(last_tok, start_pos, rjson.get("temperature", self.model.TEMPERATURE))
                    start_pos += 1
                    last_tok = tok
                    self.model.last_seen_toks.append(tok)
                    
                    if tok in self.model.tokenizer.stop_tokens: 
                        break
                        
                    generated_tokens.append(tok)
                
                # Return a single response with the complete message
                yield json.dumps({
                    "id": random_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": str(self.model.model_path),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": self.model.tokenizer.decode(generated_tokens),
                        },
                        "finish_reason": "stop",
                    }]
                })

            else:
                # Streaming response (existing implementation)
                while True:
                    tok = self.model.generate_next_token(last_tok, start_pos, rjson.get("temperature", self.model.TEMPERATURE))
                    start_pos += 1
                    last_tok = tok
                    self.model.last_seen_toks.append(tok)
                    
                    if tok in self.model.tokenizer.stop_tokens: 
                        break

                    res = {
                        "id": random_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": str(self.model.model_path),
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": self.model.tokenizer.decode([tok]),
                            },
                            "finish_reason": None,
                        }]
                    }
                    yield f"data: {json.dumps(res)}\n\n"

                res = {
                    "id": random_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": str(self.model.model_path),
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }]
                }
                yield f"data: {json.dumps(res)}\n\n"
    
    def run(self, host="0.0.0.0", port=7776, debug=False):
        """Start the API server"""
        self.app.run(host=host, port=port, debug=debug) 