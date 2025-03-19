from pathlib import Path
import json, random, time
from bottle import Bottle, request, response, HTTPResponse, abort, static_file
from .inference import LlamaInferenceEngine, GenerationOptions
from .tools import LlamaToolParser, ToolDefinition

class LlamaAPI:
    def __init__(self, inference_engine: LlamaInferenceEngine):
        """Initialize the API server with a loaded model instance"""
        self.inference_engine = inference_engine
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
            model_path = str(self.inference_engine.model.model_path)
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
            return json.dumps(len(self.inference_engine.tokenizer.encode(rjson.get("text", ""))))
            
        @self.app.post("/v1/token/encode")
        def token_encode():
            rjson = json.loads(request.body.read())
            return json.dumps(self.inference_engine.tokenizer.encode(rjson.get("text", "")))
        
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

            if rjson.get("tools"):
                tools = [ToolDefinition.model_validate(tool) for tool in rjson.get("tools")]
            else:
                tools = []

            tools_parser = LlamaToolParser()
            prompt = tools_parser.build_prompt(rjson["messages"], tools)
            grammar = tools_parser.to_grammar(tools, rjson.get("strict", False))
            generation_options = GenerationOptions(temperature=rjson.get("temperature"), max_tokens=rjson.get("max_tokens"), grammar=grammar)

            # ensure that the last message was a user message
            if rjson["messages"][-1]["role"] != "user": 
                abort(400, "last message must be a user message")
                
            random_id = random.randbytes(16).hex()

            # For non-streaming, collect all generated tokens
            if not streaming:
                inference_result, finish_reason = self.inference_engine.run_inference(prompt, generation_options)
                
                # Return a single response with the complete message
                yield json.dumps({
                    "id": random_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": str(self.inference_engine.model.model_path),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": inference_result,
                        },
                        "finish_reason": finish_reason.value,
                    }]
                })

            else:
                # Store chunks and get finish_reason from the generator
                chunks = []
                for chunk in self.inference_engine.run_inference_stream(prompt, generation_options):
                    chunks.append(chunk)
                    yield f"data: {json.dumps({
                            'id': random_id,
                            'object': 'chat.completion.chunk',
                            'created': int(time.time()),
                            'model': str(self.inference_engine.model.model_path),
                            'choices': [{
                                'index': 0,
                                'delta': {'content': chunk},
                            }],
                            'finish_reason': None,
                        })}\n\n"

                # Get the finish_reason from the generator
                finish_reason = yield
                
                res = {
                    'id': random_id,
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': str(self.inference_engine.model.model_path),
                    'choices': [{
                        'index': 0,
                        'delta': {},
                        'finish_reason': finish_reason.value,
                    }]
                }
                yield f"data: {json.dumps(res)}\n\n"
    
    def run(self, host="0.0.0.0", port=7776, debug=False):
        """Start the API server"""
        self.app.run(host=host, port=port, debug=debug) 