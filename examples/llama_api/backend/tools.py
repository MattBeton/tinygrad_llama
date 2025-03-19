from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel
import json
from .grammars import lark_grammar
import re

class UnplacedToolCall(BaseModel):
  name: str
  arguments: str

class ToolDefinition(BaseModel):
    """
    This model maps to elements of the tools array in the request body.
    """
    class FunctionDefinition(BaseModel):
        name: str
        description: Optional[str]
        parameters: Optional[dict[str, Any]]
        strict: Optional[bool] = False

    type: Literal["function"]
    function: FunctionDefinition    

class ToolParser:
    def __init__(self):
        pass

    def build_prompt(self, messages: List[Dict[str, str]], tools: List[Dict[str, str]]) -> str:
        pass

    def is_start_of_tool_section(self, token: int):
        pass

    def to_grammar(self, tools: list[ToolDefinition], required: bool, parallel_tool_calling: bool) -> str:
        pass

    def parse_complete(self, content: str, parallel_tool_calling: bool = False) -> list[UnplacedToolCall]:
        pass

class LlamaToolParser(ToolParser):
    def __init__(self):
        pass

    def build_prompt(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> str:
        def encode_role(role: str):
            return '<|start_header_id|>' + role + '<|end_header_id|>\n\n'

        def encode_message(role: str, content: str):
            return encode_role(role) + content + '<|eot_id|>'

        if tools:
            tools = [tool.model_dump() for tool in tools]
            function_definitions = json.dumps(tools, indent=2)

            system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. 
            Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
            If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
            also point it out. You should only return the function call in tools call sections.

            If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n
            You SHOULD NOT include any other text in the response.

            Here is a list of functions in JSON format that you can invoke.\n\n{functions}\n""".format(functions=function_definitions)

            messages.insert(0, {"role": "system", "content": system_prompt})

        prompt = ''
        for message in messages:
            prompt += encode_message(message["role"], message["content"])

        prompt += encode_role("assistant")

        return prompt

    def to_grammar(self, tools: list[ToolDefinition], required: bool) -> str:
        """
        Returns a Lark grammar string that describes the new plaintext function call format,
        restricted to the available tool names and their parameters.
        The expected output should be a list of function calls, e.g.:
            
        [func_name1(param1=value1, param2=value2), func_name2(arg)]
        """
        # Generate the list of valid function names
        function_names = [f'"{tool.function.name}"' for tool in tools]
        function_names_str = " | ".join(function_names)
        
        # Generate parameter rules for each function
        param_rules = []
        for tool in tools:
            fname = tool.function.name
            if tool.function.parameters and "properties" in tool.function.parameters:
                # Get valid parameter names for this function
                param_names = list(tool.function.parameters["properties"].keys())
                param_names_str = " | ".join(f'"{name}"' for name in param_names)
                
                # Create a specific parameter rule for this function
                param_rules.append(f'{fname}_params: {fname}_param ("," {fname}_param)*')
                param_rules.append(f'{fname}_param: ({param_names_str}) "=" PARAM_VALUE')
            else:
                # If no parameters defined, create an empty rule
                param_rules.append(f'{fname}_params: ')

        param_rules_str = "\n".join(param_rules)

        return lark_grammar(f"""
    %llguidance {{}}

    start: {"fun_calls" if required else "TEXT | fun_calls"}
    TEXT: /[^\\[](.|\n)*/
    fun_calls: "[" fun_call ("," fun_call)* "]"
    fun_call: FUNCTION_NAME "(" [function_parameters] ")"
    FUNCTION_NAME: {function_names_str}

    ?function_parameters: {" | ".join(f'{tool.function.name}_params' for tool in tools)}

    {param_rules_str}

    PARAM_VALUE: /[^,()\\]]+/
        """.strip())

#     def to_grammar(self, tools: list[ToolDefinition], required: bool) -> str:
#         """
#         Returns a Lark grammar string that describes the new plaintext function call format.
#         The expected output should be a list of function calls, e.g.:
        
#           [func_name1(param1=value1, param2=value2), func_name2(arg)]
#         """
#         return lark_grammar(f"""
# %llguidance {{}}

# start: {"fun_calls" if required else "TEXT | fun_calls"}
# TEXT: /[^\\[](.|\n)*/
# fun_calls: "[" fun_call ("," fun_call)* "]"
# fun_call: FUNCTION_NAME "(" [parameters] ")"
# FUNCTION_NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
# ?parameters: parameter ("," parameter)*
# ?parameter: PARAM_NAME "=" PARAM_VALUE   -> key_value
#          | PARAM_VALUE                   -> value_only
# PARAM_NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
# PARAM_VALUE: /[^,()\\]]+/
#         """.strip())

    def parse_complete(self, content: str) -> list[UnplacedToolCall]:
        """
        Parses plaintext function call outputs in the form:
          [func_name1(param1=value1, param2=value2), func_name2(arg)]
        Returns a list of UnplacedToolCall objects where the "arguments" field is a JSON-dumped dict.
        If a parameter is provided without a key (as in func_name2(arg)), it is stored with the key "arg".
        """
        tool_calls = []
        content = content.strip()
        # Remove surrounding square brackets if they exist.
        if content.startswith('[') and content.endswith(']'):
            inner = content[1:-1].strip()
        else:
            inner = content

        # Regex pattern to match each function call.
        # It captures the function name and the content between the parentheses.
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\(\s*(.*?)\s*\)'
        matches = re.findall(pattern, inner)
        for func_name, params_str in matches:
            params = {}
            if params_str:
                # Split by commas (assumes no nested commas in parameter values).
                for part in re.split(r',\s*', params_str):
                    if '=' in part:
                        key, value = part.split('=', 1)
                        params[key.strip()] = value.strip()
                    else:
                        # If there is no "=", treat the whole part as a single positional argument.
                        params["arg"] = part.strip()
            tool_calls.append(UnplacedToolCall(
                name=func_name,
                arguments=json.dumps(params)
            ))
        return tool_calls


    def generate_plaintext_tool_call_schema(self, tools: list[ToolDefinition]) -> str:
        """
        Generate a plaintext schema representation for tool calling in the new Llama function calling format.
        
        For each tool, the format is:
        func_name(param1=type1, param2=type2, …)
        
        This function builds a signature for each tool based on its function name and its parameter types,
        then returns a comma-separated list of these signatures enclosed in square brackets.
        """
        tool_signatures = []
        for tool in tools:
            # Extract the function name.
            func_name = tool.function.name
            # Build a simple signature from the parameter definitions.
            params_schema = tool.function.parameters
            if "properties" in params_schema:
                params = []
                for param, details in params_schema["properties"].items():
                    # Use the parameter type as a simple representation.
                    params.append(f"{param}={details.get('type', 'any')}")
                params_str = ", ".join(params)
            else:
                params_str = ""
            tool_signatures.append(f"{func_name}({params_str})")
        # The schema is a list of function call signatures.
        return f"[{', '.join(tool_signatures)}]"

    # def is_start_of_tool_section(self, token: int):
    #     return token == 128010 or token == 5324

    # def to_grammar(self, tools: list[ToolDefinition], required: bool) -> str:
    #     def generate_tool_call_json_schema(tools: list[ToolDefinition], parameter_key: str = "arguments") -> dict[str, Any]:
    #         """
    #         Generate a JSON schema for tool calling. For a given tool name, the schema should have the rough form of:

    #         type ValidToolCall[name] = {
    #         "name": name,
    #         "arguments": tools[name].parameters
    #         }

    #         With the overall schema looking like:

    #         // For each tool in the list
    #         type ValidToolCall = ValidToolCall[name] | ...;

    #         Ie it should be a union of all the tool calls, disjoint from each other by the unqiue "name" field.
    #         """
    #         if len(tools) == 0:
    #             raise ValueError("No tools provided")

    #         schema_variants = []

    #         for tool in tools:
    #             # Create a schema variant for this tool
    #             tool_schema = {
    #                 "type": "object",
    #                 "properties": {
    #                     # TODO: The LLama example on LLGuidance uses "name": { "const": "get_weather" } which might be easier?
    #                     "name": {
    #                         "type": "string",
    #                         "enum": [tool.function.name]
    #                     },
    #                     parameter_key: tool.function.parameters
    #                 },
    #                 "required": ["name", parameter_key],
    #                 "additionalProperties": False
    #             }
    #             schema_variants.append(tool_schema)

    #         # Combine all tool schemas into a oneOf union
    #         if len(schema_variants) == 1:
    #             # Just return the single schema if only one tool
    #             return schema_variants[0]
    #         else:
    #             # Return a union of all tool schemas
    #             return {"oneOf": schema_variants}

    #     # This is lifted from https://github.com/guidance-ai/llguidance/blob/cc83715f/docs/syntax.md#special-tokens
    #     return lark_grammar(f"""
    # %llguidance {{}}

    # start: {"fun_call" if required else "TEXT | fun_call"}
    # TEXT: /[^{{](.|\n)*/
    # fun_call: <|python_tag|> json_body <|eom_id|>
    # json_body: %json{json.dumps(generate_tool_call_json_schema(tools, "parameters"))}
    #     """.strip())

    # def parse_complete(self, content: str, parallel_tool_calling: bool = False) -> list[UnplacedToolCall]:
    #     raise NotImplementedError("Not implemented")
    #     tool_calls = []

    #     for m in re.finditer(r"<\|python_tag\|>(.+)<\|eom_id\|>", content, re.DOTALL):
    #         try:
    #             remapped = json.loads(m.group(1))

    #             # Rename "parameters" to "arguments" as that is the expected format
    #             tool_calls.append(UnplacedToolCall(
    #             name=remapped["name"],
    #             arguments=json.dumps(remapped["parameters"])
    #             ))
    #         except json.JSONDecodeError as e:
    #             print(f"Failed to parse python_tag tool calls: {e}")

    #     return tool_calls

