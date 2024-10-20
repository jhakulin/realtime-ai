from typing import Any, Dict, List, Tuple, get_origin

import inspect
import json
import logging

from openai.types import FunctionDefinition
from openai.types.beta.threads import RequiredActionFunctionToolCall


# Define type_map to translate Python type annotations to JSON Schema types
type_map = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "bytes": "string",  # Typically encoded as base64-encoded strings in JSON
    "NoneType": "null",
    "datetime": "string",  # Use format "date-time"
    "date": "string",  # Use format "date"
    "UUID": "string",  # Use format "uuid"
}


def _map_type(annotation) -> str:

    if annotation == inspect.Parameter.empty:
        return "string"  # Default type if annotation is missing

    origin = get_origin(annotation)

    if origin in {list, List}:
        return "array"
    elif origin in {dict, Dict}:
        return "object"
    elif hasattr(annotation, "__name__"):
        return type_map.get(annotation.__name__, "string")
    elif isinstance(annotation, type):
        return type_map.get(annotation.__name__, "string")

    return "string"  # Fallback to "string" if type is unrecognized


def _serialize_function_definition(function_def: FunctionDefinition) -> Dict[str, Any]:
    """
    Serialize a FunctionDefinition object to a dictionary.

    :param function_def: The FunctionDefinition object to serialize.
    :return: A dictionary representation of the function definition with type 'function'.
    """
    return {
        "type": "function",
        "name": function_def.name,
        "description": function_def.description,
        "parameters": function_def.parameters
    }


class FunctionTool:
    """
    A tool that executes user-defined functions.
    """

    def __init__(self, functions: Dict[str, Any]):
        """
        Initialize FunctionTool with a dictionary of functions.

        :param functions: A dictionary where keys are function names and values are the function objects.
        """
        self._functions = functions
        self._definitions = self._build_function_definitions(functions)

    def _build_function_definitions(self, functions: Dict[str, Any]) -> List[FunctionDefinition]:
        specs = []
        for name, func in functions.items():
            sig = inspect.signature(func)
            params = sig.parameters
            docstring = inspect.getdoc(func)
            description = docstring.split("\n")[0] if docstring else "No description"

            properties = {}
            for param_name, param in params.items():
                param_type = _map_type(param.annotation)
                param_description = param.annotation.__doc__ if param.annotation != inspect.Parameter.empty else None
                properties[param_name] = {"type": param_type, "description": param_description}

            function_def = FunctionDefinition(
                name=name,
                description=description,
                parameters={"type": "object", "properties": properties, "required": list(params.keys())},
            )
            specs.append(function_def)
        return specs

    def execute(self, function_name, arguments) -> Any:
        try:
            function = self._functions[function_name]
            parsed_arguments = json.loads(arguments)
            return function(**parsed_arguments) if parsed_arguments else function()
        except TypeError as e:
            logging.error(f"Error executing function '{function_name}': {e}")
            raise

    @property
    def definitions(self) -> List[Dict[str, Any]]:
        """
        Get the function definitions serialized as dictionaries.

        :return: A list of dictionary representations of the function definitions.
        """
        return [_serialize_function_definition(fd) for fd in self._definitions]
