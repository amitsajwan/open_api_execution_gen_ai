import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

# Assuming models.py defines GraphOutput, Node, Edge
from models import GraphOutput, Node, Edge

# Module-level logger
logger = logging.getLogger(__name__)

# --- Caching Utilities ---

SCHEMA_CACHE: Dict[str, Dict[str, Any]] = {}

def get_cache_key(spec_text: str) -> str:
    """Generates a cache key based on the hash of the spec text."""
    return hashlib.md5(spec_text.encode('utf-8')).hexdigest()

def load_cached_schema(cache_key: str) -> Optional[Dict[str, Any]]:
    """Loads a parsed schema from the cache if it exists."""
    schema = SCHEMA_CACHE.get(cache_key)
    if schema:
        logger.debug(f"Cache hit for key: {cache_key}")
    else:
        logger.debug(f"Cache miss for key: {cache_key}")
    return schema

def save_schema_to_cache(cache_key: str, schema: Dict[str, Any]):
    """Saves a parsed schema to the cache."""
    SCHEMA_CACHE[cache_key] = schema
    logger.debug(f"Saved schema to cache with key: {cache_key}")

# --- Graph Utilities ---

def check_for_cycles(graph: GraphOutput) -> Tuple[bool, str]:
    """
    Checks if the given execution graph is a Directed Acyclic Graph (DAG).
    Returns a tuple: (is_dag, cycle_message).
    """
    if not graph or not graph.nodes:
        return True, "Graph is empty or has no nodes."

    # Build adjacency list
    adj: Dict[str, List[str]] = {node.operationId: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.from_node in adj and edge.to_node in adj: # Only add valid edges
             adj[edge.from_node].append(edge.to_node)
        else:
             logger.warning(f"Skipping invalid edge in cycle check: {edge.from_node} -> {edge.to_node}")


    visited: Dict[str, bool] = {node.operationId: False for node in graph.nodes}
    recursion_stack: Dict[str, bool] = {node.operationId: False for node in graph.nodes}
    cycle_path: List[str] = []

    def dfs(node_id: str) -> bool:
        visited[node_id] = True
        recursion_stack[node_id] = True
        cycle_path.append(node_id)

        for neighbor_id in adj.get(node_id, []):
            if not visited[neighbor_id]:
                if dfs(neighbor_id):
                    return True
            elif recursion_stack[neighbor_id]:
                # Cycle detected
                cycle_start_index = cycle_path.index(neighbor_id)
                cycle = " -> ".join(cycle_path[cycle_start_index:]) + f" -> {neighbor_id}"
                logger.error(f"Cycle detected: {cycle}")
                return True

        cycle_path.pop()
        recursion_stack[node_id] = False
        return False

    # Perform DFS from each node
    for node_id in adj.keys():
        if not visited[node_id]:
            if dfs(node_id):
                return False, f"Cycle detected: {' -> '.join(cycle_path)}" # Return False and the detected cycle path

    return True, "No cycles detected."

# --- LLM Call Helper ---

def llm_call_helper(llm: Any, prompt: str) -> str:
    """
    Helper function to make an LLM call with basic logging.

    Args:
        llm: The LLM instance with an 'invoke' method.
        prompt: The prompt to send to the LLM.

    Returns:
        The response text from the LLM.
    """
    logger.debug(f"Calling LLM with prompt (first 500 chars): {prompt[:500]}...")
    try:
        response = llm.invoke(prompt)
        logger.debug(f"LLM call successful. Response (first 500 chars): {response[:500]}...")
        return response
    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        raise # Re-raise the exception so the calling node can handle it

# --- JSON Parsing Helper ---

def parse_llm_json_output(llm_output: str, expected_model: Optional[BaseModel] = None) -> Any:
    """
    Parses JSON output from an LLM, handling potential markdown formatting
    and optional Pydantic model validation.

    Args:
        llm_output: The raw string output from the LLM.
        expected_model: An optional Pydantic model to validate the parsed JSON against.

    Returns:
        The parsed JSON object (dict or list), or None if parsing fails.
    """
    logger.debug(f"Attempting to parse LLM output as JSON (first 500 chars): {llm_output[:500]}...")
    try:
        # Attempt to find JSON block if LLM wrapped it in markdown
        if '```json' in llm_output:
            json_block = llm_output.split('```json', 1)[1].split('```', 1)[0]
            logger.debug("Extracted JSON block from markdown.")
        else:
            json_block = llm_output
            logger.debug("Assuming raw output is JSON.")

        # Attempt to parse the JSON
        parsed_data = json.loads(json_block)
        logger.debug("Successfully parsed JSON.")

        # If an expected model is provided, validate the data
        if expected_model:
            logger.debug(f"Validating parsed JSON against model: {expected_model.__name__}")
            validated_data = expected_model.model_validate(parsed_data)
            logger.debug("JSON validated successfully against model.")
            return validated_data # Return the validated model instance
        else:
            return parsed_data # Return the raw parsed data

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}", exc_info=True)
        return None
    except ValidationError as e:
        logger.error(f"JSON validation failed against model {expected_model.__name__}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSON parsing/validation: {e}", exc_info=True)
        return None

# --- Placeholder for JSONPath Extraction (Moved to graph.py for now) ---
# def extract_data_with_jsonpath(data: Any, path: str) -> Any:
#     """
#     Placeholder function to extract data from a dictionary/list using a simple path.
#     A real implementation would use a library like jsonpath-ng.
#     Example path: '$.items[0].id'
#     """
#     # Implementation is now in graph.py
#     pass
