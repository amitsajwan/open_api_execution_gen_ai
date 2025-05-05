# filename: utils.py
import hashlib
import json
import logging
import os # Added for cache directory path
from typing import Any, Dict, List, Optional, Tuple, Type # Import Type
from pydantic import BaseModel, ValidationError

# Assuming models.py defines GraphOutput, Node, Edge
# Need to import BaseModel if expected_model type hint is used directly
# from pydantic import BaseModel
# Assuming models.py is available
try:
    from models import GraphOutput, Node, Edge, BaseModel # Import BaseModel here
except ImportError:
    # Define dummy classes if models.py is not found, to allow utils.py to load
    logger.warning("Could not import models from models.py. Using dummy classes.")
    class GraphOutput: pass
    class Node:
        @property
        def effective_id(self):
            return getattr(self, 'operationId', 'dummy_node')
    class Edge: pass
    class BaseModel: pass # Dummy for type hint if needed


# Module-level logger
logger = logging.getLogger(__name__)

# --- Persistent Caching Setup ---
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".openapi_cache") # Cache in subdirectory
try:
    SCHEMA_CACHE = diskcache.Cache(CACHE_DIR)
    logger.info(f"Initialized persistent schema cache at: {CACHE_DIR}")
except Exception as e:
    logger.error(f"Failed to initialize disk cache at {CACHE_DIR}: {e}. Caching will not work.", exc_info=True)
    SCHEMA_CACHE = None # Disable cache if initialization fails

def get_cache_key(spec_text: str) -> str:
    """Generates a cache key based on the hash of the spec text."""
    return hashlib.md5(spec_text.encode('utf-8')).hexdigest()

def load_cached_schema(cache_key: str) -> Optional[Dict[str, Any]]:
    """Loads a parsed schema from the persistent cache if it exists."""
    if SCHEMA_CACHE is None:
        logger.warning("Schema cache is not available.")
        return None
    try:
        schema = SCHEMA_CACHE.get(cache_key)
        if schema:
            logger.debug(f"Cache hit for key: {cache_key}")
            return schema
        else:
            logger.debug(f"Cache miss for key: {cache_key}")
            return None
    except Exception as e:
        logger.error(f"Error loading schema from cache (key: {cache_key}): {e}", exc_info=True)
        return None

def save_schema_to_cache(cache_key: str, schema: Dict[str, Any]):
    """Saves a parsed schema to the persistent cache."""
    if SCHEMA_CACHE is None:
        logger.warning("Schema cache is not available. Cannot save.")
        return
    try:
        SCHEMA_CACHE.set(cache_key, schema)
        logger.debug(f"Saved schema to cache with key: {cache_key}")
    except Exception as e:
        logger.error(f"Error saving schema to cache (key: {cache_key}): {e}", exc_info=True)


# --- Graph Utilities ---

def check_for_cycles(graph: GraphOutput) -> Tuple[bool, str]:
    """
    Checks if the given execution graph is a Directed Acyclic Graph (DAG).
    Uses the effective_id of nodes for checks.
    Returns a tuple: (is_dag, cycle_message).
    """
    # Ensure graph and graph.nodes are valid before proceeding
    if not isinstance(graph, GraphOutput) or not isinstance(graph.nodes, list):
         logger.warning("Invalid graph object passed to check_for_cycles.")
         return False, "Invalid graph structure provided." # Treat invalid graph as potentially cyclic

    # Use effective_id for unique identification in graph checks
    node_ids = {node.effective_id for node in graph.nodes if isinstance(node, Node)} # Use effective_id
    if not node_ids:
        return True, "Graph is empty or has no valid node identifiers."

    # Build adjacency list using effective_id
    adj: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
    if not isinstance(graph.edges, list):
         logger.warning("Graph edges attribute is not a list in check_for_cycles.")
         return False, "Invalid graph edges structure." # Treat invalid graph as potentially cyclic

    for edge in graph.edges:
        if isinstance(edge, Edge) and hasattr(edge, 'from_node') and hasattr(edge, 'to_node'):
            # Edge nodes must exist as effective_ids
            if edge.from_node in adj and edge.to_node in node_ids:
                 adj[edge.from_node].append(edge.to_node)
            else:
                 logger.warning(f"Skipping invalid edge in cycle check: {edge.from_node} -> {edge.to_node} (One or both nodes may not exist as effective_id)")
        else:
             logger.warning(f"Skipping invalid edge object in cycle check: {edge}")


    visited: Dict[str, bool] = {node_id: False for node_id in node_ids}
    recursion_stack: Dict[str, bool] = {node_id: False for node_id in node_ids}

    for node_id in node_ids: # Iterate through all valid node effective IDs
        if not visited[node_id]:
            # Use a helper function for the recursive DFS part
            if _dfs_cycle_check(node_id, visited, recursion_stack, adj):
                # Cycle detected by helper
                # Note: Reconstructing the exact cycle path here is complex,
                # the helper function just returns True if a cycle is found.
                return False, f"Cycle detected involving node '{node_id}' or its descendants."

    return True, "No cycles detected."

def _dfs_cycle_check(node_id: str, visited: Dict[str, bool], recursion_stack: Dict[str, bool], adj: Dict[str, List[str]]) -> bool:
    """Recursive helper for DFS cycle detection."""
    visited[node_id] = True
    recursion_stack[node_id] = True

    # Check neighbors
    for neighbor_id in adj.get(node_id, []):
        if not visited[neighbor_id]:
            if _dfs_cycle_check(neighbor_id, visited, recursion_stack, adj):
                return True # Cycle found in deeper recursion
        elif recursion_stack[neighbor_id]:
            # Cycle detected: neighbor is already in the current recursion stack
            logger.error(f"Cycle detected: Edge from '{node_id}' to '{neighbor_id}' completes a cycle.")
            return True

    # Remove node from recursion stack as we backtrack
    recursion_stack[node_id] = False
    return False


# --- LLM Call Helper ---

def llm_call_helper(llm: Any, prompt: Any) -> str:
    """
    Helper function to make an LLM call with basic logging and error handling.
    Accepts string or structured prompts (like lists of messages).

    Args:
        llm: The LLM instance with an 'invoke' method.
        prompt: The prompt (string or structured) to send to the LLM.

    Returns:
        The response content (usually text) from the LLM.

    Raises:
        Exception: Re-raises exceptions from the llm.invoke call.
    """
    prompt_repr = str(prompt)[:500] # Get a string representation for logging
    logger.debug(f"Calling LLM with prompt (first 500 chars): {prompt_repr}...")
    try:
        # Assuming the LLM's invoke method returns an object with a 'content' attribute
        # (like AIMessage or ChatCompletion). Adjust if your LLM client returns differently.
        response_obj = llm.invoke(prompt)

        # Extract content based on typical LangChain patterns
        if hasattr(response_obj, 'content'):
             response_content = response_obj.content
        elif isinstance(response_obj, str):
             response_content = response_obj # Handle cases where invoke directly returns a string
        else:
             logger.warning(f"LLM response object type ({type(response_obj)}) has no 'content' attribute and is not a string. Returning raw object representation.")
             response_content = str(response_obj)

        # Ensure response is a string
        if not isinstance(response_content, str):
             logger.warning(f"LLM response content is not a string ({type(response_content)}). Converting to string.")
             response_content = str(response_content)

        logger.debug(f"LLM call successful. Response (first 500 chars): {response_content[:500]}...")
        return response_content
    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        raise # Re-raise the exception so the calling node can handle it

# --- JSON Parsing Helper ---

# Renamed function for clarity when using optional model validation
def parse_llm_json_output_with_model(llm_output: str, expected_model: Optional[Type[BaseModel]] = None) -> Any:
    """
    Parses JSON output from an LLM string, handling potential markdown formatting
    and optional Pydantic model validation.

    Args:
        llm_output: The raw string output from the LLM.
        expected_model: An optional Pydantic model class (type) to validate against.

    Returns:
        The parsed JSON object (dict, list, or Pydantic model instance),
        or None if parsing or validation fails.
    """
    if not isinstance(llm_output, str):
        logger.error(f"Cannot parse non-string LLM output as JSON. Type: {type(llm_output)}")
        return None

    logger.debug(f"Attempting to parse LLM output as JSON (first 500 chars): {llm_output[:500]}...")
    json_block = llm_output.strip()

    # Attempt to extract JSON block if LLM wrapped it in markdown code fences
    if json_block.startswith('```json') and json_block.endswith('```'):
        json_block = json_block[7:-3].strip() # Remove ```json and ```
        logger.debug("Extracted JSON block from markdown.")
    elif json_block.startswith('```') and json_block.endswith('```'):
         # Handle generic ``` fences
         json_block = json_block[3:-3].strip()
         logger.debug("Extracted JSON block from generic markdown.")

    # Sometimes LLMs might just output the JSON without fences
    # Basic check if it looks like JSON
    if not (json_block.startswith('{') and json_block.endswith('}')) and \
       not (json_block.startswith('[') and json_block.endswith(']')):
        logger.warning("LLM output doesn't start/end with {} or []. Parsing might fail.")
        # Consider adding more heuristics here if needed

    try:
        # Attempt to parse the JSON
        parsed_data = json.loads(json_block)
        logger.debug("Successfully parsed JSON.")

        # If an expected Pydantic model class is provided, validate the data
        # Check if expected_model is actually a subclass of BaseModel
        if expected_model and issubclass(expected_model, BaseModel):
            logger.debug(f"Validating parsed JSON against model: {expected_model.__name__}")
            try:
                # Validate using the provided model class
                validated_data = expected_model.model_validate(parsed_data)
                logger.debug("JSON validated successfully against model.")
                return validated_data # Return the validated model instance
            except ValidationError as e:
                logger.error(f"Pydantic validation failed against model {expected_model.__name__}: {e}", exc_info=True)
                # Optionally, log the problematic data: logger.error(f"Data: {parsed_data}")
                return None # Validation failed
        else:
            # No validation requested or invalid model provided, return raw parsed data
            if expected_model:
                 logger.warning(f"expected_model ({expected_model}) is not a valid Pydantic model type. Skipping validation.")
            return parsed_data

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}. Problematic text (approx): '{json_block[e.pos-10:e.pos+10]}'", exc_info=False) # Log context around error
        logger.debug(f"Full text attempted parsing: {json_block}") # Log full text on debug level
        return None
    except Exception as e:
        # Catch any other unexpected errors (e.g., during validation if not ValidationError)
        logger.error(f"An unexpected error occurred during JSON parsing/validation: {e}", exc_info=True)
        return None
