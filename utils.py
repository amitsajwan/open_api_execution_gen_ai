import os
import json
import logging
import hashlib
from typing import Any, Dict, List, Tuple, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError
from pydantic import BaseModel
import jsonschema # For basic JSON schema validation if needed

# Assuming models.py contains BotState, GraphOutput, Node, Edge
from models import BotState, GraphOutput, Edge

# --- Configuration ---
CACHE_DIR = os.getenv("OPENAPI_CACHE_DIR", "./.openapi_cache")
STATE_DIR = os.getenv("OPENAPI_STATE_DIR", "./.state_cache")
LLM_RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", 3))
LLM_RETRY_MIN_WAIT = int(os.getenv("LLM_RETRY_MIN_WAIT", 1))
LLM_RETRY_MAX_WAIT = int(os.getenv("LLM_RETRY_MAX_WAIT", 10))

# Module-level logger
logger = logging.getLogger(__name__)

# --- Directory Setup ---
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    logger.info(f"Cache directory: {CACHE_DIR}")
    logger.info(f"State directory: {STATE_DIR}")
except OSError as e:
    logger.warning(f"Could not create cache/state directories: {e}")

# --- LLM Interaction Helper ---
@retry(wait=wait_random_exponential(min=LLM_RETRY_MIN_WAIT, max=LLM_RETRY_MAX_WAIT), 
       stop=stop_after_attempt(LLM_RETRY_ATTEMPTS))
def llm_call_helper(llm: Any, prompt: str, **kwargs) -> str:
    """
    Invoke the LLM with retries and exponential backoff on transient errors.
    Assumes the llm object has an `invoke` method (like LangChain LLMs).
    Returns the raw LLM response string or raises after maximum retries.
    """
    try:
        logger.debug(f"Invoking LLM (attempt {llm_call_helper.retry.statistics.get('attempt_number', 1)}). Prompt length: {len(prompt)}")
        # Example using LangChain's invoke structure
        if hasattr(llm, 'invoke'):
             # Check if the LLM expects a list of messages or a simple string
            try:
                # Attempt standard invoke first
                response = llm.invoke(prompt, **kwargs)
                # Handle different response types (content common in LangChain)
                content = getattr(response, 'content', response)
                if not isinstance(content, str):
                     raise TypeError(f"LLM response content is not a string: {type(content)}")
                logger.debug(f"LLM response received. Length: {len(content)}")
                return content
            except TypeError as te:
                 # If invoke fails with type error, maybe it expects messages
                 if "expected str" in str(te).lower() or "expected sequence" in str(te).lower():
                      logger.debug("LLM might expect messages, trying with HumanMessage structure.")
                      # Assuming LangChain's HumanMessage - adjust if using a different library
                      from langchain_core.messages import HumanMessage
                      response = llm.invoke([HumanMessage(content=prompt)], **kwargs)
                      content = getattr(response, 'content', response)
                      if not isinstance(content, str):
                           raise TypeError(f"LLM response content is not a string: {type(content)}")
                      logger.debug(f"LLM response received (as message). Length: {len(content)}")
                      return content
                 else:
                      raise # Re-raise original TypeError if not the expected one
        else:
            raise NotImplementedError("LLM object does not have an 'invoke' method.")

    except RetryError as re:
        logger.error(f"LLM invocation failed after {LLM_RETRY_ATTEMPTS} attempts: {re}", exc_info=True)
        raise # Re-raise the RetryError to signal failure
    except Exception as e:
        logger.error(f"LLM invocation error: {e}", exc_info=True)
        # Depending on the error, decide if it's retryable by tenacity
        # For now, let tenacity handle standard exceptions, raise others
        # You might want to add specific exception types here that shouldn't be retried
        if isinstance(e, (NotImplementedError, TypeError, ValueError)):
             raise # Don't retry programming errors
        raise # Re-raise to allow tenacity to retry if applicable

# --- Schema Caching ---
def get_cache_key(spec_text: str) -> str:
    """Generate a stable cache key (hash) from the spec text."""
    return hashlib.sha256(spec_text.encode('utf-8')).hexdigest()

def load_cached_schema(cache_key: str) -> Optional[Dict[str, Any]]:
    """Load a resolved OpenAPI schema from cache if available."""
    if not cache_key:
        return None
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            logger.info(f"Loaded schema from cache: {cache_path}")
            return schema
        except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to load schema cache file {cache_path}: {e}")
            # Optionally remove corrupted cache file
            try:
                os.remove(cache_path)
            except OSError:
                pass
    return None

def save_schema_to_cache(cache_key: str, schema: Dict[str, Any]):
    """Save resolved OpenAPI schema to cache."""
    if not cache_key:
        logger.warning("Attempted to save schema with no cache key.")
        return
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2) # Use indent for readability
        logger.info(f"Saved schema to cache: {cache_path}")
    except (IOError, TypeError, UnicodeEncodeError) as e:
        logger.warning(f"Failed to write schema cache file {cache_path}: {e}")

# --- State Persistence ---
def get_state_path(session_id: str) -> str:
    """Get the file path for persisting a session's state."""
    return os.path.join(STATE_DIR, f"session_{session_id}.json")

def save_state(state: BotState):
    """Persist the current BotState to a file."""
    state_path = get_state_path(state.session_id)
    try:
        with open(state_path, 'w', encoding='utf-8') as f:
            # Use Pydantic's serialization method
            f.write(state.model_dump_json(indent=2))
            print(" ---- json dump  --- ", state.model_dump_json(indent=2))
        logger.info(f"Saved state for session {state.session_id} to {state_path}")
    except (IOError, TypeError, UnicodeEncodeError) as e:
        logger.error(f"Failed to save state for session {state.session_id}: {e}", exc_info=True)

def load_state(session_id: str) -> Optional[BotState]:
    """Load BotState from a file if it exists."""
    state_path = get_state_path(session_id)
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
                # Use Pydantic's parsing method
                state = BotState.model_validate(state_data)
            logger.info(f"Loaded state for session {session_id} from {state_path}")
            return state
        except (json.JSONDecodeError, IOError, UnicodeDecodeError, ValidationError) as e:
            logger.error(f"Failed to load or validate state for session {session_id}: {e}", exc_info=True)
            # Optionally delete corrupted state file
            # try:
            #     os.remove(state_path)
            # except OSError:
            #     pass
    logger.info(f"No previous state found for session {session_id}")
    return None

# --- Graph Validation ---
def check_for_cycles(graph: GraphOutput) -> Tuple[bool, str]:
    """
    Checks for cycles in the graph using Depth First Search.
    Returns (is_acyclic, message).
    """
    if not graph or not graph.nodes:
        return True, "Graph is empty or has no nodes. No cycles possible."

    nodes = {node.operationId for node in graph.nodes}
    adj: Dict[str, List[str]] = {node_id: [] for node_id in nodes}
    for edge in graph.edges:
        # Ensure edges only connect existing nodes
        if edge.from_node in adj and edge.to_node in nodes:
             # Check if target node is actually in the node list (safeguard)
             if edge.to_node in adj:
                adj[edge.from_node].append(edge.to_node)
             else:
                  logger.warning(f"Edge target '{edge.to_node}' not found in graph nodes list, skipping edge for cycle check.")
        else:
             logger.warning(f"Edge references non-existent node(s): {edge.from_node} -> {edge.to_node}. Skipping edge for cycle check.")


    # 0: unvisited, 1: visiting, 2: visited
    visited: Dict[str, int] = {node_id: 0 for node_id in nodes}
    recursion_stack: Dict[str, bool] = {node_id: False for node_id in nodes} # More explicit than using visited states

    path = [] # To report the cycle path

    def dfs(node_id: str) -> bool:
        """Returns True if a cycle is detected."""
        visited[node_id] = 1
        recursion_stack[node_id] = True
        path.append(node_id)

        for neighbor in adj.get(node_id, []):
            if neighbor not in visited or visited[neighbor] == 0: # Not visited yet
                if dfs(neighbor):
                    return True
            elif recursion_stack.get(neighbor, False): # Neighbor is in the current recursion stack
                # Cycle detected - find the cycle path
                try:
                    cycle_start_index = path.index(neighbor)
                    cycle_path = path[cycle_start_index:] + [neighbor]
                    logger.warning(f"Cycle detected: {' -> '.join(cycle_path)}")
                    return True # Cycle found
                except ValueError:
                     # Should not happen if logic is correct, but handle gracefully
                     logger.error(f"Error finding cycle path starting from {neighbor}")
                     return True # Report cycle anyway

        # Backtrack
        path.pop()
        recursion_stack[node_id] = False
        visited[node_id] = 2 # Mark as fully visited
        return False # No cycle found starting from this node

    for node_id in nodes:
        if visited[node_id] == 0:
            path = [] # Reset path for each new DFS tree
            if dfs(node_id):
                # Constructing the message outside DFS for clarity
                # The actual cycle path is logged within DFS upon detection
                return False, f"Cycle detected in the graph (involving node {node_id} or its descendants)."

    return True, "Graph is a Directed Acyclic Graph (DAG)."

# --- JSON Parsing Helper ---
def parse_llm_json_output(llm_output: str, expected_model: Optional[BaseModel] = None) -> Optional[Any]:
    """
    Attempts to parse JSON from LLM output, optionally validating against a Pydantic model.
    Handles common LLM JSON issues like markdown code fences.
    Returns the parsed data (dict or model instance) or None on failure.
    """
    cleaned_output = llm_output.strip()

    # Remove potential markdown code fences
    if cleaned_output.startswith("```json"):
        cleaned_output = cleaned_output[7:]
    elif cleaned_output.startswith("```"):
         cleaned_output = cleaned_output[3:]

    if cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[:-3]

    cleaned_output = cleaned_output.strip()

    try:
        parsed_json = json.loads(cleaned_output)
        if expected_model:
            try:
                validated_data = expected_model.model_validate(parsed_json)
                logger.debug(f"Successfully parsed and validated JSON against {expected_model.__name__}")
                return validated_data
            except ValidationError as ve:
                logger.error(f"JSON validation failed against {expected_model.__name__}: {ve}\nRaw JSON: {cleaned_output}", exc_info=True)
                return None # Validation failed
        else:
            logger.debug("Successfully parsed JSON (no model validation requested).")
            return parsed_json # Return raw parsed JSON if no model provided
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM output: {e}\nRaw Output: {llm_output}", exc_info=True)
        return None # Parsing failed

