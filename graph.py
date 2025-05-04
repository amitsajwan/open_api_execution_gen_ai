import logging
import json
import requests # Added for making HTTP requests
from typing import Any, Dict, List, Optional, Tuple
from langgraph.graph import StateGraph, START, END
from jsonpath_ng import parse as jsonpath_parse # Renamed import for clarity
from pydantic import ValidationError # Added for specific exception handling

from models import BotState, GraphOutput, Node # Import Node to access its structure
from core_logic import OpenAPICoreLogic
from router import OpenAPIRouter

# Assuming MemorySaver is defined appropriately (e.g., in utils or directly)
# from utils import MemorySaver
# Using LangGraph's built-in memory saver for simplicity here
from langgraph.checkpoint.memory import MemorySaver

from utils import llm_call_helper, parse_llm_json_output
# Removed placeholder extract_data_with_jsonpath as it's now implemented here

logger = logging.getLogger(__name__)

# --- JSONPath Extraction Utility ---
def extract_data_with_jsonpath(data: Any, path: str) -> Any:
    """
    Extracts data from a dictionary/list using a JSONPath expression.
    Uses the jsonpath-ng library.
    Example path: '$.items[0].id', '$.data.token'
    """
    if not path or data is None:
        logger.debug(f"JSONPath extraction skipped: No path ('{path}') or data provided.")
        return None
    try:
        jsonpath_expression = jsonpath_parse(path)
        matches = jsonpath_expression.find(data)
        if matches:
            # Return the value of the first match
            match_value = matches[0].value
            logger.debug(f"JSONPath extraction successful for path '{path}'. Found: {match_value}")
            return match_value
        else:
            logger.warning(f"JSONPath extraction: No match found for path '{path}' in data: {str(data)[:100]}...")
            return None
    except Exception as e:
        # Catch potential errors during parsing or finding
        logger.error(f"Error extracting data with JSONPath '{path}': {e}", exc_info=True)
        return None

# --- Schema Lookup Helper (Placeholder) ---
def find_operation_details(schema: Dict[str, Any], operation_id: str) -> Optional[Dict[str, Any]]:
    """
    Finds the details (path, method, parameters, etc.) for a given operationId
    within the parsed OpenAPI schema.

    Args:
        schema: The parsed OpenAPI schema dictionary.
        operation_id: The operationId to search for.

    Returns:
        A dictionary containing details like 'path', 'method', 'parameters',
        'requestBody', 'responses', or None if not found.
    """
    if not schema or 'paths' not in schema:
        return None

    for path, path_item in schema.get('paths', {}).items():
        if not isinstance(path_item, dict): continue
        for method, operation_obj in path_item.items():
             # Check if operation_obj is a dictionary and has 'operationId'
             if isinstance(operation_obj, dict) and operation_obj.get('operationId') == operation_id:
                 # Found the operation, return its details along with path and method
                 details = operation_obj.copy()
                 details['path'] = path
                 details['method'] = method.upper() # Ensure method is uppercase
                 # Combine path-level parameters with operation-level parameters
                 # Operation-level params override path-level if names conflict (as per OpenAPI spec)
                 path_params = {p['name']: p for p in path_item.get('parameters', []) if isinstance(p, dict) and 'name' in p}
                 op_params = {p['name']: p for p in operation_obj.get('parameters', []) if isinstance(p, dict) and 'name' in p}
                 path_params.update(op_params) # op_params take precedence
                 details['parameters'] = list(path_params.values())

                 logger.debug(f"Found details for operationId '{operation_id}': Method={details['method']}, Path={details['path']}")
                 return details
    logger.warning(f"OperationId '{operation_id}' not found in the provided OpenAPI schema.")
    return None


# --- Graph Building ---
def build_graph(router_llm: Any, worker_llm: Any) -> StateGraph:
    """
    Builds and compiles the LangGraph StateGraph for the OpenAPI execution agent.
    """
    core_logic = OpenAPICoreLogic(worker_llm)
    router = OpenAPIRouter(router_llm)

    builder = StateGraph(BotState)

    # --- Node Definitions ---

    # Helper function to wrap core_logic methods.
    def wrap_core_logic_method(fn):
        def node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Validate input state dictionary against the BotState model
                state = BotState.model_validate(state_dict)
            except ValidationError as e:
                 logger.error(f"State validation error entering node {fn.__name__}: {e}", exc_info=True)
                 # Decide how to handle invalid state - maybe route to error handler or return minimal state?
                 # Returning the original dict might cause issues downstream.
                 # Let's try returning a basic error state.
                 error_state = BotState(session_id=state_dict.get("session_id", "unknown"), user_input=state_dict.get("user_input"))
                 error_state.response = f"Internal Error: Invalid state entering node {fn.__name__}."
                 error_state.scratchpad['last_error'] = f"State validation error: {e}"
                 return error_state.model_dump()


            node_name = fn.__name__
            logger.debug(f"Executing core_logic node: {node_name}")
            state.scratchpad['last_executed_node'] = node_name # Track entry

            try:
                updated_state = fn(state) # Call the actual core logic method
                # Ensure the method returned a valid BotState instance
                if not isinstance(updated_state, BotState):
                     logger.error(f"Node {node_name} did not return a BotState instance. Returned: {type(updated_state)}")
                     # Handle this error - maybe revert to original state with an error message?
                     state.response = f"Internal Error: Node {node_name} failed to return valid state."
                     state.scratchpad['last_error'] = f"Node {node_name} returned type {type(updated_state)}"
                     return state.model_dump()

                # Return the updated state as a dictionary for LangGraph
                return updated_state.model_dump()

            except Exception as e:
                logger.error(f"Error executing core_logic node {node_name}: {e}", exc_info=True)
                # Update state with error information before returning
                state.response = f"An error occurred in {node_name}: {e}" # Provide basic error info
                state.scratchpad['last_error'] = f"Error in {node_name}: {e}"
                return state.model_dump() # Return the state (with error info) as a dictionary
        node.__name__ = fn.__name__ # Preserve original function name for node identification
        return node

    # 1. The Planner Node (Simplified for Brevity - Keep original logic if needed)
    def planner_node(state: BotState) -> Dict[str, Any]:
        """
        Analyzes user query and current state using an LLM to decide the next action.
        (Keeping the core logic similar to the original, focusing on fixing executor)
        """
        logger.debug("---PLANNER NODE---")
        # ... (Keep the detailed prompt construction and LLM call logic from original graph.py) ...
        # This node's primary job is to set state.scratchpad['planner_decision']
        # and potentially state.plan and state.extracted_params.

        # --- Placeholder Logic (Replace with original planner logic) ---
        logger.warning("Planner node logic is simplified. Restore original logic if needed.")
        query = state.user_input or ""
        planner_decision = "handle_unknown"
        reasoning = "Simplified planner logic."
        planned_operation_ids = []
        extracted_params = {}

        if state.openapi_schema and state.execution_graph and "execute" in query.lower():
             planner_decision = "executor"
             # Example: Extract plan from graph (needs actual logic)
             planned_operation_ids = [node.operationId for node in state.execution_graph.nodes]
             reasoning = "Simplified: Detected 'execute' and graph exists."
        elif state.openapi_schema and "describe" in query.lower():
             planner_decision = "describe_graph"
             reasoning = "Simplified: Detected 'describe' and schema exists."
        elif "parse" in query.lower():
             planner_decision = "parse_openapi_spec"
             reasoning = "Simplified: Detected 'parse'."
        # --- End Placeholder ---

        state.scratchpad['planner_decision'] = planner_decision
        state.plan = planned_operation_ids
        state.extracted_params = extracted_params
        state.scratchpad['planner_reasoning'] = reasoning
        logger.info(f"Planner Decision: {planner_decision}")

        # Store the node that executed immediately before the planner
        if state.scratchpad.get('last_executed_node'):
             state.scratchpad['last_executed_node_before_planner'] = state.scratchpad['last_executed_node']
        if 'last_executed_node' in state.scratchpad:
             del state.scratchpad['last_executed_node'] # Clear before next node runs

        return state.model_dump()


    # 2. The Executor Node (Rewritten for Real API Calls)
    def executor_node(state: BotState) -> Dict[str, Any]:
        """
        Executes the API calls specified in state.plan, using input mappings
        and making real HTTP requests.
        Iterates through the plan, calls APIs, and updates state.results.
        Updates state.current_step.
        """
        logger.debug("---EXECUTOR NODE---")
        plan = state.plan
        current_step = state.current_step
        execution_graph = state.execution_graph
        openapi_schema = state.openapi_schema
        initial_params = state.extracted_params
        results_so_far = state.results # Get results from previous steps

        state.scratchpad['last_executed_node'] = "executor" # Mark entry

        # --- Pre-execution Checks ---
        if not plan:
             step_error = "Execution Error: No plan (sequence of operations) provided."
             state.scratchpad['execution_error'] = step_error
             state.results["executor_error"] = {"status": "error", "error": step_error}
             logger.error(step_error)
             state.current_step = 0 # Reset step
             return state.model_dump()

        if current_step >= len(plan):
             step_error = f"Execution Error: Current step ({current_step}) is out of bounds for plan length ({len(plan)})."
             state.scratchpad['execution_error'] = step_error
             state.results["executor_error"] = {"status": "error", "error": step_error}
             logger.error(step_error)
             state.current_step = len(plan) # Ensure it stays at end
             return state.model_dump()

        if not execution_graph or not execution_graph.nodes:
             step_error = "Execution Error: Execution graph is missing or empty."
             state.scratchpad['execution_error'] = step_error
             state.results["executor_error"] = {"status": "error", "error": step_error}
             logger.error(step_error)
             state.current_step = len(plan) # Stop execution
             return state.model_dump()

        if not openapi_schema:
             step_error = "Execution Error: OpenAPI schema is missing. Cannot determine API details."
             state.scratchpad['execution_error'] = step_error
             state.results["executor_error"] = {"status": "error", "error": step_error}
             logger.error(step_error)
             state.current_step = len(plan) # Stop execution
             return state.model_dump()

        # --- Get Current Operation Details ---
        current_operation_id = plan[current_step]
        logger.info(f"Executor: Executing step {current_step + 1}/{len(plan)}: {current_operation_id}")

        # Find the node definition for input mappings
        current_node_definition: Optional[Node] = next(
            (node for node in execution_graph.nodes if node.operationId == current_operation_id), None
        )
        if not current_node_definition:
            step_error = f"Execution Error: OperationId '{current_operation_id}' not found in the execution graph nodes."
            state.scratchpad['execution_error'] = step_error
            state.results[current_operation_id] = {"status": "error", "operation": current_operation_id, "error": step_error}
            logger.error(step_error)
            state.current_step = len(plan) # Stop execution
            return state.model_dump()

        # Find operation details in the OpenAPI schema
        operation_details = find_operation_details(openapi_schema, current_operation_id)
        if not operation_details:
            step_error = f"Execution Error: Details for operationId '{current_operation_id}' not found in the OpenAPI schema."
            state.scratchpad['execution_error'] = step_error
            state.results[current_operation_id] = {"status": "error", "operation": current_operation_id, "error": step_error}
            logger.error(step_error)
            state.current_step = len(plan) # Stop execution
            return state.model_dump()

        # --- Parameter Binding ---
        params_for_this_call: Dict[str, Any] = {}
        # Start with initial parameters from planner/user for this operation
        if initial_params and current_operation_id in initial_params:
             params_for_this_call.update(initial_params.get(current_operation_id, {}))
             logger.debug(f"Applied initial parameters for {current_operation_id}: {params_for_this_call}")

        # Apply input mappings from previous step results
        if current_node_definition.input_mappings:
             logger.debug(f"Applying {len(current_node_definition.input_mappings)} input mappings for {current_operation_id}")
             for mapping in current_node_definition.input_mappings:
                 source_op_id = mapping.source_operation_id
                 source_path = mapping.source_data_path
                 target_param_name = mapping.target_parameter_name

                 if source_op_id in results_so_far:
                     source_result = results_so_far[source_op_id]
                     # Assume source_result['response_body'] holds the parsed JSON response body
                     source_data = source_result.get('response_body')

                     if source_data is not None:
                         extracted_value = extract_data_with_jsonpath(source_data, source_path)
                         if extracted_value is not None:
                             # TODO: Implement transformation logic if mapping.transformation exists
                             if mapping.transformation:
                                 logger.warning(f"Transformation '{mapping.transformation}' specified but not implemented.")
                             params_for_this_call[target_param_name] = extracted_value
                             logger.debug(f"Mapped data from {source_op_id}.{source_path} -> {target_param_name} = {extracted_value}")
                         else:
                             logger.warning(f"Could not extract data from {source_op_id} using path '{source_path}' for target '{target_param_name}'.")
                     else:
                         logger.warning(f"Source result for '{source_op_id}' has no 'response_body'. Cannot apply mapping for '{target_param_name}'.")
                 else:
                     logger.warning(f"Source operation '{source_op_id}' not found in results. Cannot apply mapping for '{target_param_name}'.")

        # --- Prepare API Request ---
        api_method = operation_details['method']
        api_path_template = operation_details['path']
        base_url = openapi_schema.get('servers', [{}])[0].get('url', '') # Get base URL from schema servers
        if not base_url:
             logger.warning("No base URL found in schema servers. Assuming relative paths.")

        # Separate parameters based on their 'in' location
        path_params = {}
        query_params = {}
        header_params = {}
        cookie_params = {} # Not handled in this example
        request_body = None

        # Get parameter definitions from the operation details
        parameter_definitions = {p['name']: p for p in operation_details.get('parameters', [])}

        for param_name, param_value in params_for_this_call.items():
            param_def = parameter_definitions.get(param_name)
            if param_def:
                param_in = param_def.get('in')
                if param_in == 'path':
                    path_params[param_name] = param_value
                elif param_in == 'query':
                    query_params[param_name] = param_value
                elif param_in == 'header':
                    header_params[param_name] = str(param_value) # Headers must be strings
                # Add handling for 'cookie' if needed
                else:
                     logger.warning(f"Parameter '{param_name}' has unhandled 'in' location: {param_in}")
            else:
                # If parameter not in defined list, assume it might be part of the request body
                # This is a simplification - real logic needs to check operation_details['requestBody'] schema
                logger.debug(f"Parameter '{param_name}' not found in explicit parameters. Assuming it's for request body.")
                if request_body is None: request_body = {}
                if isinstance(request_body, dict): # Only add if body is expected to be a dict
                     request_body[param_name] = param_value
                else:
                     logger.warning(f"Cannot add parameter '{param_name}' to non-dict request body.")


        # Check if requestBody is defined and if we haven't assigned anything yet
        if operation_details.get('requestBody') and request_body is None:
             # Maybe the entire payload was provided under a specific key by the planner?
             # Or maybe the example_payload should be used? Needs clearer logic.
             # For now, if parameters didn't fill the body, try using the node's example payload.
             if current_node_definition.example_payload:
                  logger.debug("Using example_payload as request body.")
                  request_body = current_node_definition.example_payload
             else:
                  logger.warning(f"Operation {current_operation_id} expects a request body, but none was constructed from parameters or example payload.")


        # Construct URL
        try:
            api_url_path = api_path_template.format(**path_params)
        except KeyError as e:
            step_error = f"Execution Error: Missing path parameter '{e}' needed for URL '{api_path_template}'."
            state.scratchpad['execution_error'] = step_error
            state.results[current_operation_id] = {"status": "error", "operation": current_operation_id, "error": step_error}
            logger.error(step_error)
            state.current_step = len(plan) # Stop execution
            return state.model_dump()

        full_api_url = base_url.rstrip('/') + api_url_path

        # --- Execute API Call ---
        step_result_data = {
            "status": "unknown",
            "operation": current_operation_id,
            "method": api_method,
            "url": full_api_url,
            "parameters_used": { # Log parameters sent
                "path": path_params,
                "query": query_params,
                "headers": header_params,
                "body": request_body
            },
            "status_code": None,
            "response_headers": None,
            "response_body": None,
            "error": None
        }
        step_error = None # Reset step error

        logger.info(f"Executor: Making API call: {api_method} {full_api_url}")
        logger.debug(f" Query Params: {query_params}")
        logger.debug(f" Headers: {header_params}")
        logger.debug(f" Body: {request_body}")

        try:
            # Add default headers, potentially merge with header_params
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            headers.update(header_params)

            # TODO: Add authentication headers if needed (e.g., Authorization: Bearer <token>)
            # auth_token = state.scratchpad.get('auth_token')
            # if auth_token: headers['Authorization'] = f"Bearer {auth_token}"

            response = requests.request(
                method=api_method,
                url=full_api_url,
                headers=headers,
                params=query_params,
                json=request_body if request_body is not None else None, # Send body as JSON if present
                timeout=30 # Add a timeout
            )

            step_result_data["status_code"] = response.status_code
            step_result_data["response_headers"] = dict(response.headers)

            logger.info(f"API call returned status: {response.status_code}")

            # Try to parse response body as JSON
            try:
                step_result_data["response_body"] = response.json()
                logger.debug("Parsed response body as JSON.")
            except json.JSONDecodeError:
                step_result_data["response_body"] = response.text
                logger.debug("Response body not JSON, stored as text.")

            # Check for HTTP errors
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            step_result_data["status"] = "success"
            logger.info(f"API call successful for {current_operation_id}.")

        except requests.exceptions.Timeout as e:
            step_error = f"API Execution Error (Timeout) for {current_operation_id}: {e}"
            step_result_data["status"] = "error"
            step_result_data["error"] = step_error
            logger.error(step_error)
        except requests.exceptions.ConnectionError as e:
            step_error = f"API Execution Error (Connection) for {current_operation_id}: {e}"
            step_result_data["status"] = "error"
            step_result_data["error"] = step_error
            logger.error(step_error)
        except requests.exceptions.HTTPError as e:
            step_error = f"API Execution Error (HTTP Status {step_result_data['status_code']}) for {current_operation_id}: {e}. Response: {step_result_data['response_body']}"
            step_result_data["status"] = "error" # Keep status as error
            step_result_data["error"] = step_error # Store the HTTP error details
            logger.error(step_error)
            # Don't necessarily stop execution on HTTP errors, let planner decide?
            # For now, we will stop.
        except requests.exceptions.RequestException as e:
            step_error = f"API Execution Error (General Request) for {current_operation_id}: {e}"
            step_result_data["status"] = "error"
            step_result_data["error"] = step_error
            logger.error(step_error, exc_info=True)
        except Exception as e:
            # Catch any other unexpected errors during request preparation or execution
            step_error = f"Unexpected Error during API execution for {current_operation_id}: {e}"
            step_result_data["status"] = "error"
            step_result_data["error"] = step_error
            logger.error(step_error, exc_info=True)


        # Store the detailed result in the state
        state.results[current_operation_id] = step_result_data

        # --- Post-execution ---
        if step_error:
            state.scratchpad['execution_error'] = step_error # Store the first critical error encountered
            # Decide whether to stop execution based on the error type
            # For now, stop on any error during execution phase
            state.current_step = len(plan)
            logger.warning(f"Executor: Execution stopped due to error in {current_operation_id}: {step_error}")
        else:
            state.current_step += 1
            logger.debug(f"Executor: Completed step {current_step}. Moved to next step: {state.current_step}")
            # Clear execution error if step was successful
            if 'execution_error' in state.scratchpad:
                 del state.scratchpad['execution_error']


        return state.model_dump()


    # 3. The Responder Node (Simplified for Brevity - Keep original logic if needed)
    def responder_node(state: BotState) -> Dict[str, Any]:
        """
        Formats the final response to the user.
        (Keeping the core logic similar to the original, focusing on fixing executor)
        """
        logger.debug("---RESPONDER NODE---")
        # ... (Keep the detailed response formatting logic from original graph.py) ...
        # This node's primary job is to set state.final_response based on
        # state.results, state.scratchpad['execution_error'], state.response, etc.

        # --- Placeholder Logic (Replace with original responder logic) ---
        logger.warning("Responder node logic is simplified. Restore original logic if needed.")
        final_response_text = "Processing complete."
        if state.scratchpad.get('execution_error'):
             final_response_text = f"Execution failed: {state.scratchpad['execution_error']}"
        elif state.results:
             final_response_text = "Execution finished. Results:\n" + json.dumps(state.results, indent=2, default=str)
        elif state.response:
             final_response_text = state.response # Use intermediate response if no results/error
        # --- End Placeholder ---

        state.final_response = final_response_text
        logger.info(f"Final response generated: {final_response_text[:200]}...")

        # Clear execution state
        state.plan = []
        state.current_step = 0
        # state.results = {} # Decide whether to clear results or keep for context
        if 'execution_error' in state.scratchpad: del state.scratchpad['execution_error']
        if 'planner_decision' in state.scratchpad: del state.scratchpad['planner_decision']
        if 'last_executed_node_before_planner' in state.scratchpad: del state.scratchpad['last_executed_node_before_planner']
        # state.response = None # Clear intermediate response?

        return state.model_dump()


    # --- Add nodes to the graph builder ---
    # Add core logic nodes
    tool_methods = {
        "parse_openapi_spec": core_logic.parse_openapi_spec,
        "identify_apis": core_logic.identify_apis,
        "generate_payloads": core_logic.generate_payloads,
        "generate_execution_graph": core_logic.generate_execution_graph,
        "describe_graph": core_logic.describe_graph,
        "get_graph_json": core_logic.get_graph_json,
        "handle_unknown": core_logic.handle_unknown,
        "handle_loop": core_logic.handle_loop,
        "answer_openapi_query": core_logic.answer_openapi_query,
    }
    for name, fn in tool_methods.items():
        builder.add_node(name, wrap_core_logic_method(fn))

    # Add main control flow nodes
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node) # Uses the rewritten executor
    builder.add_node("responder", responder_node)

    # --- Define Graph Flow ---

    # 1. Entry point: Start with the initial router
    builder.add_conditional_edges(
        START,
        router.route, # Use the route method of the OpenAPIRouter instance
        {intent: intent for intent in router.AVAILABLE_INTENTS} # Map intents directly to node names
    )

    # 2. After the planner node executes, route based on its decision.
    def planner_router(state: BotState) -> str:
        # Default to handle_unknown if decision is missing
        decision = state.scratchpad.get('planner_decision', 'handle_unknown')
        logger.debug(f"Planner Router: Routing based on decision '{decision}'")
        # Ensure the decision maps to a valid node name or END
        valid_nodes = list(tool_methods.keys()) + ["planner", "executor", "responder", END]
        if decision in valid_nodes:
             # Specific handling if executor is chosen but plan is empty
             if decision == "executor" and not state.plan:
                  logger.warning("Planner decided 'executor' but plan is empty. Routing to responder.")
                  # Update state to reflect this issue?
                  state.response = "Planner decided to execute, but no execution plan was generated."
                  return "responder" # Route to responder to report the issue
             return decision
        else:
            logger.error(f"Planner Router: Invalid planner decision '{decision}'. Routing to handle_unknown.")
            return "handle_unknown"

    builder.add_conditional_edges(
        "planner",
        planner_router,
        # Define potential target nodes from the planner
        {
            **{name: name for name in tool_methods.keys()}, # All core logic tools
            "executor": "executor",
            "responder": "responder",
            "handle_unknown": "handle_unknown", # Explicitly handle unknown case
            END: END # Allow planner to end graph if needed
        }
    )

    # 3. After core_logic nodes: Route back to planner generally
    # (Original logic had specific routing, adjust as needed)
    # For simplicity now, most tools route back to planner to decide next step
    CORE_LOGIC_NODES = list(tool_methods.keys())

    for node_name in CORE_LOGIC_NODES:
         # Nodes that handle final output or errors might route differently
         if node_name not in ["handle_unknown", "handle_loop"]: # Example: these might go to responder
              builder.add_edge(node_name, "planner")
              logger.debug(f"Added edge: {node_name} -> planner")

    # Specific routes for nodes that should go directly to responder
    builder.add_edge("handle_unknown", "responder")
    builder.add_edge("handle_loop", "responder")
    # Consider if get_graph_json, describe_graph etc. should go directly to responder
    # builder.add_edge("get_graph_json", "responder")
    # builder.add_edge("describe_graph", "responder")

    # 4. After the executor node executes, decide continuation
    def executor_router(state: BotState) -> str:
        logger.debug(f"Executor Router: current_step={state.current_step}, plan_length={len(state.plan)}")
        # Check for critical error first
        if state.scratchpad.get('execution_error'):
            logger.warning("Executor Router: Execution error detected. Routing to responder.")
            return "finish_execution" # Go to responder to report error
        # Check if more steps remain
        elif state.current_step < len(state.plan):
            logger.debug("Executor Router: More steps in plan, looping back to executor.")
            return "continue_execution"
        else:
            logger.debug("Executor Router: Plan complete, routing to responder.")
            return "finish_execution"

    builder.add_conditional_edges(
        "executor",
        executor_router,
        {
            "continue_execution": "executor", # Loop back to executor
            "finish_execution": "responder",  # Go to responder when done or error
        }
    )

    # 5. From the responder node, the graph ends.
    builder.add_edge("responder", END)

    # Compile the graph with memory
    # Ensure MemorySaver is correctly imported or defined
    app = builder.compile(checkpointer=MemorySaver())
    logger.info("Graph compiled with real API execution capabilities.")
    return app
