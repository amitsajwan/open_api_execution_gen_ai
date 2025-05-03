import logging
import json # Import json for potential JSON parsing of LLM output
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import your existing components
from models import BotState, GraphOutput # Assuming GraphOutput is in models.py
from core_logic import OpenAPICoreLogic
from router import OpenAPIRouter # Keep router for initial high-level routing

# Assuming utils.py contains llm_call_helper and parse_llm_json_output
from utils import llm_call_helper, parse_llm_json_output

logger = logging.getLogger(__name__)

# Define a broader set of graph nodes, including the new P/E/R components
# and reusing your core_logic methods.
# The initial router will direct traffic to these or to the new 'planner'.
# Note: The actual nodes added to the graph builder will be from tool_methods
# and the explicitly defined planner, executor, responder.
# This list is more for conceptual clarity and ensuring all potential destinations are considered.
ALL_POSSIBLE_NODE_NAMES = [
    "planner",          # Decides which API calls (operationIds) to execute OR which core_logic step to take
    "executor",         # Executes the planned API calls from execution_graph
    "responder",        # Formats the final response
    # Reuse relevant core_logic methods as nodes for graph building/preparation
    "parse_openapi_spec",
    "identify_apis",
    "generate_payloads",
    "generate_execution_graph",
    "describe_graph",
    "get_graph_json",
    "handle_unknown",
    "handle_loop",
    # Add validate_graph if you want it as an explicit step
    # "validate_graph", # Assuming validate_logic is in core_logic
]

def build_graph(router_llm: Any, worker_llm: Any) -> Any:
    core_logic = OpenAPICoreLogic(worker_llm)
    # The OpenAPIRouter will be used for initial high-level intent routing from START.
    # It should ideally route to the 'planner' for most user inputs,
    # unless the intent is a clear command like "load spec".
    router = OpenAPIRouter(router_llm)

    builder = StateGraph(BotState)

    # --- Node Definitions ---

    # Helper to wrap core_logic methods to fit the graph node signature
    # This wrapper ensures the state is validated and exceptions are caught.
    def wrap_core_logic_method(fn):
        def node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            state = BotState.model_validate(state_dict)
            node_name = fn.__name__ # Get the original function name
            logger.debug(f"Executing core_logic node: {node_name}")
            state.scratchpad['last_executed_node'] = node_name # Track last executed node

            try:
                # Call the actual core_logic method
                # Assuming core_logic methods return a BotState instance or a dictionary
                result = fn(state)
                # Ensure the result is a dictionary (model_dump) if it's a Pydantic model
                return result.model_dump() if isinstance(result, BotState) else result
            except Exception as e:
                logger.error(f"Error in core_logic node {node_name}: {e}", exc_info=True)
                # Capture error in state for potential error handling or logging
                state.scratchpad['last_error'] = str(e)
                # Return the modified state as a dictionary
                return state.model_dump()
        # Assign a name to the wrapped function for logging/debugging
        node.__name__ = fn.__name__
        return node

    # 1. The Planner Node
    # This node takes the user query and the current state to decide the next action.
    # It's the core of the dynamic behavior.
    # It needs to interact with an LLM (router_llm or worker_llm) to make decisions.
    def planner_node(state: BotState) -> Dict[str, Any]:
        """
        Analyzes user query and current state using an LLM to decide the next action.
        Considers the outcome of the previous step.
        Possible outcomes:
        - Route to a core_logic graph building step (e.g., generate_execution_graph).
        - Set state.plan (list of operationIds) and route to the executor.
        - Route to handle_unknown or handle_loop.
        - Route to responder if the query seems fulfilled.
        Sets state.scratchpad['planner_decision'] to indicate the chosen path.
        """
        logger.debug("---PLANNER NODE---")
        query = state.user_input
        execution_graph = state.execution_graph
        openapi_schema = state.openapi_schema
        last_executed_node = state.scratchpad.get('last_executed_node') # Get the node that ran before planner
        last_error = state.scratchpad.get('last_error') # Get any error from the previous node

        logger.debug(f"User Query: {query}")
        logger.debug(f"Execution Graph Exists: {execution_graph is not None}")
        logger.debug(f"OpenAPI Schema Exists: {openapi_schema is not None}")
        logger.debug(f"Loop Counter: {state.loop_counter}") # Check loop counter from router
        logger.debug(f"Last Executed Node: {last_executed_node}")
        logger.debug(f"Last Error: {last_error}")


        # --- LLM Planning Logic ---
        # Build the prompt for the LLM
        prompt_context = "Analyze the user's request and the current state of the system to determine the next best action."

        # Describe the current state, including the outcome of the previous step
        state_description = f"""
Current State:
- User Input: "{query}"
- OpenAPI Specification Loaded: {'Yes' if openapi_schema else 'No'}
- Execution Graph Exists: {'Yes' if execution_graph else 'No'}
- Last Executed Node: {last_executed_node or 'None'}
- Last Error Encountered: {last_error or 'None'}
"""
        if execution_graph:
             state_description += f"- Execution Graph Description: {execution_graph.description or 'No description available.'}\n"
             if execution_graph.nodes:
                 # Provide a summary of available operations in the graph
                 node_summaries = "\n".join([f"  - {node.operationId}: {node.summary or 'No summary'}" for node in execution_graph.nodes[:10]]) # Limit for brevity
                 state_description += f"- Available API Operations in Graph ({len(execution_graph.nodes)} total, showing first 10):\n{node_summaries}\n"
             else:
                 state_description += "- Execution Graph is empty.\n"
        if openapi_schema and not execution_graph:
             # If schema is loaded but no graph, list some potential actions from the schema
             # This would require parsing schema paths/operations, which might be complex.
             # For simplicity, we'll just note that the schema is available.
             state_description += "- OpenAPI Schema is loaded and available for analysis/graph building.\n"

        # Add information about the outcome of the last step if relevant
        if last_executed_node == "identify_apis" and state.identified_apis is not None:
             state_description += f"- Outcome of identify_apis: Found {len(state.identified_apis)} APIs.\n"
        # Add similar checks for other relevant nodes (e.g., parse_openapi_spec, generate_execution_graph)


        # Describe the possible actions (nodes the LLM can route to)
        # Map node names to user-friendly descriptions of the action they perform.
        available_actions = {
            "parse_openapi_spec": "Parse a new OpenAPI specification provided by the user.",
            "identify_apis": "Analyze the loaded OpenAPI specification to identify relevant API operations.",
            "generate_payloads": "Generate example request payloads for identified API operations based on the schema.",
            "generate_execution_graph": "Generate a sequence/workflow (execution graph) of API calls based on the loaded schema and user goal.",
            "executor": "Execute the planned sequence of API calls from the existing execution graph.",
            "describe_graph": "Provide a natural language description of the existing execution graph.",
            "get_graph_json": "Output the JSON representation of the existing execution graph.",
            "handle_unknown": "Respond to the user when the request is unclear or cannot be fulfilled in the current state.",
            "handle_loop": "Respond to the user when the system appears to be stuck in a loop.",
            "responder": "Generate the final response to the user based on the results or current state.", # Explicitly include responder as a possible action
            # Add descriptions for validate_graph, add_graph_edge if used
            # "validate_graph": "Validate the structure of the existing execution graph (e.g., check for cycles).",
            # "add_graph_edge": "Add a new dependency edge between two nodes in the existing execution graph based on user instructions.",
        }

        actions_description = "Available Actions (choose one to route to):\n" + "\n".join([f"- `{name}`: {desc}" for name, desc in available_actions.items()])

        # Instruct the LLM on the desired output format and decision logic
        output_instruction = """
Based on the user input and the current state, determine the single best 'next_action' from the Available Actions.
Crucially, consider the 'Last Executed Node' and 'Last Error Encountered'. If the previous action successfully addressed the user's input, the `next_action` should likely be `responder`.

If the 'next_action' is `executor`, you MUST also provide a `plan` which is a JSON list of `operationId` strings from the execution graph nodes, representing the sequence of API calls to make. You should also extract any relevant `parameters` from the user query needed for these API calls and return them as a JSON object where keys are operationIds and values are parameter dictionaries.
If the 'next_action' is `generate_execution_graph`, capture the user's goal/instructions for graph generation.
For all other `next_action` values, the `plan` and `parameters` fields are optional but can be included if relevant.

Output ONLY a JSON object in the following format:
```json
{{
  "next_action": "chosen_action_name",
  "plan": ["operationId1", "operationId2", ...], // Required only if next_action is "executor"
  "parameters": {{ "operationId1": {{ "param1": "value1", ... }}, ... }}, // Optional, extracted from user query
  "reasoning": "Brief explanation of why this action was chosen." // Optional
}}
```
Ensure the `next_action` exactly matches one of the Available Actions names.
"""

        full_prompt = f"{prompt_context}\n\n{state_description}\n{actions_description}\n{output_instruction}\nUser Query: \"{query}\"\n\nOutput JSON:"

        planner_decision = "handle_unknown" # Default fallback
        planned_operation_ids = []
        extracted_params = {}
        reasoning = "Default fallback - LLM call failed or response unparseable."

        # Simple loop detection check before calling LLM
        if state.loop_counter >= 2:
             planner_decision = "handle_loop"
             reasoning = "Detected potential loop based on router's loop counter."
             logger.info("Planner: Routing to handle_loop due to loop counter.")
        elif last_error:
             # If the previous step had an error, maybe route to handle_unknown or responder with error
             # For now, let's route to responder, which can report the error from scratchpad
             planner_decision = "responder"
             reasoning = f"Previous node ({last_executed_node}) reported an error."
             logger.warning(f"Planner: Routing to responder due to error in {last_executed_node}.")
             # Clear the last_error after handling it
             if 'last_error' in state.scratchpad:
                  del state.scratchpad['last_error']
        else: # No loop detected and no previous error
            try:
                # --- Call the LLM ---
                # Use the worker_llm or a dedicated router_llm for this planning step
                llm_response = llm_call_helper(worker_llm, full_prompt) # Using worker_llm as example

                # --- Parse LLM Response ---
                # Expecting a JSON object with 'next_action', 'plan', 'parameters', 'reasoning'
                parsed_response = parse_llm_json_output(llm_response) # Use your JSON parsing helper

                if isinstance(parsed_response, dict) and 'next_action' in parsed_response:
                    determined_action = parsed_response['next_action']

                    # Validate the determined action against available nodes
                    if determined_action in available_actions: # Check against keys of available_actions
                        planner_decision = determined_action
                        planned_operation_ids = parsed_response.get('plan', [])
                        extracted_params = parsed_response.get('parameters', {})
                        reasoning = parsed_response.get('reasoning', 'LLM provided action.')

                        # Additional validation for 'executor' action
                        if planner_decision == "executor":
                            if not isinstance(planned_operation_ids, list):
                                logger.warning(f"LLM returned non-list plan for executor: {planned_operation_ids}. Defaulting to handle_unknown.")
                                planner_decision = "handle_unknown"
                                reasoning = "LLM returned invalid plan format for executor."
                                planned_operation_ids = []
                                extracted_params = {}
                            elif not execution_graph:
                                logger.warning("LLM decided 'executor' but no execution graph exists. Defaulting to handle_unknown.")
                                planner_decision = "handle_unknown"
                                reasoning = "LLM decided 'executor' but no graph exists."
                                planned_operation_ids = []
                                extracted_params = {}
                            else:
                                # Optional: Validate if planned_operation_ids actually exist in the execution_graph nodes
                                graph_op_ids = {node.operationId for node in execution_graph.nodes}
                                valid_plan = [op_id for op_id in planned_operation_ids if op_id in graph_op_ids]
                                if len(valid_plan) != len(planned_operation_ids):
                                     invalid_ids = set(planned_operation_ids) - graph_op_ids
                                     logger.warning(f"LLM planned execution of non-existent operationIds: {invalid_ids}. Using valid subset.")
                                     planned_operation_ids = valid_plan # Use only valid operationIds
                                     if not planned_operation_ids:
                                         # If all planned IDs were invalid, route to unknown
                                         planner_decision = "handle_unknown"
                                         reasoning = f"LLM planned execution of operationIds not found in graph: {invalid_ids}."


                        # If the planner decided to build the graph, capture instructions
                        if planner_decision == "generate_execution_graph":
                             # The LLM might provide specific instructions in the reasoning or a dedicated field
                             # For now, we'll rely on the router capturing the original query as instructions,
                             # but a more advanced LLM could provide structured graph building instructions here.
                             pass # Instructions are captured by the router or in state.graph_generation_instructions

                    else:
                        logger.warning(f"LLM returned invalid next_action: {determined_action}. Defaulting to handle_unknown.")
                        planner_decision = "handle_unknown"
                        reasoning = f"LLM returned invalid action name: {determined_action}."

                else:
                    logger.warning(f"LLM response not a valid JSON object with 'next_action': {llm_response[:500]}...")
                    # Default to handle_unknown
                    planner_decision = "handle_unknown"
                    reasoning = "LLM response was not in the expected JSON format."


            except Exception as e:
                logger.error(f"Error during LLM planning call or parsing: {e}", exc_info=True)
                # Default to handle_unknown on any error
                planner_decision = "handle_unknown"
                reasoning = f"Error during planning: {e}"


        # Update state based on the planning outcome
        state.scratchpad['planner_decision'] = planner_decision
        state.plan = planned_operation_ids
        state.extracted_params = extracted_params
        state.scratchpad['planner_reasoning'] = reasoning # Store reasoning for debugging/logging

        logger.debug(f"Planner Final Decision: {planner_decision}")
        logger.debug(f"Planner Final Plan: {state.plan}")
        logger.debug(f"Planner Final Extracted Params: {state.extracted_params}")


        # The actual transition happens in the router function below,
        # which reads state.scratchpad['planner_decision']

        # Return the updated state.
        return state.model_dump()

    # 2. The Executor Node
    # This node iterates through the state.plan and executes the corresponding API calls.
    # It needs access to the actual API calling logic for each operationId.
    # This logic is NOT provided in your core_logic.py, so you will need to implement it
    # or integrate an existing library/module for making HTTP requests based on OpenAPI details.
    # This node decides to loop back to itself for the next step or move to the responder.
    def executor_node(state: BotState) -> Dict[str, Any]:
        """
        Executes the API calls specified in state.plan.
        Iterates through the plan, calls APIs, and updates state.results.
        Updates state.current_step.
        """
        logger.debug("---EXECUTOR NODE---")
        plan = state.plan
        current_step = state.current_step
        execution_graph = state.execution_graph # Needed to get node details like summary/description
        openapi_schema = state.openapi_schema # Needed to get full API path, method, parameters
        extracted_params = state.extracted_params

        state.scratchpad['last_executed_node'] = "executor" # Track last executed node

        # Check if execution is complete (router handles the transition based on this)
        if not plan or current_step >= len(plan):
            logger.debug("Executor: Execution plan finished or empty.")
            state.scratchpad['executor_status'] = "completed" # Signal completion
            # The executor_router will check this status
            return state.model_dump()

        # Get the operationId for the current step
        current_operation_id = plan[current_step]
        logger.info(f"Executor: Executing step {current_step + 1}/{len(plan)}: {current_operation_id}")

        api_result_data = None
        step_error = None

        try:
            # --- API Execution Logic (User Implementation Required) ---
            # You need to implement the actual code to call the API here.
            # This involves:
            # 1. Finding the API details (path, method, parameters, request body schema)
            #    from state.openapi_schema using the current_operation_id.
            # 2. Constructing the full request URL.
            # 3. Preparing request headers (e.g., Content-Type, Authorization if needed).
            # 4. Preparing the request body if required, potentially using state.generated_payloads
            #    or parameters from state.extracted_params.
            # 5. Making the HTTP request (using libraries like 'requests').
            # 6. Handling the HTTP response (status codes, parsing JSON response body).
            # 7. Extracting relevant data from the response to potentially be used by
            #    subsequent API calls in the plan (this is advanced dependency handling).

            logger.info(f"Executor: Simulating API call for operationId: {current_operation_id}")
            # Placeholder: Simulate success or failure
            if "error" in current_operation_id.lower(): # Simulate an error for certain operationIds
                 raise Exception(f"Simulated error during execution of {current_operation_id}")

            # Find the node details from the execution graph for context (optional)
            api_node_details = next((node for node in (execution_graph.nodes if execution_graph else []) if node.operationId == current_operation_id), None)

            # Find the full API details from the openapi_schema (REQUIRED for actual call)
            # This part needs implementation: iterate through paths/methods in openapi_schema
            # to find the matching operationId and get path, method, parameters, etc.
            # Example: api_details_from_schema = get_api_details_from_schema(openapi_schema, current_operation_id)
            api_details_from_schema = {"path": "/simulated/path", "method": "get"} # Placeholder

            # Use extracted_params for this operation if available
            params_for_this_call = extracted_params.get(current_operation_id, {})

            # --- Replace with actual HTTP request ---
            # Example (using a hypothetical function you need to create):
            # from your_api_caller_module import make_http_request
            # response = make_http_request(
            #     method=api_details_from_schema['method'],
            #     url=f"your_base_url{api_details_from_schema['path']}", # Need base URL from schema or config
            #     parameters=params_for_this_call,
            #     # Add headers, body based on schema and state.generated_payloads
            #     # You might need to pass state.generated_payloads or look up the payload for this operationId
            #     # body=state.generated_payloads.get(current_operation_id) if api_details_from_schema['method'] in ['post', 'put', 'patch'] else None
            # )
            # response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            # api_result_data = response.json() # Or response.text, depending on API response type

            # Simulated successful result:
            api_result_data = {"status": "success", "operation": current_operation_id, "data": f"Simulated data for {current_operation_id}"}
            logger.info(f"Executor: Simulated API call success for {current_operation_id}. Result: {api_result_data}")


        except Exception as e:
            logger.error(f"Executor: Error executing API {current_operation_id}: {e}", exc_info=True)
            step_error = f"API Execution Error for {current_operation_id}: {e}"
            api_result_data = {"status": "error", "operation": current_operation_id, "error": str(e)} # Store error info in results

        # Store the result (or error info) in the state's results list
        state.results.append(api_result_data)

        # Decide next step: either the next API in the plan or signal completion
        if step_error:
            # Decide if execution should stop on error or continue
            state.scratchpad['execution_error'] = step_error
            # For this example, stop execution on the first error
            state.current_step = len(plan) # Mark plan as complete to stop loop
            logger.warning(f"Executor: Execution stopped due to error in {current_operation_id}")
        else:
            # Move to the next step in the plan
            state.current_step += 1
            logger.debug(f"Executor: Moved to next step. current_step: {state.current_step}")


        # The executor_router function will check if current_step < len(plan) to loop back
        # No need to explicitly set executor_status here, the router checks current_step vs len(plan)
        # state.scratchpad['executor_status'] = "in_progress" if state.current_step < len(plan) else "completed"

        return state.model_dump()

    # 3. The Responder Node
    # This node formats the final response based on the results from the executor
    # or other information in the state (e.g., graph description).
    # It sets the state.final_response field.
    def responder_node(state: BotState) -> Dict[str, Any]:
        """
        Formats the final response to the user based on state.results,
        state.scratchpad['execution_error'], or other state information.
        Sets state.final_response. Clears execution-related state.
        """
        logger.debug("---RESPONDER NODE---")
        results = state.results
        execution_error = state.scratchpad.get('execution_error')
        # We can check the planner's last decision from scratchpad to understand the context
        planner_last_decision = state.scratchpad.get('planner_decision')
        query = state.user_input

        final_response_text = ""

        # Determine what kind of response is needed based on the planner's last decision
        if planner_last_decision == "executor":
            # Response based on API execution results
            if execution_error:
                final_response_text = f"An error occurred during API execution: {execution_error}\n"
                if results:
                    # Include partial results if any
                    final_response_text += "Partial results obtained:\n" + json.dumps(results, indent=2)
                else:
                    final_response_text += "No results were obtained."
            elif results:
                # --- Response Generation Placeholder (using LLM recommended) ---
                # Use an LLM here to synthesize the 'results' list (which contains dicts/data)
                # into a natural language response that directly answers the user's original 'query'.
                # Prompt the LLM with: user_input, the raw results data.
                # For simplicity, just dump the results JSON.
                final_response_text = "Here are the results from the API calls:\n" + json.dumps(results, indent=2)
            else:
                 final_response_text = f"Execution completed, but no results were returned for query: '{query}'."

        # Handle responses for other planner decisions that lead directly to responder
        elif planner_last_decision == "describe_graph":
             # The describe_graph node should have set state.response. Use that.
             final_response_text = state.response or "No execution graph description available."

        elif planner_last_decision == "get_graph_json":
             # The get_graph_json node should have set state.response. Use that.
             final_response_text = state.response or "No execution graph available to output as JSON."

        elif planner_last_decision in ["handle_unknown", "unsupported_by_planner"]:
             # The handle_unknown node should have set state.response. Use that.
             final_response_text = state.response or f"Sorry, I couldn't understand your request: '{query}'. Can you please rephrase? (Planner decision: {planner_last_decision})"

        elif planner_last_decision == "handle_loop":
             # The handle_loop node should have set state.response. Use that.
             final_response_text = state.response or "It looks like we're in a loop. How would you like to proceed?"

        # Fallback response if planner_last_decision was not set or not handled above
        # This might happen if a core_logic node routed directly to responder without planner
        if not final_response_text and state.response:
             # Use the response set by the last core_logic node if no specific P/E/R response was generated
             final_response_text = state.response
        elif not final_response_text:
             # Final fallback
             final_response_text = "Processing complete." # Or a more informative default

        logger.info(f"Setting final_response: {final_response_text[:200]}...") # Log snippet
        state.final_response = final_response_text # Set the final response field

        # Clear execution-related state fields after responding
        state.plan = []
        state.results = []
        state.current_step = 0
        if 'execution_error' in state.scratchpad:
             del state.scratchpad['execution_error']
        if 'planner_decision' in state.scratchpad:
             del state.scratchpad['planner_decision'] # Clear planner decision after use
        # Keep state.response if it contains a message from a core_logic node
        # It might be useful for the next turn if the user asks a follow-up.
        # Or clear it if you want a clean slate for the next turn.
        # Decided to keep state.response as it might contain info from graph building steps.

        return state.model_dump()


    # --- Add nodes to the graph builder ---

    # Wrap existing core_logic methods as nodes
    tool_methods = {
        "parse_openapi_spec": core_logic.parse_openapi_spec,
        "identify_apis": core_logic.identify_apis,
        "generate_payloads": core_logic.generate_payloads,
        "generate_execution_graph": core_logic.generate_execution_graph,
        "describe_graph": core_logic.describe_graph,
        "get_graph_json": core_logic.get_graph_json,
        "handle_unknown": core_logic.handle_unknown,
        "handle_loop": core_logic.handle_loop,
        # Add validate_graph if you included it in core_logic
        # "validate_graph": core_logic.validate_graph,
        # Add add_graph_edge if you included it
        # "add_graph_edge": core_logic.add_graph_edge,
    }
    for name, fn in tool_methods.items():
        builder.add_node(name, wrap_core_logic_method(fn))

    # Add the new P/E/R nodes
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("responder", responder_node)


    # --- Define Graph Flow ---

    # 1. Entry point: Start with the existing router for initial high-level intent
    # The router should now primarily route to the 'planner' for dynamic handling,
    # unless it's a very specific, non-planning command.
    # You need to modify router.py's _determine_intent to return 'planner'
    # for queries that require dynamic API execution or graph building based on query.
    # Assuming router.AVAILABLE_INTENTS now includes 'planner' as a possible return value.
    builder.add_conditional_edges(
        START,
        router.route, # Use your existing router
        # Map AVAILABLE_INTENTS (including 'planner') to their corresponding nodes
        {intent: intent for intent in router.AVAILABLE_INTENTS}
    )

    # 2. After the planner, route based on its decision
    # This router reads state.scratchpad['planner_decision']
    def planner_router(state: BotState) -> str:
        decision = state.scratchpad.get('planner_decision', 'handle_unknown')
        logger.debug(f"Planner Router: Routing based on decision '{decision}'")
        # Map planner decisions to node names
        # Ensure all possible decisions from planner_node are handled here
        if decision == "executor":
            return "executor"
        elif decision == "generate_execution_graph":
            return "generate_execution_graph"
        elif decision == "describe_graph":
             return "describe_graph"
        elif decision == "get_graph_json":
             return "get_graph_json"
        elif decision == "parse_openapi_spec":
             return "parse_openapi_spec"
        elif decision == "identify_apis":
             return "identify_apis"
        elif decision == "generate_payloads":
             return "generate_payloads"
        elif decision == "handle_loop":
             return "handle_loop"
        elif decision == "responder": # Planner can now explicitly route to responder
             return "responder"
        # Add validate_graph and add_graph_edge if used
        # elif decision == "validate_graph":
        #      return "validate_graph"
        # elif decision == "add_graph_edge":
        #      return "add_graph_edge"
        else:
            # Default or unsupported decision goes to handle_unknown
            # The planner should ideally set a specific decision, but this is a fallback
            logger.warning(f"Planner Router: Unhandled planner decision '{decision}', routing to handle_unknown.")
            return "handle_unknown"


    builder.add_conditional_edges(
        "planner", # From the planner node
        planner_router, # Use the new planner_router function
        # Define all possible transitions from the planner
        {
            "executor": "executor",
            "generate_execution_graph": "generate_execution_graph",
            "describe_graph": "describe_graph",
            "get_graph_json": "get_graph_json",
            "parse_openapi_spec": "parse_openapi_spec",
            "identify_apis": "identify_apis",
            "generate_payloads": "generate_payloads",
            "handle_unknown": "handle_unknown",
            "handle_loop": "handle_loop",
            "responder": "responder", # Added responder as a possible transition
            # Add validate_graph and add_graph_edge if used
            # "validate_graph": "validate_graph",
            # "add_graph_edge": "add_graph_edge",
        }
    )

    # 3. After core_logic nodes:
    # - Graph building/modifying nodes route back to the planner for re-evaluation.
    # - Output nodes route to the responder.

    # Define which core_logic nodes should route back to the planner
    # These are nodes that change the state in a way that might require further planning or action
    # (e.g., after parsing a spec, the next step might be to identify APIs or build a graph).
    CORE_LOGIC_TO_PLANNER = [
        "parse_openapi_spec",
        "identify_apis",
        "generate_payloads",
        "generate_execution_graph",
        # Add "add_graph_edge", "validate_graph" if you want the planner to decide
        # the next step after these operations.
        # "add_graph_edge",
        # "validate_graph",
    ]

    for node_name in CORE_LOGIC_TO_PLANNER:
        if node_name in tool_methods: # Ensure the node exists
             # Route these nodes to the planner for the next decision
             builder.add_edge(node_name, "planner")
             logger.debug(f"Added edge: {node_name} -> planner")


    # Define which core_logic nodes should route directly to the responder
    # These are nodes whose primary purpose is to generate a final output message.
    CORE_LOGIC_TO_RESPONDER = [
        "describe_graph",
        "get_graph_json",
        "handle_unknown",
        "handle_loop",
    ]

    for node_name in CORE_LOGIC_TO_RESPONDER:
         if node_name in tool_methods: # Ensure the node exists
              # Route these nodes directly to the responder
              builder.add_edge(node_name, "responder")
              logger.debug(f"Added edge: {node_name} -> responder")


    # 4. After the executor, loop back to the executor if there are more steps in the plan,
    # otherwise, go to the responder.
    def executor_router(state: BotState) -> str:
        logger.debug(f"Executor Router: current_step={state.current_step}, plan_length={len(state.plan)}")
        # If there are more steps in the plan, loop back to the executor
        if state.current_step < len(state.plan):
            logger.debug("Executor Router: More steps in plan, looping back to executor.")
            return "continue_execution" # Return a key that maps back to 'executor'
        else:
            # Plan is complete, move to the responder
            logger.debug("Executor Router: Plan complete, routing to responder.")
            return "finish_execution" # Return a key that maps to 'responder'

    builder.add_conditional_edges(
        "executor", # From the executor node
        executor_router, # Use the executor_router function
        {
            "continue_execution": "executor", # Loop back to executor
            "finish_execution": "responder",  # Move to responder
        }
    )


    # 5. From the responder, the graph ends
    builder.add_edge("responder", END)


    # Compile the graph
    # Ensure checkpointer is configured if you need state persistence across calls
    app = builder.compile(checkpointer=MemorySaver()) # Using MemorySaver as in your original code
    logger.info("Graph compiled with dynamic API execution capabilities (Planner/Executor/Responder).")
    return app

# --- Example Usage (Conceptual) ---
# To run this, you would need:
# 1. Implement the actual API calling logic within the executor_node or a function it calls.
# 2. Replace the placeholder LLM calls in planner_node and potentially responder_node.
# 3. Ensure your router.py's _determine_intent can return 'planner' for relevant queries.
# 4. Have placeholder or actual LLM instances (router_llm, worker_llm) to pass to build_graph.
# 5. Have a way to set the initial state with user_input and potentially openapi_spec_text.

# Example of how you might invoke the graph (assuming you have LLMs and initial state setup):
# from your_llm_module import router_llm, worker_llm # Replace with your actual imports
# from persistence import load_state # Assuming you have load_state
# import uuid # To generate session IDs

# # Assume router_llm and worker_llm are initialized LLM instances
# # Assume save_state and load_state are implemented in persistence.py

# # Example 1: Load spec and build graph
# print("\n--- Running Example 1 (Load Spec & Build Graph) ---")
# session_id_1 = str(uuid.uuid4())
# initial_state_load_spec = BotState(
#     session_id=session_id_1,
#     user_input="Here is an OpenAPI spec. Please parse it and build a workflow graph.",
#     openapi_spec_text="""
# openapi: 3.0.0
# info:
#   title: Example API
#   version: 1.0.0
# paths:
#   /users:
#     get:
#       operationId: listUsers
#       summary: List all users
#       responses:
#         '200':
#           description: A list of users.
#   /users/{userId}:
#     get:
#       operationId: getUser
#       summary: Get user details by ID
#       parameters:
#         - name: userId
#           in: path
#           required: true
#           schema:
#             type: integer
#       responses:
#         '200':
#           description: User details.
#   /items:
#     get:
#       operationId: listItems
#       summary: List all items
#       responses:
#         '200':
#           description: A list of items.
# """ # Replace with a real spec for testing
# )
# app = build_graph(router_llm, worker_llm)
# # The router should route this to 'parse_openapi_spec', which routes to 'planner'.
# # The planner should decide 'generate_execution_graph', which routes to 'planner'.
# # The planner might then decide 'describe_graph', which routes to 'responder'.
# # The final state should contain the description of the generated graph.
# # Note: The exact sequence depends on your LLM's responses in planner_node
# # and the routing logic in router.py.
# # For this example to work end-to-end, you need LLM implementations.
# # final_state_load_spec = app.invoke(initial_state_load_spec.model_dump())
# # print(f"Session 1 Final Response: {final_state_load_spec.get('final_response', final_state_load_spec.get('response', 'No response'))}")


# # Example 2: Execute APIs from an existing graph
# print("\n--- Running Example 2 (Execute APIs) ---")
# session_id_2 = str(uuid.uuid4())
# # Simulate a state where a spec is parsed and a graph exists.
# # In a real app, you'd load the state using the checkpointer/persistence layer.
# # For demo, we'll manually create a state with a mock graph and plan.
# mock_schema = {"paths": {"/users": {"get": {"operationId": "listUsers"}}, "/items": {"get": {"operationId": "listItems"}}}}
# mock_graph = GraphOutput(
#     nodes=[{"operationId": "listUsers", "summary": "List all users"}, {"operationId": "listItems", "summary": "List all items"}],
#     edges=[], # Simple graph with no dependencies
#     description="Workflow to list users and items."
# )
# initial_state_execute = BotState(
#     session_id=session_id_2,
#     user_input="Run the workflow to get the list of users and items.",
#     openapi_schema=mock_schema, # Schema needed by executor to get API details
#     execution_graph=mock_graph,
#     # The planner would set this plan based on the query and graph
#     plan=["listUsers", "listItems"], # Planner determines this sequence
#     current_step=0,
#     results=[]
# )
# app = build_graph(router_llm, worker_llm)
# # The router should route this to 'planner', then the planner should decide 'executor',
# # then route to 'executor', which will loop until plan is done, then route to 'responder'.
# # final_state_execute = app.invoke(initial_state_execute.model_dump())
# # print(f"Session 2 Final Response: {final_state_execute.get('final_response', final_state_execute.get('response', 'No response'))}")

