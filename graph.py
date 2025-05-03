import logging
import json
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END

from models import BotState, GraphOutput, Node # Import Node to access its structure
from core_logic import OpenAPICoreLogic
from router import OpenAPIRouter

from utils import llm_call_helper, parse_llm_json_output, MemorySaver # Assuming MemorySaver is in utils
# Assuming a utility function for JSONPath-like extraction exists
# from utils import extract_data_with_jsonpath # You would need to implement this

logger = logging.getLogger(__name__)

# Placeholder for a JSONPath-like extraction utility
def extract_data_with_jsonpath(data: Any, path: str) -> Any:
    """
    Placeholder function to extract data from a dictionary/list using a simple path.
    A real implementation would use a library like jsonpath-ng.
    Example path: '$.items[0].id'
    """
    if not path or not data:
        return None

    # Basic implementation for simple dot notation and array indexing
    try:
        value = data
        parts = path.split('.')
        for part in parts:
            if '[' in part and part.endswith(']'):
                key, index_str = part.split('[')
                index = int(index_str[:-1])
                if isinstance(value, dict) and key in value:
                    value = value[key]
                    if isinstance(value, list) and index < len(value):
                        value = value[index]
                    else:
                        logger.warning(f"JSONPath extraction failed at index {index} for key '{key}'. Value is not a list or index out of bounds.")
                        return None # Invalid index or not a list
                else:
                    logger.warning(f"JSONPath extraction failed: Key '{key}' not found in dictionary or value is not a dictionary.")
                    return None # Not a dict or key not found
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                logger.warning(f"JSONPath extraction failed: Key '{part}' not found in dictionary or value is not a dictionary.")
                return None # Not a dict or key not found
        return value
    except (ValueError, IndexError, TypeError, AttributeError) as e:
        logger.warning(f"Error extracting data with path '{path}': {e}")
        return None


def build_graph(router_llm: Any, worker_llm: Any) -> Any:
    """
    Builds and compiles the LangGraph StateGraph for the OpenAPI execution agent.
    """
    core_logic = OpenAPICoreLogic(worker_llm)
    router = OpenAPIRouter(router_llm)

    builder = StateGraph(BotState)

    # --- Node Definitions ---

    # Helper function to wrap core_logic methods.
    # This allows core_logic methods to be used as nodes in the graph
    # while ensuring state validation, error handling, and tracking the last executed node.
    def wrap_core_logic_method(fn):
        def node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            state = BotState.model_validate(state_dict)
            node_name = fn.__name__
            logger.debug(f"Executing core_logic node: {node_name}")
            # Track the node that just finished executing
            state.scratchpad['last_executed_node'] = node_name

            try:
                result = fn(state)
                return result.model_dump() if isinstance(result, BotState) else result
            except Exception as e:
                logger.error(f"Error in core_logic node {node_name}: {e}", exc_info=True)
                state.scratchpad['last_error'] = str(e)
                return state.model_dump()
        node.__name__ = fn.__name__
        return node

    # 1. The Planner Node
    # This node is the brain of the agent. It uses an LLM to analyze the user's query
    # and the current state to decide the next action (which node to execute next).
    def planner_node(state: BotState) -> Dict[str, Any]:
        """
        Analyzes user query and current state using an LLM to decide the next action.
        Considers the outcome of the previous step (last_executed_node, last_error).
        Sets state.scratchpad['planner_decision'] to indicate the chosen path,
        and potentially state.plan (for executor) and state.extracted_params.
        Implements proactive steps after schema parsing and routes general queries.
        """
        logger.debug("---PLANNER NODE---")
        query = state.user_input
        execution_graph = state.execution_graph
        openapi_schema = state.openapi_schema
        last_executed_node = state.scratchpad.get('last_executed_node')
        last_error = state.scratchpad.get('last_error')
        # Get the name of the node that ran immediately before the planner, if routed via planner
        last_executed_before_planner = state.scratchpad.get('last_executed_node_before_planner')


        logger.debug(f"User Query: {query}")
        logger.debug(f"Execution Graph Exists: {execution_graph is not None}")
        logger.debug(f"OpenAPI Schema Exists: {openapi_schema is not None}")
        logger.debug(f"Loop Counter: {state.loop_counter}")
        logger.debug(f"Last Executed Node: {last_executed_node}")
        logger.debug(f"Last Error: {last_error}")
        logger.debug(f"Last Executed Before Planner: {last_executed_before_planner}")


        prompt_context = "Analyze the user's request and the current state of the system to determine the next best action."

        state_description = f"""
Current State:
- User Input: "{query}"
- OpenAPI Specification Loaded: {'Yes' if openapi_schema else 'No'}
- Execution Graph Exists: {'Yes' if execution_graph else 'No'}
- Last Executed Node: {last_executed_node or 'None'}
- Last Error Encountered: {last_error or 'None'}
"""
        if openapi_schema: # Include schema summary status if schema is loaded
             state_description += f"- OpenAPI Schema Summary Available: {'Yes' if state.schema_summary else 'No'}\n"
             if state.identified_apis is not None:
                  state_description += f"- Identified APIs Available: Yes ({len(state.identified_apis)} found)\n"
             if state.generated_payloads is not None:
                  state_description += f"- Generated Payloads Available: Yes ({len(state.generated_payloads)} operations)\n"

        if execution_graph:
             state_description += f"- Execution Graph Description: {execution_graph.description or 'No description available.'}\n"
             if execution_graph.nodes:
                 node_summaries = "\n".join([f"  - {node.operationId}: {node.summary or 'No summary'}" for node in execution_graph.nodes[:10]])
                 state_description += f"- Available API Operations in Graph ({len(execution_graph.nodes)} total, showing first 10):\n{node_summaries}\n"
             else:
                 state_description += "- Execution Graph is empty.\n"


        # --- Explicit Task Completion Checks for the LLM ---
        task_completion_status = "No specific task completion detected in the last step."
        if last_executed_node == "identify_apis" and state.identified_apis is not None:
             task_completion_status = f"The 'identify_apis' node successfully ran. It identified {len(state.identified_apis)} APIs. This action likely fulfills user queries asking to list or see available APIs."
        elif last_executed_node == "describe_graph" and state.execution_graph and state.execution_graph.description:
             task_completion_status = f"The 'describe_graph' node successfully ran. It generated a description of the graph. This action likely fulfills user queries asking to describe the graph."
        elif last_executed_node == "get_graph_json" and state.execution_graph:
             task_completion_status = f"The 'get_graph_json' node successfully ran. It generated the JSON for the graph. This action likely fulfills user queries asking for the graph JSON."
        elif last_executed_node == "parse_openapi_spec" and state.openapi_schema:
             task_completion_status = f"The 'parse_openapi_spec' node successfully ran. It parsed the OpenAPI schema and generated a summary. This is a prerequisite for many tasks, and triggers proactive analysis."
        elif last_executed_node == "generate_execution_graph" and state.execution_graph:
             task_completion_status = f"The 'generate_execution_graph' node successfully ran. It generated an execution graph with explicit input mappings. This is a prerequisite for execution or description."
        elif last_executed_node == "generate_payloads" and state.generated_payloads is not None: # Check if payloads are generated (can be empty dict)
             task_completion_status = f"The 'generate_payloads' node successfully ran. It generated example payloads ({len(state.generated_payloads)} operations). This might fulfill a query asking for example payloads."

        state_description += f"\nTask Completion Status from Last Step:\n- {task_completion_status}\n"


        available_actions = {
            "parse_openapi_spec": "Parse a new OpenAPI specification provided by the user.",
            "identify_apis": "Analyze the loaded OpenAPI specification to identify relevant API operations.",
            "generate_payloads": "Generate example request payloads for identified API operations based on the schema.",
            "generate_execution_graph": "Generate a sequence/workflow (execution graph) of API calls based on the loaded schema and user goal.",
            "executor": "Execute the planned sequence of API calls from the existing execution graph.",
            "describe_graph": "Provide a natural language description of the existing execution graph.",
            "get_graph_json": "Output the JSON representation of the existing execution graph.",
            "answer_openapi_query": "Answer a general question about the loaded OpenAPI specification, identified APIs, generated payloads, or execution graph using the available state information.", # New action
            "handle_unknown": "Respond to the user when the request is unclear or cannot be fulfilled in the current state.",
            "handle_loop": "Respond to the user when the system appears to be stuck in a loop.",
            "responder": "Generate the final response to the user based on the results or current state.",
        }

        actions_description = "Available Actions (choose one to route to):\n" + "\n".join([f"- `{name}`: {desc}" for name, desc in available_actions.items()])

        # --- Refined Output Instruction for Planner with Proactive & Query Answering Logic ---
        output_instruction = """
Based on the user input and the current state, determine the single best 'next_action' from the Available Actions.
**CRITICAL DECISION MAKING PROCESS:**
1.  **Check for Errors:** If the 'Last Error Encountered' is not 'None', the `next_action` MUST be `responder` to report the error to the user.
2.  **Proactive Analysis after Parsing:** If the 'Last Executed Node' was `parse_openapi_spec` and `OpenAPI Specification Loaded` is 'Yes', and if `Identified APIs`, `Generated Payloads`, and `Execution Graph` are NOT yet fully populated in the state, the `next_action` should be one of the following proactive analysis steps in order: `identify_apis`, then `generate_payloads`, then `generate_execution_graph`. Choose the first of these steps that hasn't been successfully completed yet.
3.  **Check for Task Completion:** Review the 'Task Completion Status from Last Step' and the original 'User Input'. If the action performed by the 'Last Executed Node' directly and successfully fulfills the user's request (e.g., user asked "list APIs" and 'identify_apis' completed; user asked "describe graph" and 'describe_graph' completed), the `next_action` MUST be `responder` to provide the final output.
4.  **Handle General Queries:** If the user's input is primarily an informational question *about* the loaded OpenAPI specification, the identified APIs, generated payloads, or the execution graph (e.g., "What APIs are there?", "Describe the graph?", "What parameters does the 'createUser' API take?"), and the relevant data is available in the state, the `next_action` should be `answer_openapi_query`.
5.  **Plan Next Step:** If none of the above conditions are met (no error, no proactive step needed, last step did NOT fully complete the user's request, and it's not a general informational query), determine the necessary next step to make progress towards fulfilling the 'User Input' using the 'Available Actions'.

If the 'next_action' is `executor`, you MUST also provide a `plan` which is a JSON list of `operationId` strings from the execution graph nodes, representing the sequence of API calls to make. You should also extract any relevant `parameters` from the user query needed for these API calls and return them as a JSON object where keys are operationIds and values are parameter dictionaries.
If the 'next_action' is `generate_execution_graph`, capture the user's goal/instructions for graph generation.
For all other `next_action` values, the `plan` and `parameters` fields are optional but can be included if relevant.

Output ONLY a JSON object in the following format:
```json
{{
  "next_action": "chosen_action_name",
  "plan": ["operationId1", "operationId2", ...], // Required only if next_action is "executor"
  "parameters": {{ "operationId1": {{ "param1": "value1", ... }}, ... }}, // Optional, extracted from user query
  "reasoning": "Brief explanation of why this action was chosen, referencing the Critical Decision Making Process steps." // Optional, but helpful for debugging
}}
```
Ensure the `next_action` exactly matches one of the Available Actions names.
"""

        full_prompt = f"{prompt_context}\n\n{state_description}\n{actions_description}\n{output_instruction}\nUser Query: \"{query}\"\n\nOutput JSON:"

        planner_decision = "handle_unknown"
        planned_operation_ids = []
        extracted_params = {}
        reasoning = "Default fallback - LLM call failed or response unparseable."

        # --- Implement Proactive Logic before calling LLM ---
        # This handles the sequence after successful parsing programmatically
        if last_executed_node == "parse_openapi_spec" and openapi_schema:
             if state.identified_apis is None:
                  planner_decision = "identify_apis"
                  reasoning = "Proactively identifying APIs after parsing schema."
                  logger.info("Planner: Proactively routing to identify_apis after parsing.")
             elif state.generated_payloads is None:
                  planner_decision = "generate_payloads"
                  reasoning = "Proactively generating payloads after identifying APIs."
                  logger.info("Planner: Proactively routing to generate_payloads after identifying APIs.")
             elif state.execution_graph is None:
                  # If generating graph proactively, use the original user query as the goal/instructions
                  state.graph_generation_instructions = query # Capture user query as goal for graph generation
                  planner_decision = "generate_execution_graph"
                  reasoning = "Proactively generating execution graph after generating payloads."
                  logger.info("Planner: Proactively routing to generate_execution_graph after generating payloads.")
             else:
                  # All proactive steps complete, route to responder to confirm
                  planner_decision = "responder"
                  reasoning = "All proactive analysis steps (identify, payloads, graph) completed after parsing."
                  logger.info("Planner: All proactive steps complete after parsing, routing to responder.")

        # --- If not a proactive step, use LLM for planning ---
        # This block is only reached if the proactive conditions above were not met.
        if planner_decision == "handle_unknown": # Only call LLM if planner_decision hasn't been set proactively
            if state.loop_counter >= 2:
                 planner_decision = "handle_loop"
                 reasoning = "Detected potential loop based on router's loop counter."
                 logger.info("Planner: Routing to handle_loop due to loop counter.")
            elif last_error:
                 planner_decision = "responder"
                 reasoning = f"Previous node ({last_executed_node}) reported an error. Routing to responder."
                 logger.warning(f"Planner: Routing to responder due to error in {last_executed_node}.")
                 if 'last_error' in state.scratchpad:
                      del state.scratchpad['last_error']
            else:
                try:
                    llm_response = llm_call_helper(worker_llm, full_prompt)

                    parsed_response = parse_llm_json_output(llm_response)

                    if isinstance(parsed_response, dict) and 'next_action' in parsed_response:
                        determined_action = parsed_response['next_action']

                        if determined_action in available_actions:
                            planner_decision = determined_action
                            planned_operation_ids = parsed_response.get('plan', [])
                            extracted_params = parsed_response.get('parameters', {})
                            reasoning = parsed_response.get('reasoning', 'LLM provided action.')

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
                                    graph_op_ids = {node.operationId for node in execution_graph.nodes}
                                    valid_plan = [op_id for op_id in planned_operation_ids if op_id in graph_op_ids]
                                    if len(valid_plan) != len(planned_operation_ids):
                                         invalid_ids = set(planned_operation_ids) - graph_op_ids
                                         logger.warning(f"LLM planned execution of non-existent operationIds: {invalid_ids}. Using valid subset.")
                                         planned_operation_ids = valid_plan
                                         if not planned_operation_ids:
                                             planner_decision = "handle_unknown"
                                             reasoning = f"LLM planned execution of operationIds not found in graph: {invalid_ids}."

                        else:
                            logger.warning(f"LLM returned invalid next_action: {determined_action}. Defaulting to handle_unknown.")
                            planner_decision = "handle_unknown"
                            reasoning = f"LLM returned invalid action name: {determined_action}."

                    else:
                        logger.warning(f"LLM response not a valid JSON object with 'next_action': {llm_response[:500]}...")
                        planner_decision = "handle_unknown"
                        reasoning = "LLM response was not in the expected JSON format."

                except Exception as e:
                    logger.error(f"Error during LLM planning call or parsing: {e}", exc_info=True)
                    planner_decision = "handle_unknown"
                    reasoning = f"Error during planning: {e}"


        state.scratchpad['planner_decision'] = planner_decision
        state.plan = planned_operation_ids
        state.extracted_params = extracted_params
        state.scratchpad['planner_reasoning'] = reasoning

        logger.info(f"Planner Decision: {planner_decision}")
        if planned_operation_ids:
             logger.info(f"Planner Plan: {planned_operation_ids}")
        if extracted_params:
             logger.info(f"Planner Extracted Params: {extracted_params}")

        # Store the node that executed immediately before the planner,
        # before clearing the general 'last_executed_node'.
        # This is crucial for the responder when planner_last_decision is 'responder'.
        if last_executed_node:
             state.scratchpad['last_executed_node_before_planner'] = last_executed_node

        # Clear last_executed_node after the planner has considered it.
        # It will be set again by the next node that runs.
        if 'last_executed_node' in state.scratchpad:
             del state.scratchpad['last_executed_node']

        return state.model_dump()

    # 2. The Executor Node
    # This node is responsible for iterating through the planned API calls (state.plan)
    # and executing them, using explicit input mappings.
    def executor_node(state: BotState) -> Dict[str, Any]:
        """
        Executes the API calls specified in state.plan, using input mappings.
        Iterates through the plan, calls APIs, and updates state.results.
        Updates state.current_step.
        """
        logger.debug("---EXECUTOR NODE---")
        plan = state.plan
        current_step = state.current_step
        execution_graph = state.execution_graph
        openapi_schema = state.openapi_schema
        # extracted_params from the planner might contain initial parameters for the first step
        initial_params = state.extracted_params

        state.scratchpad['last_executed_node'] = "executor"

        if not execution_graph or not execution_graph.nodes:
             step_error = "Execution graph is missing or empty."
             state.scratchpad['execution_error'] = step_error
             state.current_step = len(plan) # Stop execution
             logger.error(f"Executor: {step_error}")
             # Add a result entry indicating the failure
             # Use a placeholder operationId if plan is empty, otherwise use current
             op_id_for_result = plan[current_step] if plan and current_step < len(plan) else "unknown_executor_error"
             state.results[op_id_for_result] = {"status": "error", "operation": op_id_for_result, "error": step_error}
             return state.model_dump()


        current_operation_id = plan[current_step]
        logger.info(f"Executor: Executing step {current_step + 1}/{len(plan)}: {current_operation_id}")

        # Find the node definition for the current operation in the execution graph
        current_node_definition: Optional[Node] = next(
            (node for node in execution_graph.nodes if node.operationId == current_operation_id),
            None
        )

        if not current_node_definition:
            step_error = f"Execution Error: OperationId '{current_operation_id}' not found in the execution graph nodes."
            state.scratchpad['execution_error'] = step_error
            state.current_step = len(plan) # Stop execution
            logger.error(step_error)
            # Add a result entry indicating the failure
            state.results[current_operation_id] = {"status": "error", "operation": current_operation_id, "error": step_error}
            return state.model_dump()

        # --- Parameter Binding using Input Mappings ---
        params_for_this_call: Dict[str, Any] = {}

        # Start with any initial parameters extracted by the planner for this operation
        if initial_params and current_operation_id in initial_params:
             params_for_this_call.update(initial_params.get(current_operation_id, {}))
             logger.debug(f"Executor: Applied initial parameters for {current_operation_id}: {params_for_this_call}")


        # Apply input mappings from previous step results
        if current_node_definition.input_mappings:
             logger.debug(f"Executor: Applying input mappings for {current_operation_id}: {current_node_definition.input_mappings}")
             for mapping in current_node_definition.input_mappings:
                 source_op_id = mapping.source_operation_id
                 source_path = mapping.source_data_path
                 target_param_name = mapping.target_parameter_name
                 # target_param_in = mapping.target_parameter_in # Not used in this basic simulation

                 if source_op_id in state.results:
                     source_result = state.results[source_op_id]
                     # Assuming source_result['data'] holds the actual API response body
                     source_data = source_result.get('data')

                     if source_data is not None:
                         # Extract data using the specified path
                         extracted_value = extract_data_with_jsonpath(source_data, source_path)

                         if extracted_value is not None:
                             # Apply transformation if specified (Placeholder)
                             if mapping.transformation:
                                 logger.warning(f"Executor: Transformation '{mapping.transformation}' specified for {target_param_name} but not implemented.")
                                 # TODO: Implement transformation logic here

                             # Add the extracted value to the parameters for the current call
                             # Note: This assumes a flat dictionary for parameters.
                             # For complex parameter structures (e.g., nested body), this needs refinement.
                             params_for_this_call[target_param_name] = extracted_value
                             logger.debug(f"Executor: Mapped data from {source_op_id}.{source_path} to {target_param_name}.")
                         else:
                             logger.warning(f"Executor: Could not extract data from {source_op_id}.{source_path} for {target_param_name}. Value is None.")
                     else:
                         logger.warning(f"Executor: Source data is None for {source_op_id}. Cannot apply mapping for {target_param_name}.")
                 else:
                     logger.warning(f"Executor: Source operation '{source_op_id}' not found in state.results for mapping to {target_param_name}. Ensure source node executed before target node.")


        # --- API Execution Logic (User Implementation Required) ---
        # This section simulates the API call using the constructed parameters.
        api_result_data = None
        step_error = None

        logger.info(f"Executor: Simulating API call for operationId: {current_operation_id} with parameters: {params_for_this_call}")

        try:
            # In a real implementation, you would use an HTTP client here
            # (like requests) to call the actual API endpoint, using the
            # openapi_schema to get the URL, method, and parameter details,
            # and passing params_for_this_call.

            # Example simulation:
            # Look up API details in the schema based on current_operation_id
            # api_details_from_schema = ... # Logic to find path, method etc.

            # Construct the actual request (URL, headers, body, query params)
            # based on api_details_from_schema and params_for_this_call.

            # Make the HTTP request:
            # response = requests.request(method, url, ...)

            # Process the response:
            # response_data = response.json() or response.text # Get response body
            # status_code = response.status_code

            # Simulated success/error based on operationId name
            if "error" in current_operation_id.lower():
                 raise Exception(f"Simulated error during execution of {current_operation_id}")

            # Simulated successful result structure (including a 'data' field for the response body)
            simulated_response_body = {"status": "success", "message": f"Operation {current_operation_id} executed.", "data": {"id": f"simulated_id_{current_operation_id}", "value": "example_value"}}
            api_result_data = {"status": "success", "operation": current_operation_id, "data": simulated_response_body, "parameters_used": params_for_this_call}
            logger.info(f"Executor: Simulated API call success for {current_operation_id}. Result: {api_result_data}")

        except Exception as e:
            logger.error(f"Executor: Error executing simulated API {current_operation_id}: {e}", exc_info=True)
            step_error = f"API Execution Error for {current_operation_id}: {e}"
            # Store error details in the result
            api_result_data = {"status": "error", "operation": current_operation_id, "error": str(e), "parameters_used": params_for_this_call}


        # Store the result in the state, indexed by operationId
        state.results[current_operation_id] = api_result_data

        if step_error:
            state.scratchpad['execution_error'] = step_error
            state.current_step = len(plan) # Stop execution on error
            logger.warning(f"Executor: Execution stopped due to error in {current_operation_id}")
        else:
            state.current_step += 1
            logger.debug(f"Executor: Moved to next step. current_step: {state.current_step}")

        return state.model_dump()

    # 3. The Responder Node
    # This node is the final output layer. It formats the response that the user sees.
    def responder_node(state: BotState) -> Dict[str, Any]:
        """
        Formats the final response to the user based on state.results,
        state.scratchpad['execution_error'], or other state information.
        Uses LLM to generate user-friendly text for data outputs.
        Sets state.final_response. Clears execution-related state.
        """
        logger.debug("---RESPONDER NODE---")
        results = state.results # Results are now a dictionary
        execution_error = state.scratchpad.get('execution_error')
        planner_last_decision = state.scratchpad.get('planner_decision')
        query = state.user_input
        # Get the name of the node that ran immediately before the planner.
        # This is the node whose output the responder might need to format if planner_last_decision is 'responder'.
        last_executed_before_planner = state.scratchpad.get('last_executed_node_before_planner')


        final_response_text = ""

        # Determine what kind of response is needed based on the planner's decision
        # AND potentially the node that ran *before* the planner (if routed via planner).

        if planner_last_decision == "executor":
            # Response based on API execution results.
            if execution_error:
                final_response_text = f"An error occurred during API execution: {execution_error}\n"
                if results:
                    # Summarize partial results
                    final_response_text += "Results obtained:\n" + json.dumps(results, indent=2)
                else:
                    final_response_text += "No results were obtained."
            elif results:
                logger.info("Responder: Using LLM to summarize execution results.")
                # Provide the full results dictionary to the LLM
                prompt = f"""
                The user asked: "{query}"
                API execution completed with the following results (indexed by operationId):
                ```json
                {json.dumps(results, indent=2)}
                ```
                Please summarize these results in a user-friendly way that directly addresses the user's original query.
                Mention the status of each executed operation.
                """
                try:
                    llm_response = llm_call_helper(worker_llm, prompt)
                    final_response_text = llm_response.strip()
                except Exception as e:
                    logger.error(f"Error calling LLM for execution summary: {e}", exc_info=True)
                    final_response_text = "Execution completed. Here are the raw results:\n" + json.dumps(results, indent=2)

            else:
                 final_response_text = f"Execution completed, but no results were returned for query: '{query}'."

        # Handle responses for specific core logic nodes that route directly to responder
        # These nodes set state.response with a pre-formatted message (like handle_unknown/loop)
        # In this case, planner_last_decision will be the name of the node (e.g., 'handle_unknown')
        elif planner_last_decision in ["handle_unknown", "handle_loop", "answer_openapi_query"]: # Added answer_openapi_query here
             # These nodes are designed to set state.response with the final message
             final_response_text = state.response or f"Operation {planner_last_decision} completed."
             logger.debug(f"Responder: Handling direct route from {planner_last_decision}. Using state.response.")

        elif planner_last_decision == "responder":
             # The planner decided it's time to respond. Now, figure out *what* to respond about
             # based on the node that ran *before* the planner (if any).
             # We use last_executed_before_planner which was set by the planner node
             # before clearing 'last_executed_node'.
             logger.debug(f"Responder: Planner decided 'responder'. Checking last executed node before planner: {last_executed_before_planner}")

             if last_executed_before_planner == "identify_apis" and state.identified_apis:
                  logger.info("Responder: Formatting output from identify_apis.")
                  prompt = f"""
                  The user asked: "{query}"
                  The identification of APIs completed successfully.
                  Based on the identified APIs below, please provide a user-friendly response that directly addresses the user's request to list or see available APIs.

                  Identified APIs:\n```json\n{json.dumps(state.identified_apis, indent=2)}\n```

                  User-friendly response:
                  """
                  try:
                       llm_response = llm_call_helper(worker_llm, prompt)
                       final_response_text = llm_response.strip()
                  except Exception as e:
                       logger.error(f"Error calling LLM for identify_apis response synthesis: {e}", exc_info=True)
                       final_response_text = "Identified APIs. Here is the raw data:\n" + json.dumps(state.identified_apis, indent=2)

             elif last_executed_before_planner == "describe_graph" and state.execution_graph and state.execution_graph.description:
                  logger.info("Responder: Formatting output from describe_graph.")
                  prompt = f"""
                  The user asked: "{query}"
                  The graph description was generated.
                  Here is the description:
                  {state.execution_graph.description}

                  Please provide a user-friendly response based on this description.
                  """
                  try:
                       llm_response = llm_call_helper(worker_llm, prompt)
                       final_response_text = llm_response.strip()
                  except Exception as e:
                       logger.error(f"Error calling LLM for describe_graph response synthesis: {e}", exc_info=True)
                       final_response_text = f"Here is the graph description:\n{state.execution_graph.description}"

             elif last_executed_before_planner == "get_graph_json" and state.execution_graph:
                  # This is a case where raw output is expected
                  logger.info("Responder: Outputting raw graph JSON.")
                  final_response_text = f"### Execution Graph JSON:\n\n```json\n{state.execution_graph.model_dump_json(indent=2)}\n```"

             elif last_executed_before_planner == "generate_payloads" and state.generated_payloads:
                  logger.info("Responder: Formatting output from generate_payloads.")
                  prompt = f"""
                  The user asked: "{query}"
                  Example payloads were generated.
                  Here are the generated payloads:\n```json\n{json.dumps(state.generated_payloads, indent=2)}\n```

                  Please provide a user-friendly response summarizing the generated payloads.
                  """
                  try:
                       llm_response = llm_call_helper(worker_llm, prompt)
                       final_response_text = llm_response.strip()
                  except Exception as e:
                       logger.error(f"Error calling LLM for generate_payloads response synthesis: {e}", exc_info=True)
                       final_response_text = "Generated payloads. Here is the raw data:\n" + json.dumps(state.generated_payloads, indent=2)

             elif last_executed_before_planner == "generate_execution_graph" and state.execution_graph:
                   logger.info("Responder: Formatting output from generate_execution_graph.")
                   # After generating the graph, we likely want to describe it.
                   # Use the description already in the graph object itself if available, or ask LLM to summarize nodes/edges.
                   if state.execution_graph and state.execution_graph.description:
                       final_response_text = f"Successfully generated the execution graph. Here is a description:\n{state.execution_graph.description}"
                   elif state.execution_graph:
                       prompt = f"""
                       The user asked: "{query}"
                       An execution graph was successfully generated.
                       The graph contains {len(state.execution_graph.nodes)} nodes and {len(state.execution_graph.edges)} edges.
                       Please provide a user-friendly confirmation that the graph was generated and briefly mention its size.
                       """
                       try:
                           llm_response = llm_call_helper(worker_llm, prompt)
                           final_response_text = llm_response.strip()
                       except Exception as e:
                           logger.error(f"Error calling LLM for graph generation confirmation: {e}", exc_info=True)
                           final_response_text = f"Successfully generated the execution graph with {len(state.execution_graph.nodes)} nodes and {len(state.execution_graph.edges)} dependencies."
                   else:
                       # Fallback if graph generation node ran but graph is not in state (indicates an issue)
                       final_response_text = "Attempted to generate the execution graph, but the graph was not found in the state."
                       logger.warning("Responder: generate_execution_graph ran, but state.execution_graph is None.")


             elif state.response:
                  # Fallback if the last executed node wasn't one of the specific data tools,
                  # but there's a message in state.response (e.g., from parse_openapi_spec success).
                  logger.info("Responder: Using state.response as final text.")
                  final_response_text = state.response
             else:
                  # Generic fallback if no specific data or message is found.
                  logger.warning("Responder: No specific data or message found for final response when planner decided 'responder'.")
                  final_response_text = "Processing complete."

        else:
             # This case should ideally not be hit if the planner always routes to a defined node.
             # As a fallback, use state.response if available.
             logger.warning(f"Responder node called with unexpected planner_last_decision: {planner_last_decision}")
             final_response_text = state.response or "An unexpected state occurred."


        # Fallback if final_response_text is still empty (shouldn't happen with the above logic, but good practice)
        if not final_response_text:
             logger.warning("Responder: final_response_text is still empty after all logic.")
             final_response_text = state.response or "Processing complete."


        logger.info(f"Setting final_response: {final_response_text[:200]}...")
        logger.info(f"Full final_response set: {final_response_text}")

        state.final_response = final_response_text

        # Clear execution-related state fields after responding
        state.plan = []
        # Keep state.results as it might be needed for subsequent turns if user asks about results
        state.current_step = 0
        if 'execution_error' in state.scratchpad:
             del state.scratchpad['execution_error']
        if 'planner_decision' in state.scratchpad:
             del state.scratchpad['planner_decision']
        # Clear the last executed node before planner after responding
        if 'last_executed_node_before_planner' in state.scratchpad:
             del state.scratchpad['last_executed_node_before_planner']

        # Keep state.response if it contains a message from a core_logic node
        # It might be useful for the next turn if the user asks a follow-up.
        # Or clear it if you want a clean slate for the next turn.
        # Decided to keep state.response for now.

        return state.model_dump()

    # --- Add nodes to the graph builder ---

    tool_methods = {
        "parse_openapi_spec": core_logic.parse_openapi_spec,
        "identify_apis": core_logic.identify_apis,
        "generate_payloads": core_logic.generate_payloads,
        "generate_execution_graph": core_logic.generate_execution_graph,
        "describe_graph": core_logic.describe_graph,
        "get_graph_json": core_logic.get_graph_json,
        "handle_unknown": core_logic.handle_unknown,
        "handle_loop": core_logic.handle_loop,
        "answer_openapi_query": core_logic.answer_openapi_query, # Added new tool
    }
    for name, fn in tool_methods.items():
        builder.add_node(name, wrap_core_logic_method(fn))

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("responder", responder_node)

    # --- Define Graph Flow ---

    # 1. Entry point: Start with the existing router for initial high-level intent.
    builder.add_conditional_edges(
        START,
        router.route,
        {intent: intent for intent in router.AVAILABLE_INTENTS}
    )

    # 2. After the planner node executes, route based on its decision.
    def planner_router(state: BotState) -> str:
        decision = state.scratchpad.get('planner_decision', 'handle_unknown')
        logger.debug(f"Planner Router: Routing based on decision '{decision}'")
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
        elif decision == "responder":
             return "responder"
        elif decision == "answer_openapi_query": # Added routing for the new tool
             return "answer_openapi_query"
        else:
            logger.warning(f"Planner Router: Unhandled planner decision '{decision}', routing to handle_unknown.")
            return "handle_unknown"

    builder.add_conditional_edges(
        "planner",
        planner_router,
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
            "responder": "responder",
            "answer_openapi_query": "answer_openapi_query", # Added routing for the new tool
        }
    )

    # 3. After core_logic nodes:
    CORE_LOGIC_TO_PLANNER = [
        "parse_openapi_spec",
        "identify_apis",
        "generate_payloads",
        "generate_execution_graph",
    ]

    for node_name in CORE_LOGIC_TO_PLANNER:
        if node_name in tool_methods:
             # These nodes route to the planner. The planner will store 'node_name'
             # in 'last_executed_node_before_planner' before clearing 'last_executed_node'.
             builder.add_edge(node_name, "planner")
             logger.debug(f"Added edge: {node_name} -> planner")

    CORE_LOGIC_TO_RESPONDER = [
        "describe_graph",
        "get_graph_json",
        "handle_unknown",
        "handle_loop",
        "answer_openapi_query", # Added new tool to route directly to responder after execution
    ]

    for node_name in CORE_LOGIC_TO_RESPONDER:
         if node_name in tool_methods:
              # These nodes route directly to the responder.
              # In the responder, planner_last_decision will be the node_name itself.
              # The responder logic handles these cases explicitly *before* the planner_last_decision == 'responder' check.
              builder.add_edge(node_name, "responder")
              logger.debug(f"Added edge: {node_name} -> responder")

    # 4. After the executor node executes, decide whether to loop back to the executor
    # for the next step in the plan or move to the responder if the plan is complete.
    def executor_router(state: BotState) -> str:
        logger.debug(f"Executor Router: current_step={state.current_step}, plan_length={len(state.plan)}")
        if state.current_step < len(state.plan):
            logger.debug("Executor Router: More steps in plan, looping back to executor.")
            return "continue_execution"
        else:
            logger.debug("Executor Router: Plan complete, routing to responder.")
            return "finish_execution"

    builder.add_conditional_edges(
        "executor",
        executor_router,
        {
            "continue_execution": "executor",
            "finish_execution": "responder",
        }
    )

    # 5. From the responder node, the graph ends, indicating the conversation turn is complete.
    builder.add_edge("responder", END)

    app = builder.compile(checkpointer=MemorySaver())
    logger.info("Graph compiled with dynamic API execution capabilities (Planner/Executor/Responder).")
    return app
