import logging
import json
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END

from models import BotState, GraphOutput
from core_logic import OpenAPICoreLogic
from router import OpenAPIRouter

from utils import llm_call_helper, parse_llm_json_output, MemorySaver # Assuming MemorySaver is in utils

logger = logging.getLogger(__name__)

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
        """
        logger.debug("---PLANNER NODE---")
        query = state.user_input
        execution_graph = state.execution_graph
        openapi_schema = state.openapi_schema
        last_executed_node = state.scratchpad.get('last_executed_node')
        last_error = state.scratchpad.get('last_error')

        logger.debug(f"User Query: {query}")
        logger.debug(f"Execution Graph Exists: {execution_graph is not None}")
        logger.debug(f"OpenAPI Schema Exists: {openapi_schema is not None}")
        logger.debug(f"Loop Counter: {state.loop_counter}")
        logger.debug(f"Last Executed Node: {last_executed_node}")
        logger.debug(f"Last Error: {last_error}")

        prompt_context = "Analyze the user's request and the current state of the system to determine the next best action."

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
                 node_summaries = "\n".join([f"  - {node.operationId}: {node.summary or 'No summary'}" for node in execution_graph.nodes[:10]])
                 state_description += f"- Available API Operations in Graph ({len(execution_graph.nodes)} total, showing first 10):\n{node_summaries}\n"
             else:
                 state_description += "- Execution Graph is empty.\n"
        if openapi_schema and not execution_graph:
             state_description += "- OpenAPI Schema is loaded and available for analysis/graph building.\n"

        # --- Explicit Task Completion Checks for the LLM ---
        task_completion_status = "No specific task completion detected in the last step."
        if last_executed_node == "identify_apis" and state.identified_apis is not None:
             task_completion_status = f"The 'identify_apis' node successfully ran. It identified {len(state.identified_apis)} APIs. This action likely fulfills user queries asking to list or see available APIs."
        elif last_executed_node == "describe_graph" and state.execution_graph and state.execution_graph.description:
             task_completion_status = f"The 'describe_graph' node successfully ran. It generated a description of the graph. This action likely fulfills user queries asking to describe the graph."
        elif last_executed_node == "get_graph_json" and state.execution_graph:
             task_completion_status = f"The 'get_graph_json' node successfully ran. It generated the JSON for the graph. This action likely fulfills user queries asking for the graph JSON."
        elif last_executed_node == "parse_openapi_spec" and state.openapi_schema:
             task_completion_status = f"The 'parse_openapi_spec' node successfully ran. It parsed the OpenAPI schema. This is a prerequisite for many tasks."
        elif last_executed_node == "generate_execution_graph" and state.execution_graph:
             task_completion_status = f"The 'generate_execution_graph' node successfully ran. It generated an execution graph. This is a prerequisite for execution or description."
        elif last_executed_node == "generate_payloads" and state.generated_payloads:
             task_completion_status = f"The 'generate_payloads' node successfully ran. It generated example payloads. This might fulfill a query asking for example payloads."

        state_description += f"\nTask Completion Status from Last Step:\n- {task_completion_status}\n"


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
            "responder": "Generate the final response to the user based on the results or current state.",
        }

        actions_description = "Available Actions (choose one to route to):\n" + "\n".join([f"- `{name}`: {desc}" for name, desc in available_actions.items()])

        # --- Refined Output Instruction for Planner ---
        output_instruction = """
Based on the user input and the current state, determine the single best 'next_action' from the Available Actions.
**CRITICAL DECISION MAKING PROCESS:**
1.  **Check for Errors:** If the 'Last Error Encountered' is not 'None', the `next_action` MUST be `responder` to report the error to the user.
2.  **Check for Task Completion:** Review the 'Task Completion Status from Last Step' and the original 'User Input'. If the action performed by the 'Last Executed Node' directly and successfully fulfills the user's request (e.g., user asked "list APIs" and 'identify_apis' completed; user asked "describe graph" and 'describe_graph' completed), the `next_action` MUST be `responder` to provide the final output.
3.  **Plan Next Step:** If there was no error and the last step did NOT fully complete the user's request, determine the necessary next step to make progress towards fulfilling the 'User Input' using the 'Available Actions'.

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

        if 'last_executed_node' in state.scratchpad:
             del state.scratchpad['last_executed_node']

        return state.model_dump()

    # 2. The Executor Node
    # This node is responsible for iterating through the planned API calls (state.plan)
    # and executing them.
    def executor_node(state: BotState) -> Dict[str, Any]:
        """
        Executes the API calls specified in state.plan.
        Iterates through the plan, calls APIs, and updates state.results.
        Updates state.current_step.
        """
        logger.debug("---EXECUTOR NODE---")
        plan = state.plan
        current_step = state.current_step
        execution_graph = state.execution_graph
        openapi_schema = state.openapi_schema
        extracted_params = state.extracted_params

        state.scratchpad['last_executed_node'] = "executor"

        if not plan or current_step >= len(plan):
            logger.debug("Executor: Execution plan finished or empty.")
            state.scratchpad['executor_status'] = "completed"
            return state.model_dump()

        current_operation_id = plan[current_step]
        logger.info(f"Executor: Executing step {current_step + 1}/{len(plan)}: {current_operation_id}")

        api_result_data = None
        step_error = None

        try:
            # --- API Execution Logic (User Implementation Required) ---
            logger.info(f"Executor: Simulating API call for operationId: {current_operation_id}")
            if "error" in current_operation_id.lower():
                 raise Exception(f"Simulated error during execution of {current_operation_id}")

            api_node_details = next((node for node in (execution_graph.nodes if execution_graph else []) if node.operationId == current_operation_id), None)

            api_details_from_schema = {"path": "/simulated/path", "method": "get"}

            params_for_this_call = extracted_params.get(current_operation_id, {})

            api_result_data = {"status": "success", "operation": current_operation_id, "data": f"Simulated data for {current_operation_id}"}
            logger.info(f"Executor: Simulated API call success for {current_operation_id}. Result: {api_result_data}")

        except Exception as e:
            logger.error(f"Executor: Error executing API {current_operation_id}: {e}", exc_info=True)
            step_error = f"API Execution Error for {current_operation_id}: {e}"
            api_result_data = {"status": "error", "operation": current_operation_id, "error": str(e)}

        state.results.append(api_result_data)

        if step_error:
            state.scratchpad['execution_error'] = step_error
            state.current_step = len(plan)
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
        results = state.results
        execution_error = state.scratchpad.get('execution_error')
        planner_last_decision = state.scratchpad.get('planner_decision')
        query = state.user_input

        final_response_text = ""

        if planner_last_decision == "executor":
            if execution_error:
                final_response_text = f"An error occurred during API execution: {execution_error}\n"
                if results:
                    final_response_text += "Partial results obtained:\n" + json.dumps(results, indent=2)
                else:
                    final_response_text += "No results were obtained."
            elif results:
                logger.info("Responder: Using LLM to summarize execution results.")
                prompt = f"""
                The user asked: "{query}"
                API execution completed with the following results:
                ```json
                {json.dumps(results, indent=2)}
                ```
                Please summarize these results in a user-friendly way that directly addresses the user's original query.
                """
                try:
                    llm_response = llm_call_helper(worker_llm, prompt)
                    final_response_text = llm_response.strip()
                except Exception as e:
                    logger.error(f"Error calling LLM for execution summary: {e}", exc_info=True)
                    final_response_text = "Execution completed. Here are the raw results:\n" + json.dumps(results, indent=2)

            else:
                 final_response_text = f"Execution completed, but no results were returned for query: '{query}'."

        elif planner_last_decision in ["identify_apis", "describe_graph", "get_graph_json", "generate_payloads", "generate_execution_graph"]:
             logger.info(f"Responder: Using LLM to format output from {planner_last_decision}.")
             prompt = f"""
             The user asked: "{query}"
             The previous step ({planner_last_decision}) completed successfully.
             Based on the current state, please provide a user-friendly response that directly addresses the user's request.

             Current State Information (Relevant to {planner_last_decision}):
             """
             if planner_last_decision == "identify_apis" and state.identified_apis:
                 prompt += f"\nIdentified APIs:\n```json\n{json.dumps(state.identified_apis, indent=2)}\n```"
             elif planner_last_decision == "describe_graph" and state.execution_graph and state.execution_graph.description:
                 prompt += f"\nExecution Graph Description:\n{state.execution_graph.description}"
             elif planner_last_decision == "get_graph_json" and state.execution_graph:
                 prompt += f"\nExecution Graph JSON:\n```json\n{state.execution_graph.model_dump_json(indent=2)}\n```"
             elif planner_last_decision == "generate_payloads" and state.generated_payloads:
                 prompt += f"\nGenerated Payloads:\n```json\n{json.dumps(state.generated_payloads, indent=2)}\n```"
             elif planner_last_decision == "generate_execution_graph" and state.execution_graph:
                  prompt += f"\nExecution Graph Description:\n{state.execution_graph.description or 'No description available.'}"
                  if state.execution_graph.nodes:
                       prompt += f"\nGraph contains {len(state.execution_graph.nodes)} nodes and {len(state.execution_graph.edges)} edges."
             elif state.response:
                 prompt += f"\nPrevious node message: {state.response}"
             else:
                 prompt += "\nNo specific data available from the previous step."

             # Removed the error check here, as the planner should route errors directly to responder
             # without hitting this specific data formatting block.

             prompt += "\n\nUser-friendly response:"

             try:
                  llm_response = llm_call_helper(worker_llm, prompt)
                  final_response_text = llm_response.strip()
             except Exception as e:
                  logger.error(f"Error calling LLM for {planner_last_decision} response synthesis: {e}", exc_info=True)
                  # Fallback to basic formatting if LLM call fails
                  if state.response:
                      final_response_text = f"(From {planner_last_decision}): {state.response}"
                  else:
                       final_response_text = f"Operation {planner_last_decision} completed."

        elif planner_last_decision in ["handle_unknown", "handle_loop"]:
             final_response_text = state.response or f"Operation {planner_last_decision} completed."
        # Added explicit handling for the case where planner_last_decision is 'responder' itself
        elif planner_last_decision == "responder":
             # This shouldn't typically happen in a clean flow, but as a fallback,
             # just use whatever might be in state.response or a generic message.
             logger.warning("Responder node called when planner_last_decision was already 'responder'.")
             final_response_text = state.response or "Processing complete."


        # Fallback response if planner_last_decision was not set or not handled above.
        if not final_response_text and state.response:
             final_response_text = state.response
        elif not final_response_text:
             final_response_text = "Processing complete."

        logger.info(f"Setting final_response: {final_response_text[:200]}...")
        logger.info(f"Full final_response set: {final_response_text}")

        state.final_response = final_response_text

        state.plan = []
        state.results = []
        state.current_step = 0
        if 'execution_error' in state.scratchpad:
             del state.scratchpad['execution_error']
        if 'planner_decision' in state.scratchpad:
             del state.scratchpad['planner_decision']

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
             builder.add_edge(node_name, "planner")
             logger.debug(f"Added edge: {node_name} -> planner")

    CORE_LOGIC_TO_RESPONDER = [
        "describe_graph",
        "get_graph_json",
        "handle_unknown",
        "handle_loop",
    ]

    for node_name in CORE_LOGIC_TO_RESPONDER:
         if node_name in tool_methods:
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
