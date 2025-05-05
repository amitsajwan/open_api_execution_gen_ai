# filename: graph.py
import logging
from typing import Any, Dict, List, Optional, Tuple
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# If using a different checkpointer (e.g., SQL), import it here

# Import state model and core logic
from models import BotState # Ensure BotState is imported
from core_logic import OpenAPICoreLogic # Ensure OpenAPICoreLogic is imported
from router import OpenAPIRouter # Ensure OpenAPIRouter is imported

logger = logging.getLogger(__name__)

# --- Responder Node Function ---
def finalize_response(state: BotState) -> Dict[str, Any]: # Return type hint is Dict[str, Any]
    """
    Sets the final_response based on the last intermediate response from a node.
    Clears the intermediate response field.
    Returns a dictionary of state updates.
    """
    tool_name = "responder"
    state.update_scratchpad_reason(tool_name, "Finalizing response for user.")
    logger.debug("Executing responder node (finalize_response).")

    updates: Dict[str, Any] = {} # Initialize updates dictionary

    # Prioritize state.response for the final message
    if state.response:
        updates['final_response'] = state.response
        logger.debug(f"Set final_response from state.response: '{state.response[:100]}...'")
    elif not state.final_response: # Only use fallback if final_response is not already set
        # Fallback if intermediate response is also missing
        updates['final_response'] = "I've completed the requested action, but there wasn't a specific message to display."
        logger.warning("Responder: No intermediate 'response' found to set as 'final_response'. Using default.")
        state.update_scratchpad_reason(tool_name, "Warning: No intermediate response found. Used default final response.")

    # Clear the intermediate response field for the next turn in the updates
    updates['response'] = None

    state.update_scratchpad_reason(tool_name, "Final response set.")

    # The responder is typically the end of a chain, so it implicitly transitions to END
    # It should not return a '__next__' key unless it's routing elsewhere.
    # For this graph, it's the end, so no '__next__' is needed in the return dictionary.

    return updates # <-- Return the updates dictionary


# --- Graph Definition ---
def build_graph(router_llm: Any, worker_llm: Any) -> StateGraph:
    """
    Builds and compiles the LangGraph StateGraph for the OpenAPI analysis agent.

    Args:
        router_llm: The LLM instance for routing.
        worker_llm: The LLM instance for core tasks.

    Returns:
        A compiled LangGraph StateGraph application.
    """
    logger.info("Building LangGraph...")

    # Instantiate core logic and router
    core_logic = OpenAPICoreLogic(worker_llm=worker_llm)
    router = OpenAPIRouter(router_llm=router_llm)

    # Define the graph
    builder = StateGraph(BotState)

    # Add nodes for each distinct step/tool
    builder.add_node("router", router.route) # Router node
    builder.add_node("parse_openapi_spec", core_logic.parse_openapi_spec)
    builder.add_node("identify_apis", core_logic.identify_apis)
    builder.add_node("generate_payloads", core_logic.generate_payloads)
    builder.add_node("generate_execution_graph", core_logic.generate_execution_graph)
    builder.add_node("describe_graph", core_logic.describe_graph)
    builder.add_node("get_graph_json", core_logic.get_graph_json)
    builder.add_node("answer_openapi_query", core_logic.answer_openapi_query)
    builder.add_node("handle_unknown", core_logic.handle_unknown)
    builder.add_node("handle_loop", core_logic.handle_loop)
    builder.add_node("plan_execution", core_logic.plan_execution) # Added plan_execution node
    builder.add_node("responder", finalize_response) # Responder node

    # 1. Set the entry point
    builder.set_entry_point("router")

    # 2. Add edges
    # Conditional edge from the router based on the __next__ key in the state dictionary
    builder.add_conditional_edges(
        "router",  # Source node
        lambda state: state.model_dump().get("__next__"), # <-- Condition function: Access __next__ from state dictionary using model_dump
        { # Mapping from __next__ value to next node name
            "parse_openapi_spec": "parse_openapi_spec",
            "plan_execution": "plan_execution",
            "identify_apis": "identify_apis",
            "generate_payloads": "generate_payloads",
            "generate_execution_graph": "generate_execution_graph",
            "describe_graph": "describe_graph",
            "get_graph_json": "get_graph_json",
            "answer_openapi_query": "answer_openapi_query",
            "handle_unknown": "handle_unknown",
            "handle_loop": "handle_loop",
            "responder": "responder", # Router might sometimes route directly to responder (e.g., empty input handled by router)
            # Add other potential router outputs here if any
        }
    )

    # Default flow after parsing: identify -> payloads -> graph -> describe -> responder
    # Conditional edge after parse_openapi_spec based on the __next__ key in the state dictionary
    builder.add_conditional_edges(
         "parse_openapi_spec",
         lambda state: state.model_dump().get("__next__"), # <-- Condition function: Access __next__ from state dictionary using model_dump
         {
             "identify_apis": "identify_apis",
             "responder": "responder", # Route to responder on failure
             # Add other potential next nodes from parse_openapi_spec if any
         }
    )


    # After identifying APIs, generate payload descriptions
    # Conditional edge after identify_apis based on the __next__ key in the state dictionary
    builder.add_conditional_edges(
         "identify_apis",
         lambda state: state.model_dump().get("__next__"), # <-- Condition function: Access __next__ from state dictionary using model_dump
         {
             "generate_payloads": "generate_payloads",
             "responder": "responder", # Route to responder on failure
         }
    )

    # After generating payload descriptions, generate the graph description
    # Conditional edge after generate_payloads based on the __next__ key in the state dictionary
    builder.add_conditional_edges(
         "generate_payloads",
         lambda state: state.model_dump().get("__next__"), # <-- Condition function: Access __next__ from state dictionary using model_dump
         {
             "generate_execution_graph": "generate_execution_graph",
             "responder": "responder", # Route to responder on failure
         }
    )

    # After generating the graph description, describe it
    # Conditional edge after generate_execution_graph based on the __next__ key in the state dictionary
    builder.add_conditional_edges(
         "generate_execution_graph",
         lambda state: state.model_dump().get("__next__"), # <-- Condition function: Access __next__ from state dictionary using model_dump
         {
             "describe_graph": "describe_graph",
             "responder": "responder", # Route to responder on failure
         }
    )


    # After describing the graph, route to the responder
    # Conditional edge after describe_graph based on the __next__ key in the state dictionary
    builder.add_conditional_edges(
        "describe_graph",
        lambda state: state.model_dump().get("__next__"), # <-- Condition function: Access __next__ from state dictionary using model_dump
        {
            "responder": "responder", # Assuming describe_graph always goes to responder
            # Add other potential next nodes from describe_graph if any
        }
    )


    # The plan_execution node is available via the router but not part of the default Parse->... flow.
    # If plan_execution runs, route based on the __next__ key in the state dictionary
    builder.add_conditional_edges(
        "plan_execution",
        lambda state: state.model_dump().get("__next__"), # <-- Condition function: Access __next__ from state dictionary using model_dump
        {
            "responder": "responder", # Assuming plan_execution always goes to responder
            # Add other potential next nodes from plan_execution if any
        }
    )


    # Other specific action nodes route to the responder after completing their task
    # Conditional edges based on the __next__ key in the state dictionary
    builder.add_conditional_edges(
        "get_graph_json",
        lambda state: state.model_dump().get("__next__"), # <-- Condition function: Access __next__ from state dictionary using model_dump
        {"responder": "responder"}
    )
    builder.add_conditional_edges(
        "answer_openapi_query",
        lambda state: state.model_dump().get("__next__"), # <-- Condition function: Access __next__ from state dictionary using model_dump
        {"responder": "responder"}
    )
    builder.add_conditional_edges(
        "handle_unknown",
        lambda state: state.model_dump().get("__next__"), # <-- Condition function: Access __next__ from state dictionary using model_dump
        {"responder": "responder"}
    )
    builder.add_conditional_edges(
        "handle_loop",
        lambda state: state.model_dump().get("__next__"), # <-- Condition function: Access __next__ from state dictionary using model_dump
        {"responder": "responder"}
    )


    # 3. From the responder node, the graph ends.
    # The responder node doesn't need a conditional edge because it's the end.
    # It should not return a '__next__' key.
    builder.add_edge("responder", END)


    # Compile the graph
    # checkpointer is passed when running the app instance, not during compile
    app = builder.compile()
    logger.info("Graph compiled (without API execution capabilities).")
    return app

# Note: The core_logic methods (parse_openapi_spec, identify_apis, etc.) and
# the router.route method must return a dictionary of state updates including
# the '__next__' key to correctly drive these conditional edges.
