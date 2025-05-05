# filename: graph.py
import logging
from typing import Any, Dict, List, Optional, Tuple
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# If using a different checkpointer (e.g., SQL), import it here

# Import state model and core logic
from models import BotState
from core_logic import OpenAPICoreLogic
from router import OpenAPIRouter # Import the router

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
    # We don't need a __next__ key here because there are no conditional edges *from* responder
    # If you had edges *from* responder, you would add updates['__next__'] = next_node_name

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
    # Conditional edge from the router based on its return value (__next__ key)
    builder.add_conditional_edges(
        "router",  # Source node
        lambda state: state.get("__next__"), # <-- Condition function: extract __next__ from state updates
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
            # Add other potential router outputs here if any
        }
    )

    # Default flow after parsing: identify -> payloads -> graph -> describe -> responder
    # Add conditional edge after parse_openapi_spec to check for success
    # (Assuming parse_openapi_spec returns updates['__next__'] = 'identify_apis' on success
    # or updates['__next__'] = 'responder' or 'handle_unknown' on failure)
    # Let's update parse_openapi_spec to return updates['__next__'] = 'identify_apis' on success
    # and updates['__next__'] = 'responder' on failure (with error message in state.response)
    # This requires modifying core_logic.parse_openapi_spec to return a dictionary with __next__
    # For now, assuming parse_openapi_spec returns updates with '__next__'
    builder.add_conditional_edges(
         "parse_openapi_spec",
         lambda state: state.get("__next__"), # Extract __next__ from updates
         {
             "identify_apis": "identify_apis",
             "responder": "responder", # Route to responder on failure
             # Add other potential next nodes from parse_openapi_spec if any
         }
    )


    # After identifying APIs, generate payload descriptions
    # Assuming identify_apis returns updates['__next__'] = 'generate_payloads' on success
    # and updates['__next__'] = 'responder' on failure
    builder.add_conditional_edges(
         "identify_apis",
         lambda state: state.get("__next__"), # Extract __next__ from updates
         {
             "generate_payloads": "generate_payloads",
             "responder": "responder", # Route to responder on failure
         }
    )

    # After generating payload descriptions, generate the graph description
    # Assuming generate_payloads returns updates['__next__'] = 'generate_execution_graph' on success
    # and updates['__next__'] = 'responder' on failure
    builder.add_conditional_edges(
         "generate_payloads",
         lambda state: state.get("__next__"), # Extract __next__ from updates
         {
             "generate_execution_graph": "generate_execution_graph",
             "responder": "responder", # Route to responder on failure
         }
    )

    # After generating the graph description, describe it
    # Assuming generate_execution_graph returns updates['__next__'] = 'describe_graph' on success
    # and updates['__next__'] = 'responder' on failure
    builder.add_conditional_edges(
         "generate_execution_graph",
         lambda state: state.get("__next__"), # Extract __next__ from updates
         {
             "describe_graph": "describe_graph",
             "responder": "responder", # Route to responder on failure
         }
    )


    # After describing the graph, route to the responder
    # Assuming describe_graph returns updates['__next__'] = 'responder'
    builder.add_edge("describe_graph", "responder") # Assuming describe_graph always goes to responder


    # The plan_execution node is available via the router but not part of the default Parse->... flow.
    # If plan_execution runs, let's route it to the responder to show the plan description.
    # Assuming plan_execution returns updates['__next__'] = 'responder'
    builder.add_edge("plan_execution", "responder") # Assuming plan_execution always goes to responder


    # Other specific action nodes route directly to the responder after completing their task
    # Assuming these nodes return updates['__next__'] = 'responder'
    builder.add_edge("get_graph_json", "responder") # Assuming get_graph_json always goes to responder
    builder.add_edge("answer_openapi_query", "responder") # Assuming answer_openapi_query always goes to responder
    builder.add_edge("handle_unknown", "responder") # Assuming handle_unknown always goes to responder
    builder.add_edge("handle_loop", "responder") # Assuming handle_loop always goes to responder


    # 3. From the responder node, the graph ends.
    # The responder node doesn't need a conditional edge because it's the end.
    # It should not return a '__next__' key.
    builder.add_edge("responder", END)


    # Compile the graph with memory
    # checkpointer is passed in the startup event in api.py
    app = builder.compile() # Compile without checkpointer here, it's added when running the app
    logger.info("Graph compiled (without API execution capabilities).")
    return app

# Note: The core_logic methods (parse_openapi_spec, identify_apis, etc.) will also need
# to be updated to return a dictionary of state updates including the '__next__' key
# to correctly drive the conditional edges added above. This was not explicitly
# requested in the prompt but is necessary for this graph structure to work.
# I will provide the updated core_logic.py as well.
