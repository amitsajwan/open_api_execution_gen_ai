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
    logger.info("Building LangGraph...")

    # Instantiate core logic and router
    core_logic = OpenAPICoreLogic(worker_llm=worker_llm)
    router = OpenAPIRouter(router_llm=router_llm)

    # Define the graph
    builder = StateGraph(BotState)

    # ── NEW: send all new inputs into our router node ────────────────────────
    builder.add_edge(START, "router")

    # Add nodes for each distinct step/tool
    builder.add_node("router", router.route)
    # ── NEW: map router’s __next__ into the proper downstream node ────────────
    builder.add_conditional_edges(
        "router",
        lambda state: state.model_dump().get("__next__", "responder"),
        {
            "parse_openapi_spec":       "parse_openapi_spec",
            "identify_apis":            "identify_apis",
            "generate_payloads":        "generate_payloads",
            "generate_execution_graph": "generate_execution_graph",
            "describe_graph":           "describe_graph",
            "answer_openapi_query":     "answer_openapi_query",
            "handle_unknown":           "handle_unknown",
            "handle_loop":              "handle_loop",
            "responder":                "responder",
        }
    )

    builder.add_node("parse_openapi_spec", core_logic.parse_openapi_spec)
    builder.add_node("identify_apis", core_logic.identify_apis)
    builder.add_node("generate_payloads", core_logic.generate_payloads)
    builder.add_node("generate_execution_graph", core_logic.generate_execution_graph)
    builder.add_node("describe_graph", core_logic.describe_graph)
    builder.add_node("answer_openapi_query", core_logic.answer_openapi_query)
    builder.add_node("handle_unknown", core_logic.handle_unknown)
    builder.add_node("handle_loop", core_logic.handle_loop)
    builder.add_node("responder", finalize_response)

    # We still mark router as entry point (optional, since START→router edge does it)
    builder.set_entry_point("router")

    # Conditional edges for other nodes (unchanged)…
    builder.add_conditional_edges(
        "parse_openapi_spec",
        lambda state: state.model_dump().get("__next__", "responder"),
        {
            "identify_apis": "identify_apis",
            "responder":     "responder",
        }
    )
    # … identify_apis, generate_payloads, generate_execution_graph, describe_graph, etc.

    # From the responder node, the graph ends.
    builder.add_edge("responder", END)

    # Compile the graph
    app = builder.compile()
    logger.info("Graph compiled (without API execution capabilities).")
    return app
# Note: The core_logic methods (parse_openapi_spec, identify_apis, etc.) and
# the router.route method must return a dictionary of state updates including
# the '__next__' key to correctly drive these conditional edges.
