**filename: graph.py**

```python
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
        logger.debug(f"Set final_response from state.response: '{state.response[:100]}...'\nFull Response: {state.response}") # Log full response
        # Clear the intermediate response after using it
        updates['response'] = None # <-- Clear intermediate response
    elif not state.final_response: # Only use fallback if final_response is not already set
        # Fallback if intermediate response is also missing
        updates['final_response'] = "I'm sorry, I don't have a specific response for that right now."
        logger.warning("Responder executed, but neither state.response nor state.final_response was set.")

    # LangGraph automatically handles clearing the state after the END node is reached.
    # No need to explicitly clear state variables here unless you want to reset mid-graph.

    # You might want to add the scratchpad to the final response for debugging
    # updates['final_response'] += f"\n\n--- Debug Scratchpad ---\n{json.dumps(state.scratchpad, indent=2)}"


    return updates # Return the updates dictionary
def build_graph(router_llm: Any, worker_llm: Any) -> StateGraph:
    logger.info("Building LangGraph...")

    # Instantiate core logic and router
    core_logic = OpenAPICoreLogic(worker_llm=worker_llm)
    router = OpenAPIRouter(router_llm=router_llm)

    # Define the graph
    builder = StateGraph(BotState)

    # Send all new inputs into our router node
    builder.add_edge(START, "router")

    # Add router node and map its next-intent to downstream nodes
    builder.add_node("router", router.route)
    # Unconditional edge for debugging: ensure parse_openapi_spec is reachable
    builder.add_edge("router", "parse_openapi_spec")
    builder.add_conditional_edges(
        "router",
        # use the __next__ update from router.route
        lambda state: state.get("__next__", "responder"),
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

    # Define tool nodes
    builder.add_node("parse_openapi_spec", core_logic.parse_openapi_spec)
    builder.add_node("identify_apis", core_logic.identify_apis)
    builder.add_node("generate_payloads", core_logic.generate_payloads)
    builder.add_node("generate_execution_graph", core_logic.generate_execution_graph)
    builder.add_node("describe_graph", core_logic.describe_graph)
    builder.add_node("answer_openapi_query", core_logic.answer_openapi_query)
    builder.add_node("handle_unknown", core_logic.handle_unknown)
    builder.add_node("handle_loop", core_logic.handle_loop)
    builder.add_node("responder", finalize_response)

    # Entry point
    builder.set_entry_point("router")

    # Conditional edges for parse_openapi_spec
    builder.add_conditional_edges(
        "parse_openapi_spec",
        lambda state: state.get("__next__", "responder"),
        {"identify_apis": "identify_apis", "responder": "responder"}
    )

    # End graph at responder
    builder.add_edge("responder", END)

    app = builder.compile()
    logger.info("Graph compiled (with debug edges).")
    return app
# Note: The core_logic methods (parse_openapi_spec, identify_apis, etc.) and
# router.route are designed to return a dictionary of state updates,
# including the critical '__next__' key for routing.
# The finalize_response (responder node) prepares the final user-facing message.
