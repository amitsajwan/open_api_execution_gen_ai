import logging
from typing import Any, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Ensure INFO-level logs are visible
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import state model and core logic
from models import BotState
from core_logic import OpenAPICoreLogic
from router import OpenAPIRouter

# --- Responder Node Function ---
def finalize_response(state: BotState) -> Dict[str, Any]:
    tool_name = "responder"
    state.update_scratchpad_reason(tool_name, "Finalizing response for user.")
    logger.debug("Executing responder node (finalize_response).")
    updates: Dict[str, Any] = {}
    if state.response:
        updates['final_response'] = state.response
    elif not state.final_response:
        updates['final_response'] = "I've completed the requested action, but there wasn't a specific message to display."
        logger.warning("Responder: No intermediate 'response' found. Using default final response.")
        state.update_scratchpad_reason(tool_name, "Warning: default final response used.")
    updates['response'] = None
    state.update_scratchpad_reason(tool_name, "Final response set.")
    return updates

# --- Graph Definition ---
def build_graph(router_llm: Any, worker_llm: Any) -> StateGraph:
    logger.info("Building LangGraph with debug-enabled routing...")

    # Instantiate core logic and router
    core_logic = OpenAPICoreLogic(worker_llm=worker_llm)
    router = OpenAPIRouter(router_llm=router_llm)

    # Create graph builder
    builder = StateGraph(BotState)

    # Send all new inputs into router
    builder.add_edge(START, "router")

    # Add router node
    builder.add_node("router", router.route)

    # Named function to extract and log the __next__ value and full state
    def debug_next(state):
        # Dump full state for inspection
        full = state.model_dump()
        nxt = full.get("__next__", "responder")
        logger.info(f"[DebugRouting] __next__ = {nxt}, full state dump: {full}")
        return nxt

    # Conditional edges: use debug_next to both log and return
    builder.add_conditional_edges(
        "router",
        debug_next,
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

    # Tool nodes
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

    # Continue routing from parse_openapi_spec
    builder.add_conditional_edges(
        "parse_openapi_spec",
        debug_next,
        {"identify_apis": "identify_apis", "responder": "responder"}
    )

    # End at responder
    builder.add_edge("responder", END)

    app = builder.compile()
    logger.info("Graph compiled with enhanced debug routing.")
    return app
