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
    """
    Sets the final_response based on state.response and clears intermediate response.
    """
    tool_name = "responder"
    state.update_scratchpad_reason(tool_name, "Finalizing response for user.")
    logger.debug("Executing responder node (finalize_response).")

    updates: Dict[str, Any] = {}
    if state.response:
        updates['final_response'] = state.response
    elif not state.final_response:
        updates['final_response'] = (
            "I've completed the requested action, but there wasn't a specific message to display."
        )
        logger.warning("Responder: No intermediate 'response' found; using default final_response.")

    updates['response'] = None
    state.update_scratchpad_reason(tool_name, "Final response set.")
    return updates

# --- Graph Definition ---
def build_graph(router_llm: Any, worker_llm: Any) -> StateGraph:
    """
    Builds and compiles the LangGraph StateGraph for the OpenAPI agent.
    """
    logger.info("Building LangGraph graph.py...")

    core_logic = OpenAPICoreLogic(worker_llm=worker_llm)
    router = OpenAPIRouter(router_llm=router_llm)

    builder = StateGraph(BotState)

    # Wrap router.route to log its __next__ output
    orig_route = router.route
    def wrapped_route(state: BotState) -> Dict[str, Any]:
        updates = orig_route(state)
        logger.info(f"Router produced __next__ = {updates.get('__next__')}")
        return updates

    # Add router node using wrapper
    builder.add_node("router", wrapped_route)

    # Add tool nodes
    builder.add_node("parse_openapi_spec", core_logic.parse_openapi_spec)
    builder.add_node("identify_apis", core_logic.identify_apis)
    builder.add_node("generate_payloads", core_logic.generate_payloads)
    builder.add_node("generate_execution_graph", core_logic.generate_execution_graph)
    builder.add_node("describe_graph", core_logic.describe_graph)
    builder.add_node("get_graph_json", core_logic.get_graph_json)
    builder.add_node("answer_openapi_query", core_logic.answer_openapi_query)
    builder.add_node("handle_unknown", core_logic.handle_unknown)
    builder.add_node("handle_loop", core_logic.handle_loop)
    builder.add_node("plan_execution", core_logic.plan_execution)
    builder.add_node("responder", finalize_response)

    # Entry point
    builder.add_edge(START, "router")
    builder.set_entry_point("router")

    # Conditional routing from router
    def debug_next(state: BotState) -> str:
        nxt = state.model_dump().get("__next__", "responder")
        logger.info(f"[DebugRouting] Next node: {nxt}")
        return nxt
    builder.add_conditional_edges(
        "router",
        debug_next,
        {
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
            "responder": "responder",
        }
    )

    # Default conditional flows for each tool node
    for node in [
        "parse_openapi_spec", "identify_apis", "generate_payloads",
        "generate_execution_graph", "describe_graph", "plan_execution",
        "get_graph_json", "answer_openapi_query", "handle_unknown", "handle_loop"
    ]:
        builder.add_conditional_edges(
            node,
            lambda state: state.model_dump().get("__next__", "responder"),
            {"responder": "responder", **{node: node}}
        )

    # End
    builder.add_edge("responder", END)

    app = builder.compile()
    logger.info("Graph compiled successfully.")
    return app
