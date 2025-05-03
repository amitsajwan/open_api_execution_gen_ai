import logging
from typing import Any, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from models import BotState
from core_logic import OpenAPICoreLogic
from router import OpenAPIRouter, AVAILABLE_INTENTS

logger = logging.getLogger(__name__)

def build_graph(router_llm: Any, worker_llm: Any) -> Any:
    """
    Builds a LangGraph StateGraph with conditional routing directly from START and after each tool node.
    """
    core_logic = OpenAPICoreLogic(worker_llm)
    router = OpenAPIRouter(router_llm)

    # Initialize StateGraph with Pydantic state schema
    builder = StateGraph(BotState)

    # Wrap core logic methods (BotState -> BotState or dict) into nodes (dict -> dict)
    def wrap_method(fn):
        def node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            # Rehydrate Pydantic model from raw dict
            state = BotState.model_validate(state_dict)
            # Execute core logic (may return BotState or dict)
            result = fn(state)
            # If result is already a dict, return it directly
            if isinstance(result, dict):
                return result
            # Otherwise assume it's a BotState and serialize
            return result.model_dump()
        return node

    # Register core logic as nodes
    tool_methods = {
        "parse_openapi_spec": core_logic.parse_openapi_spec,
        "identify_apis": core_logic.identify_apis,
        "generate_payloads": core_logic.generate_payloads,
        "generate_execution_graph": core_logic.generate_execution_graph,
        "add_graph_edge": core_logic.add_graph_edge,
        "validate_graph": core_logic.validate_graph,
        "describe_graph": core_logic.describe_graph,
        "get_graph_json": core_logic.get_graph_json,
        "handle_unknown": core_logic.handle_unknown,
    }
    for name, fn in tool_methods.items():
        if name in AVAILABLE_INTENTS:
            builder.add_node(name, wrap_method(fn))
            logger.debug(f"Added node: {name}")
        else:
            logger.warning(f"Skipping unknown intent: {name}")

    # Conditional entry: route from START to first tool based on intent
    builder.add_conditional_edges(
        START,
        router.route,
        {intent: intent for intent in AVAILABLE_INTENTS}
    )

    # After each tool node, route again
    for intent in AVAILABLE_INTENTS:
        if intent == "handle_unknown":
            # End the graph on unknown intent
            builder.add_edge("handle_unknown", END)
        else:
            # Route back to router for further processing
            builder.add_conditional_edges(
                intent,
                router.route,
                {i: i for i in AVAILABLE_INTENTS}
            )

    # Compile graph with in-memory checkpointing
    app = builder.compile(checkpointer=MemorySaver())
    logger.info("LangGraph compiled successfully.")
    return app
