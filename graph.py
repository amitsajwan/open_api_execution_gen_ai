import logging
from typing import Any, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from models import BotState
from core_logic import OpenAPICoreLogic
from router import OpenAPIRouter, AVAILABLE_INTENTS

logger = logging.getLogger(__name__)


def build_graph(router_llm: Any, worker_llm: Any) -> Any:
    core_logic = OpenAPICoreLogic(worker_llm)
    router = OpenAPIRouter(router_llm)

    builder = StateGraph(BotState)

    # Adapter for core logic methods
    def wrap_method(fn):
        def node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            state = BotState.model_validate(state_dict)
            result = fn(state)
            return result if isinstance(result, dict) else result.model_dump()
        return node

    # Register tool nodes (including handle_loop)
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
        builder.add_node(name, wrap_method(fn))

    # Entry routing from START via router
    builder.add_conditional_edges(
        START,
        router.route,
        {i: i for i in AVAILABLE_INTENTS}
    )

    # After each tool: loop or end
    TERMINAL = {"describe_graph","get_graph_json","handle_unknown","handle_loop"}
    for intent in AVAILABLE_INTENTS:
        if intent in TERMINAL:
            builder.add_edge(intent, END)
        else:
            builder.add_conditional_edges(
                intent,
                router.route,
                {i: i for i in AVAILABLE_INTENTS}
            )

    app = builder.compile(checkpointer=MemorySaver())
    logger.info("Graph compiled with cycle detection")
    return app
