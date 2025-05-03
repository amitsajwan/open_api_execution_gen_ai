import logging
from typing import Any, Dict
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from models import BotState
from core_logic import OpenAPICoreLogic
from router import OpenAPIRouter, AVAILABLE_INTENTS

logger = logging.getLogger(__name__)

def build_graph(router_llm: Any, worker_llm: Any) -> Any:
    logger.info("Building LangGraph...")

    core_logic = OpenAPICoreLogic(worker_llm)
    router = OpenAPIRouter(router_llm)

    # Use StateGraph (Pydantic model BotState)  
    builder = StateGraph(BotState)

    # 1) Router node (BotState → str)
    builder.add_node("router", router.route)
    logger.debug("Added node: router")

    # 2) Core‑logic adapter: BotState→BotState  →  dict→dict
    def wrap_method(fn):
        def node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            # Rehydrate Pydantic model
            state = BotState.model_validate(state_dict)
            # Execute your logic (returns BotState)
            new_state = fn(state)
            # Dump to plain dict for LangGraph
            return new_state.model_dump()
        return node

    # 3) Register all tool methods through the wrapper
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

    for intent_name, method in tool_methods.items():
        if intent_name in AVAILABLE_INTENTS:
            builder.add_node(intent_name, wrap_method(method))
            logger.debug(f"Added wrapped node: {intent_name}")
        else:
            logger.warning(f"Skipping unknown intent: {intent_name}")

    # 4) Entry point & edges
    builder.set_entry_point("router")
    builder.add_conditional_edges(
        "router",
        lambda state: state.intent,
        {intent: intent for intent in AVAILABLE_INTENTS}
    )
    for intent_name in tool_methods:
        if intent_name in AVAILABLE_INTENTS:
            builder.add_edge(intent_name, "router")

    # 5) Compile with memory checkpointing
    app = builder.compile(checkpointer=MemorySaver())
    logger.info("LangGraph compiled successfully.")
    return app
