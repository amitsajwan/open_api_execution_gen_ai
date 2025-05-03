import logging
from typing import Any, Dict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # Example checkpointer

# Assuming models, core_logic, router are defined in respective files
from models import BotState
from core_logic import OpenAPICoreLogic
from router import OpenAPIRouter, AVAILABLE_INTENTS

# Module-level logger
logger = logging.getLogger(__name__)



def build_graph(router_llm: Any, worker_llm: Any) -> Any:
    """
    Builds the LangGraph StateGraph for the OpenAPI assistant.

    Args:
        router_llm: The LLM instance for routing and parameter extraction.
        worker_llm: The LLM instance for core OpenAPI tasks.

    Returns:
        A compiled LangGraph application ready for execution.
    """
    logger.info("Building LangGraph...")

    # Initialize components
    core_logic = OpenAPICoreLogic(worker_llm)
    router = OpenAPIRouter(router_llm)

    # Define the graph structure using BotState
    builder = StateGraph(BotState)

    # --- Add Nodes ---
    # Router Node: Determines the next step (returns str)
    builder.add_node("router", router.route)
    logger.debug("Added node: router")

    # Core Logic Nodes: One for each tool/action
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

    # Adapter: wrap BotState->BotState to dict->dict
    def wrap_method(fn):
        def node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            # Reconstruct Pydantic BotState
            state = BotState.model_validate(state_dict)
            # Execute logic (returns BotState)
            new_state = fn(state)
            # Return plain dict for LangGraph
            return new_state.model_dump()
        return node

    # Add wrapped core logic nodes
    for intent_name, method in tool_methods.items():
        if intent_name in AVAILABLE_INTENTS:
            builder.add_node(intent_name, wrap_method(method))
            logger.debug(f"Added wrapped node: {intent_name}")
        else:
            logger.warning(f"Skipping unknown intent: {intent_name}")

    # --- Define Edges ---
    # Entry Point
    builder.set_entry_point("router")

    # Conditional edges from router
    builder.add_conditional_edges(
        "router",
        lambda state: state.intent,
        {intent: intent for intent in AVAILABLE_INTENTS}
    )

    # Loop back edges from tools
    for intent_name in tool_methods:
        if intent_name in AVAILABLE_INTENTS:
            builder.add_edge(intent_name, "router")

    # --- Compile with MemorySaver ---
    memory = MemorySaver()
    app = builder.compile(checkpointer=memory)
    logger.info("LangGraph compiled successfully.")
    return app