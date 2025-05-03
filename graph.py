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
    
    # Router Node: Determines the next step
    builder.add_node("router", router.route) 
    logger.debug("Added node: router")

    # Core Logic Nodes: One for each tool/action
    # Map intent strings to the corresponding methods in OpenAPICoreLogic
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
        if intent_name in AVAILABLE_INTENTS: # Ensure we only add defined intents
            builder.add_node(intent_name, method)
            logger.debug(f"Added node: {intent_name}")
        else:
             logger.warning(f"Method {intent_name} not in AVAILABLE_INTENTS, skipping node addition.")

    # --- Define Edges ---

    # Entry Point: Start at the router
    builder.set_entry_point("router")
    logger.debug("Set entry point: router")

    # Conditional Edges from Router: Based on the intent returned by the router node
    # The router's return value (the intent string) directly maps to the next node name.
    builder.add_conditional_edges(
        "router",  # Source node
        lambda state: state.intent, # Function to determine the next node (returns intent string)
        {intent: intent for intent in AVAILABLE_INTENTS} # Map intent strings to node names
    )
    logger.debug(f"Added conditional edges from 'router' to: {list(AVAILABLE_INTENTS)}")

    # Edges from Core Logic Nodes: After a tool runs, go back to the router 
    # to process the next user input or decide the next step based on the updated state.
    # Alternatively, some tools could lead to END if they signify completion.
    for intent_name in tool_methods.keys():
         if intent_name in AVAILABLE_INTENTS:
              # For now, all tools loop back to the router
              builder.add_edge(intent_name, "router") 
              logger.debug(f"Added edge: {intent_name} -> router")
              # Example of ending the graph after a specific action:
              # if intent_name == "get_graph_json":
              #     builder.add_edge(intent_name, END)
              # else:
              #     builder.add_edge(intent_name, "router")


    # --- Compile the Graph ---
    
    # Using MemorySaver for in-memory checkpointing (state persistence between steps)
    # Replace with a persistent checkpointer (e.g., SqliteSaver, RedisSaver) for longer sessions.
    memory = MemorySaver() 
    
    try:
        app = builder.compile(checkpointer=memory)
        logger.info("LangGraph compiled successfully.")
        return app
    except Exception as e:
        logger.error(f"Error compiling LangGraph: {e}", exc_info=True)
        raise

