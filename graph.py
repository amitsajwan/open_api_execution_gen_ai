# filename: graph.py
import logging
from typing import Any, Dict, List, Optional, Tuple
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import state model and core logic
from models import BotState
from core_logic import OpenAPICoreLogic
from router import OpenAPIRouter

logger = logging.getLogger(__name__)

# --- Responder Node Function ---
def finalize_response(state: BotState) -> BotState:
    """
    Sets the final_response based on the last intermediate response from a node.
    Clears the intermediate response field.
    """
    tool_name = "responder"
    state.update_scratchpad_reason(tool_name, "Finalizing response for user.")
    logger.debug("Executing responder node (finalize_response).")

    if state.response:
        state.final_response = state.response
        logger.debug(f"Set final_response from state.response: '{state.response[:100]}...'")
    elif not state.final_response: # Avoid overwriting if already set by error handling etc.
        # Fallback if intermediate response is also missing
        state.final_response = "I've completed the requested action, but there wasn't a specific message to display."
        logger.warning("Responder: No intermediate 'response' found to set as 'final_response'. Using default.")
        state.update_scratchpad_reason(tool_name, "Warning: No intermediate response found. Used default final response.")

    # Clear the intermediate response field for the next turn
    state.response = None
    return state

# --- Graph Building ---
def build_graph(router_llm: Any, worker_llm: Any) -> Any:
    """
    Builds the LangGraph for the OpenAPI assistant without API execution.
    The graph focuses on parsing, planning (describing), and answering queries.
    """
    # Initialize the core logic and router instances
    core_logic = OpenAPICoreLogic(worker_llm=worker_llm)
    router_instance = OpenAPIRouter(router_llm=router_llm)

    # Define the nodes using the methods from OpenAPICoreLogic and the router
    nodes = {
        "router": router_instance.route, # Initial routing node
        "parse_openapi_spec": core_logic.parse_openapi_spec,
        "plan_execution": core_logic.plan_execution, # Plans (describes) execution
        "identify_apis": core_logic.identify_apis,
        "generate_payloads": core_logic.generate_payloads, # Generates payload descriptions
        "generate_execution_graph": core_logic.generate_execution_graph, # Generates graph description
        "describe_graph": core_logic.describe_graph, # Describes the graph description
        "get_graph_json": core_logic.get_graph_json, # Outputs graph description JSON
        "answer_openapi_query": core_logic.answer_openapi_query, # Answers general questions
        "handle_unknown": core_logic.handle_unknown, # Handles unclear intent
        "handle_loop": core_logic.handle_loop, # Handles detected loops
        "responder": finalize_response # Use the dedicated responder function
    }

    # Define the graph
    builder = StateGraph(BotState)

    # Add nodes to the graph
    for node_name, node_function in nodes.items():
        builder.add_node(node_name, node_function)

    # --- Define the entry point ---
    builder.set_entry_point("router")

    # --- Define the graph flow (edges) ---

    # 1. From the router, route to different nodes based on intent
    # The router node sets state.intent, which is used by the conditional edge.
    builder.add_conditional_edges(
        "router", # Source node
        lambda state: state.intent, # Function to determine the next node based on state.intent
        { # Map intent strings to target node names
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
            # If router returns an intent not in this map, LangGraph will raise an error.
            # The router logic should ensure it always returns a valid intent or 'handle_unknown'.
        }
    )

    # 2. Define transitions between core logic nodes (Happy Path Workflow)
    # This defines a common sequence, but the router can jump to later steps if needed.

    # After parsing, maybe identify APIs or plan execution next.
    # Let's make identifying APIs the default follow-up to parsing.
    # We also need to handle the case where parsing fails.
    def check_parse_success(state: BotState) -> str:
        """Routes after parsing based on success."""
        if state.openapi_schema:
            logger.debug("Routing from parse_openapi_spec: Success -> identify_apis")
            return "identify_apis" # Proceed to identify APIs if schema parsed
        else:
            logger.debug("Routing from parse_openapi_spec: Failure -> responder")
            return "responder" # Go to responder to output the error message

    builder.add_conditional_edges("parse_openapi_spec", check_parse_success)

    # After identifying APIs, generate payload descriptions
    # Consider adding a check if APIs were actually identified
    builder.add_edge("identify_apis", "generate_payloads")

    # After generating payload descriptions, generate the graph description
    builder.add_edge("generate_payloads", "generate_execution_graph")

    # After generating the graph description, describe it
    # Consider adding a check if graph was generated
    builder.add_edge("generate_execution_graph", "describe_graph")

    # After describing the graph, route to the responder
    builder.add_edge("describe_graph", "responder")

    # The plan_execution node is available via the router but not part of the default Parse->... flow.
    # If plan_execution runs, let's route it to the responder to show the plan.
    builder.add_edge("plan_execution", "responder")

    # Other specific action nodes route directly to the responder after completing their task
    builder.add_edge("get_graph_json", "responder")
    builder.add_edge("answer_openapi_query", "responder")
    builder.add_edge("handle_unknown", "responder")
    builder.add_edge("handle_loop", "responder")

    # 3. From the responder node, the graph ends.
    builder.add_edge("responder", END)

    # Compile the graph with memory
    app = builder.compile(checkpointer=MemorySaver())
    logger.info("Graph compiled (without API execution capabilities).")
    return app
