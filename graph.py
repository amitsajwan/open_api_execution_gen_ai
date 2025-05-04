# filename: graph.py
import logging
# Removed requests and jsonpath_ng imports as execution is removed
from typing import Any, Dict, List, Optional, Tuple
from langgraph.graph import StateGraph, START, END
# Removed jsonpath_parse import
# Removed ValidationError import

from models import BotState # We still need BotState
from core_logic import OpenAPICoreLogic # We need the updated core_logic methods
from router import OpenAPIRouter # We still need the router

# Assuming MemorySaver is defined appropriately (e.g., in utils or directly)
# from utils import MemorySaver
# Using LangGraph's built-in memory saver for simplicity here
from langgraph.checkpoint.memory import MemorySaver

# Removed llm_call_helper and parse_llm_json_output as they are used inside core_logic
# Removed extract_data_with_jsonpath

logger = logging.getLogger(__name__)

# Removed JSONPath Extraction Utility function

# Removed API Execution Utility function

# Removed wrap_core_logic_method as it was for executor result handling

def build_graph(router_llm: Any, worker_llm: Any) -> Any:
    """
    Builds the LangGraph for the OpenAPI assistant without API execution.
    The graph focuses on parsing, planning (describing), and answering queries.
    """
    # Initialize the core logic and router instances
    core_logic = OpenAPICoreLogic(worker_llm=worker_llm)
    router_instance = OpenAPIRouter(router_llm=router_llm)

    # Define the state keys that each node can update (excluding 'results' now)
    # Note: LangGraph StateGraph automatically handles state updates,
    # this definition is more for clarity/documentation in simpler graphs.
    # In complex graphs with multiple writers, you might need StateGraph(channels=...)
    # For this simplified graph, we'll rely on the node functions updating the BotState instance.
    # The core_logic methods return a BotState instance, effectively being 'setters'.

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
        # Removed "executor" node
        "responder": lambda state: state # The responder node simply finalizes the state
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
    # The router returns the string name of the next node
    def router_conditional_edge(state: BotState) -> str:
        """
        Conditional edge based on the router's output (the determined intent).
        """
        # The router node directly returns the next node name
        # Check the state for the intent set by the router
        # Note: The router node function should set state.intent *before* returning
        # For simplicity, let's assume the router directly returns the node name string.
        # If router.route returns a string, LangGraph automatically uses it for transition.
        # If router.route returns a state object, you'd need to check state.intent here.
        # Assuming router.route returns the node name string directly as in original.
        intent = state.intent # Assuming router node sets state.intent before exiting

        if intent in nodes:
             return intent
        else:
             # This case should ideally be handled by router returning 'handle_unknown'
             logger.warning(f"Router returned unknown intent '{intent}'. Defaulting to handle_unknown.")
             return "handle_unknown"

    # The router node returns the intent string, which serves as the next node name
    # LangGraph automatically transitions based on the string returned by a node
    # if it matches a node name or a defined edge.
    # We can use add_edge for explicit transitions from router to known intents.
    # Or, if the router *only* returns valid node names, we don't strictly need
    # a conditional edge here, just edges for each possible output string.
    # Let's use add_edge for clarity for the primary routes.
    # If router can return *any* string, a conditional edge is safer.

    # Let's assume the router node sets state.intent and we use a conditional edge based on that.
    # The router node needs to return the state instance after setting state.intent.
    # We updated the router to return BotState in a previous diff, let's stick to that pattern.

    builder.add_conditional_edges(
        "router", # From the router node
        lambda state: state.intent, # Conditional function: route based on state.intent
        {
            "parse_openapi_spec": "parse_openapi_spec",
            "plan_execution": "plan_execution",
            "identify_apis": "identify_apis", # Can still be routed to directly
            "generate_payloads": "generate_payloads", # Can still be routed to directly
            "generate_execution_graph": "generate_execution_graph", # Can still be routed to directly
            "describe_graph": "describe_graph",
            "get_graph_json": "get_graph_json",
            "answer_openapi_query": "answer_openapi_query",
            "handle_unknown": "handle_unknown",
            "handle_loop": "handle_loop",
            # Removed "executor" and intents related to execution
            # Add a route for when the router doesn't recognize the intent
            # This should be handled by router returning 'handle_unknown', but as a fallback:
            # LangGraph's default behavior might handle this, but explicit can be clearer.
            # If router returns a string not in this map, it will raise an error.
            # It's safer if the router guarantees returning one of the valid node names or 'handle_unknown'.
        }
    )

    # 2. Define transitions between core logic nodes

    # After parsing, we might want to proactively plan or identify APIs
    # Let's route parse_openapi_spec to plan_execution as a common follow-up
    builder.add_edge("parse_openapi_spec", "plan_execution")

    # After planning, the state.execution_plan is set.
    # We could route to describe the plan, or identify APIs, or generate the graph.
    # Let's route plan_execution to identify_apis as a common next step in the workflow
    builder.add_edge("plan_execution", "identify_apis")

    # After identifying APIs, generate payloads
    builder.add_edge("identify_apis", "generate_payloads")

    # After generating payloads, generate the execution graph description
    builder.add_edge("generate_payloads", "generate_execution_graph")

    # After generating the graph description, maybe describe it or just finish the turn.
    # Let's route generate_execution_graph to describe_graph
    builder.add_edge("generate_execution_graph", "describe_graph")

    # After describing the graph, route to the responder to output the description/summary
    builder.add_edge("describe_graph", "responder")

    # Other nodes typically route directly to the responder after completing their task
    builder.add_edge("get_graph_json", "responder")
    builder.add_edge("answer_openapi_query", "responder")
    builder.add_edge("handle_unknown", "responder")
    builder.add_edge("handle_loop", "responder")

    # Removed executor node and its conditional edges

    # 3. From the responder node, the graph ends.
    builder.add_edge("responder", END)

    # Compile the graph with memory
    # Ensure MemorySaver is correctly imported or defined
    # Assuming MemorySaver is available (e.g., imported from langgraph.checkpoint.memory)
    app = builder.compile(checkpointer=MemorySaver())
    logger.info("Graph compiled (without API execution capabilities).")
    return app

# Removed executor_router function
# Removed api_execution_node function
# Removed wrap_core_logic_method
