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

# --- Graph Building Function ---
def build_graph(router_llm: Any, worker_llm: Any, checkpointer: Any):
    """
    Builds and compiles the LangGraph StateGraph.

    Args:
        router_llm: The LLM for the router.
        worker_llm: The LLM for core logic tasks.
        checkpointer: The checkpointer instance.
    """
    logger.info("Building the LangGraph.")

    # Define the state for the graph
    # The state is defined in models.py as BotState

    # Initialize the core logic components
    router = OpenAPIRouter(router_llm)
    core_logic = OpenAPICoreLogic(worker_llm)
    # Note: APIExecutor is NOT initialized as this agent does not execute APIs

    # Create a StateGraph with the BotState
    builder = StateGraph(BotState)

    # Add nodes to the graph
    # Each node corresponds to a step in the process (a function or method)
    builder.add_node("router", router.route) # The first step
    builder.add_node("parse_openapi_spec", core_logic.parse_openapi_spec)
    builder.add_node("identify_apis", core_logic.identify_apis)
    builder.add_node("generate_payloads", core_logic.generate_payloads)
    builder.add_node("generate_execution_graph", core_logic.generate_execution_graph)
    builder.add_node("describe_graph", core_logic.describe_graph)
    # builder.add_node("get_graph_json", core_logic.get_graph_json) # Add if you implement this
    builder.add_node("answer_openapi_query", core_logic.answer_openapi_query)
    builder.add_node("handle_unknown", core_logic.handle_unknown)
    builder.add_node("handle_loop", core_logic.handle_loop)
    builder.add_node("responder", finalize_response) # The final response node


    # Set the entry point of the graph
    # When the graph is invoked, it starts here.
    builder.set_entry_point("router")

    # Add edges between nodes
    # Edges define the flow of execution.
    # Conditional edges use a function to decide the next step based on the state.

    # From router, transition based on the '__next__' key set by the router
    builder.add_conditional_edges(
        "router",
        # The condition function reads the '__next__' key from the state
        # If __next__ is not set, default to 'responder' (or 'handle_unknown')
        lambda state: state.model_dump().get("__next__", "handle_unknown"), # Default to handle_unknown if router fails to set __next__
        {
            "parse_openapi_spec":       "parse_openapi_spec",
            "identify_apis":            "identify_apis",
            "generate_payloads":        "generate_payloads",
            "generate_execution_graph": "generate_execution_graph",
            "describe_graph":           "describe_graph",
            # "get_graph_json":           "get_graph_json", # Add if you implement this
            "answer_openapi_query":     "answer_openapi_query",
            "handle_unknown":           "handle_unknown",
            "handle_loop":              "handle_loop",
            "responder":                "responder", # Allow router to directly route to responder
        }
    )

    # From parse_openapi_spec, transition based on the '__next__' key set by parse_openapi_spec
    builder.add_conditional_edges(
        "parse_openapi_spec",
        lambda state: state.model_dump().get("__next__", "responder"), # Default to responder on failure
        {
            "identify_apis": "identify_apis", # Success route
            "responder":     "responder",     # Error/failure route
        }
    )

    # From identify_apis, transition based on the '__next__' key
    builder.add_conditional_edges(
        "identify_apis",
        lambda state: state.model_dump().get("__next__", "responder"), # Default to responder
        {
            "generate_payloads": "generate_payloads",
            "answer_openapi_query": "answer_openapi_query", # Allow routing to answer questions after identifying APIs
            "responder":         "responder",
        }
    )

    # From generate_payloads, transition based on the '__next__' key
    builder.add_conditional_edges(
        "generate_payloads",
        lambda state: state.model_dump().get("__next__", "responder"), # Default to responder
        {
            "generate_execution_graph": "generate_execution_graph",
            "answer_openapi_query": "answer_openapi_query", # Allow routing to answer questions after generating payloads
            "responder":              "responder",
        }
    )

    # From generate_execution_graph, transition based on the '__next__' key
    builder.add_conditional_edges(
        "generate_execution_graph",
        lambda state: state.model_dump().get("__next__", "responder"), # Default to responder
        {
            "describe_graph": "describe_graph",
            "answer_openapi_query": "answer_openapi_query", # Allow routing to answer questions after generating graph
             # Potentially route to a node that prepares for execution, but NOT execution itself in this agent
            "responder":      "responder",
        }
    )

    # From describe_graph, transition based on the '__next__' key
    builder.add_conditional_edges(
        "describe_graph",
        lambda state: state.model_dump().get("__next__", "responder"), # Default to responder
        {
            "answer_openapi_query": "answer_openapi_query", # Often the next step is answering questions about the graph
            "responder":          "responder",
        }
    )

    # From answer_openapi_query, transition based on the '__next__' key
    builder.add_conditional_edges(
        "answer_openapi_query",
        lambda state: state.model_dump().get("__next__", "responder"), # Default to responder
        {
            # After answering a question, where should it go?
            # It could go back to the router to process a follow-up query,
            # or directly to the responder if the conversation turn ends.
            # Let's route to responder for now, assuming each turn ends after an answer.
            # If you want multi-turn interaction after an answer, route back to 'router'.
            "router":    "router", # <-- Route back to router for follow-up
            "responder": "responder",
        }
    )

    # From handle_unknown, always transition to responder
    builder.add_edge("handle_unknown", "responder")

    # From handle_loop, always transition to responder
    builder.add_edge("handle_loop", "responder")

    # From the responder node, the graph ends.
    builder.add_edge("responder", END)


    # Compile the graph
    # The compiled graph is the runnable object.
    app = builder.compile()#.with_types(input_type=Dict[str, Any], output_type=BotState) # Optional: add type hints

    # Add checkpointer to the compiled graph
    # This enables state persistence across invocations for the same thread_id
    app = app.with_state(BotState).add_graph(builder.compile()).with_checkpointer(checkpointer)


    logger.info("Graph compiled (without API execution capabilities).")
    return app

# Note: The core_logic methods (parse_openapi_spec, identify_apis, etc.) and
# router.route are designed to return a dictionary of state updates,
# including the critical '__next__' key for routing.
# The finalize_response (responder node) prepares the final user-facing message.
