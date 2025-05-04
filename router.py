# filename: router.py
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError

# Assuming models.py defines BotState
from models import BotState

# Assuming utils.py has llm_call_helper and parse_llm_json_output
from utils import llm_call_helper, parse_llm_json_output

# Module-level logger
logger = logging.getLogger(__name__)

class OpenAPIRouter:
    """
    Initial router for the LangGraph agent.
    Uses an LLM to determine the high-level intent of the user's query
    and route to the appropriate starting node in the graph.
    Handles basic loop detection.
    This router is for an agent that DOES NOT perform API execution.
    """
    # Define the possible high-level intents/routes for the graph
    AVAILABLE_INTENTS = [
        "parse_openapi_spec", # User provides a spec
        "plan_execution", # User asks to plan a workflow
        "identify_apis", # User asks to identify relevant APIs
        "generate_payloads", # User asks to generate payload descriptions
        "generate_execution_graph", # User asks to generate the graph description
        "describe_graph", # User asks to describe the graph description
        "get_graph_json", # User asks for the graph description JSON
        "answer_openapi_query", # User asks a general question about spec/plan
        "handle_unknown", # Intent could not be determined
        "handle_loop", # Detected potential loop in routing
        # Removed "executor" as execution is not performed
    ]

    def __init__(self, router_llm: Any):
        """
        Initializes the router component.

        Args:
            router_llm: The language model instance dedicated to routing.
        """
        if not hasattr(router_llm, 'invoke'):
             raise TypeError("router_llm must have an 'invoke' method.")
        self.router_llm = router_llm
        logger.info("OpenAPIRouter initialized (without execution capabilities).")


    def route(self, state: BotState) -> str:
        """
        Determines the user's high-level intent and returns the name of the
        next node in the graph to transition to.
        Updates state.intent, state.previous_intent, and state.loop_counter.
        """
        user_input = state.user_input
        previous_intent = state.previous_intent
        loop_counter = state.loop_counter

        logger.debug(f"Routing user input: '{user_input}' (Previous intent: {previous_intent}, Loop counter: {loop_counter})")
        state.update_scratchpad_reason("router", f"Routing user input: '{user_input}'")

        # --- LLM Call to Determine Intent ---
        prompt = f"""
        The user's input is: "{user_input}"

        Determine the user's high-level intent based on their input and the current state.
        Choose ONE intent from the following list: {self.AVAILABLE_INTENTS}.
        If the user's input seems like an OpenAPI specification (starts with 'openapi:', 'swagger:', '{'{'}', '-', etc.), the intent is 'parse_openapi_spec'.
        If the user is asking to create a workflow or plan, the intent is 'plan_execution'.
        If the user is asking specifically about identifying APIs, use 'identify_apis'.
        If the user is asking about generating example data for API calls, use 'generate_payloads'.
        If the user is asking to generate the structure of an API workflow, use 'generate_execution_graph'.
        If the user is asking to describe the generated workflow graph, use 'describe_graph'.
        If the user is asking for the raw data of the generated graph, use 'get_graph_json'.
        If the user is asking a general question about the OpenAPI spec, the identified APIs, the graph, or the plan, use 'answer_openapi_query'.
        If the intent is unclear, use 'handle_unknown'.
        If you detect a potential loop based on the history, the 'handle_loop' node will be triggered by the graph flow, but the router should still return the determined intent.

        Current State Summary:
        - OpenAPI spec loaded: {'Yes' if state.openapi_schema else 'No'}
        - Execution graph description exists: {'Yes' if state.execution_graph else 'No'}
        - Execution plan description exists: {'Yes' if state.execution_plan else 'No'} ({len(state.execution_plan) if state.execution_plan else 0} steps)
        - Previous Intent: {previous_intent or 'None'}

        Output ONLY the chosen intent string.

        Chosen Intent:
        """
        try:
            llm_response = llm_call_helper(self.router_llm, prompt)
            determined_intent = llm_response.strip().lower()

            # Basic validation: check if the determined intent is in the allowed list
            if determined_intent not in self.AVAILABLE_INTENTS:
                logger.warning(f"Router LLM returned invalid intent '{determined_intent}'. Defaulting to handle_unknown.")
                determined_intent = "handle_unknown"
                state.update_scratchpad_reason("router", f"LLM returned invalid intent '{llm_response.strip()}'. Defaulted to '{determined_intent}'.")
            else:
                logger.debug(f"Router LLM determined intent: {determined_intent}")
                state.update_scratchpad_reason("router", f"LLM determined intent: '{determined_intent}'.")

        except Exception as e:
            logger.error(f"Error calling Router LLM: {e}", exc_info=True)
            determined_intent = "handle_unknown" # Fallback on error
            state.update_scratchpad_reason("router", f"LLM call failed: {e}. Defaulted to '{determined_intent}'.")
            state.response = "An internal error occurred while trying to understand your request. Please try again." # Set a user-facing error message


        # --- Apply Loop Detection based on determined_intent ---
        # We will keep the loop detection logic here, but the 'handle_loop' node
        # will now explain the situation without mentioning execution.
        if determined_intent == previous_intent and determined_intent not in ["handle_unknown", "handle_loop", "responder"]:
             # Increment loop counter if the same non-final intent repeats
             loop_counter += 1
             logger.warning(f"Router detected repeated intent: {determined_intent}. Loop counter: {loop_counter}")
             if loop_counter >= 3: # Increased threshold slightly for less aggressive loop detection
                 logger.error(f"Router detected potential loop (intent '{determined_intent}' repeated {loop_counter} times). Routing to handle_loop.")
                 determined_intent = "handle_loop"
                 # Reset counter after routing to handle_loop
                 state.loop_counter = 0
             else:
                 # Update counter but proceed with the determined intent for now
                 state.loop_counter = loop_counter
        else:
            # Reset loop counter if intent changes or is a final state
            state.loop_counter = 0
            logger.debug("Router: Intent changed or is final state. Resetting loop counter.")


        # Update previous_intent in state for the next turn
        state.previous_intent = determined_intent
        # Update current intent in state as well, for the graph conditional edge
        state.intent = determined_intent


        logger.info(f"Router routing to: {determined_intent}")

        # The router node must return the state instance when used with StateGraph
        # and the conditional edge checks state.intent.
        # The previous diff correctly changed the return type to BotState.
        # We just need to ensure state.intent is set before returning.
        return state
