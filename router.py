import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError

# Assuming models.py defines BotState
from models import BotState

# Module-level logger
logger = logging.getLogger(__name__)

class OpenAPIRouter:
    """
    Initial router for the LangGraph agent.
    Uses an LLM to determine the high-level intent of the user's query
    and route to the appropriate starting node in the graph.
    Also handles basic loop detection.
    """
    # Define the possible high-level intents/routes
    AVAILABLE_INTENTS = [
        "parse_openapi_spec",
        "generate_execution_graph",
        "executor",
        "describe_graph",
        "get_graph_json",
        "answer_openapi_query", # Added the new query answering intent
        "handle_unknown",
        "handle_loop",
        # Note: identify_apis and generate_payloads are now typically
        # triggered proactively by the planner after parse_openapi_spec,
        # but could potentially be direct intents if needed.
        # For now, the planner handles routing to them.
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
        logger.info("OpenAPIRouter initialized.")

    def route(self, state: BotState) -> str:
        """
        Analyzes the user's input and determines the next node to route to.
        Uses an LLM for intent classification.
        Implements basic loop detection by checking recent intents.

        Args:
            state: The current BotState.

        Returns:
            The name of the next node to execute.
        """
        logger.debug("---ROUTER NODE---")
        user_input = state.user_input
        previous_intent = state.previous_intent
        loop_counter = state.loop_counter

        logger.debug(f"User Input: {user_input}")
        logger.debug(f"Previous Intent: {previous_intent}")
        logger.debug(f"Loop Counter: {loop_counter}")

        # --- Loop Detection ---
        # If the previous intent was the same as the current determined intent
        # multiple times, route to a loop handling node.
        # This check happens *after* determining the current intent below.

        # --- Intent Classification using LLM ---
        prompt = f"""
        Analyze the following user request and determine the primary high-level intent.
        Choose ONE intent from the list below that best matches the user's goal.

        Available Intents:
        {', '.join(self.AVAILABLE_INTENTS)}

        Consider the user's request carefully: "{user_input}"

        Output ONLY the name of the chosen intent, exactly as it appears in the list.
        If the request does not clearly match any of the specific intents, choose 'handle_unknown'.
        Do not include any other text or punctuation.

        Chosen Intent:
        """
        determined_intent = "handle_unknown" # Default fallback

        try:
            llm_response = self.router_llm.invoke(prompt)
            # Clean the response to get just the intent name
            cleaned_response = llm_response.strip().lower().replace('.', '').replace('"', '').replace("'", "")

            # Validate the determined intent against the available list
            if cleaned_response in self.AVAILABLE_INTENTS:
                determined_intent = cleaned_response
                logger.info(f"Router LLM determined intent: {determined_intent}")
            else:
                logger.warning(f"Router LLM returned invalid intent '{cleaned_response}'. Defaulting to handle_unknown.")
                determined_intent = "handle_unknown"

        except Exception as e:
            logger.error(f"Error during router LLM call: {e}", exc_info=True)
            logger.warning("Router LLM call failed. Defaulting to handle_unknown.")
            determined_intent = "handle_unknown" # Default on error


        # --- Apply Loop Detection based on determined_intent ---
        if determined_intent == previous_intent and determined_intent not in ["handle_unknown", "handle_loop", "responder"]:
             # Increment loop counter if the same non-final intent repeats
             loop_counter += 1
             logger.warning(f"Router detected repeated intent: {determined_intent}. Loop counter: {loop_counter}")
             if loop_counter >= 2: # Threshold for loop detection
                 logger.error(f"Router detected potential loop (intent '{determined_intent}' repeated {loop_counter} times). Routing to handle_loop.")
                 determined_intent = "handle_loop"
                 # Reset counter after routing to handle_loop? Or let handle_loop manage it?
                 # For now, let's reset it here to break the loop detection for the *next* turn.
                 state.loop_counter = 0 # Resetting counter
             else:
                 # Update counter but proceed with the determined intent for now
                 state.loop_counter = loop_counter
        else:
            # Reset loop counter if intent changes or is a final state
            state.loop_counter = 0
            logger.debug("Router: Intent changed or is final state. Resetting loop counter.")


        # Update previous_intent in state for the next turn
        state.previous_intent = determined_intent

        logger.info(f"Router routing to: {determined_intent}")
        return determined_intent

