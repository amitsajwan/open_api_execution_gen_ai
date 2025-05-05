# filename: router.py
import logging
from typing import Dict, Any, List, Optional
import re # Import regex for spec detection

# Assuming models.py defines BotState
from models import BotState

# Assuming utils.py has llm_call_helper and the renamed JSON parsing utility
# Corrected import statement:
from utils import llm_call_helper, parse_llm_json_output_with_model

# Module-level logger
logger = logging.getLogger(__name__)

class OpenAPIRouter:
    """
    Initial router for the LangGraph agent.
    Uses heuristics and an LLM to determine the high-level intent of the user's query
    and route to the appropriate starting node in the graph.
    Handles basic loop detection. Sets the input_is_spec flag in the state.
    This router is for an agent that DOES NOT perform API execution.
    """
    # Define the possible high-level intents/routes for the graph
    AVAILABLE_INTENTS = [
        "parse_openapi_spec", # User provides a spec
        "plan_execution", # User asks to plan a workflow description
        "identify_apis", # User asks to identify relevant APIs
        "generate_payloads", # User asks to generate payload descriptions
        "generate_execution_graph", # User asks to generate the graph description
        "describe_graph", # User asks to describe the graph description
        "get_graph_json", # User asks for the graph description JSON
        "answer_openapi_query", # User asks a general question about spec/plan
        "handle_unknown", # Intent could not be determined
        "handle_loop", # Detected potential loop in routing
    ]

    # Regex to quickly check if input looks like JSON or YAML spec start
    # Looks for 'openapi:', 'swagger:', initial '{', or initial '-' (for YAML list start)
    # Allows for optional leading whitespace/newlines
    SPEC_START_REGEX = re.compile(r"^\s*(\"openapi\":|\'openapi\':|openapi:|swagger:|{|-)")

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


    def route(self, state: BotState) -> BotState:
        """
        Determines the user's high-level intent and returns the name of the
        next node in the graph to transition to.
        Updates state.intent, state.previous_intent, state.loop_counter, and state.input_is_spec.
        """
        user_input = state.user_input
        previous_intent = state.previous_intent
        loop_counter = state.loop_counter

        # Reset the input_is_spec flag at the start of routing
        state.input_is_spec = False

        if not user_input:
            logger.warning("Router received empty user input. Routing to handle_unknown.")
            state.intent = "handle_unknown"
            state.previous_intent = "handle_unknown" # Avoid loop detection on empty input
            state.loop_counter = 0
            state.update_scratchpad_reason("router", "Empty input, routing to handle_unknown.")
            return state # Return state directly for handle_unknown

        logger.debug(f"Routing user input: '{user_input[:100]}...' (Prev: {previous_intent}, Loop: {loop_counter})")
        state.update_scratchpad_reason("router", f"Routing user input: '{user_input[:100]}...'")

        # --- Heuristic Check for OpenAPI Spec ---
        if self.SPEC_START_REGEX.search(user_input):
            logger.info("Router heuristic detected potential OpenAPI spec input.")
            state.update_scratchpad_reason("router", "Heuristic detected potential spec. Routing to parse_openapi_spec.")
            determined_intent = "parse_openapi_spec"
            state.input_is_spec = True # Set the flag for the parser node
        else:
            # --- LLM Call to Determine Intent (if not detected as spec) ---
            logger.debug("Input not detected as spec by heuristic. Using LLM for intent.")
            prompt = f"""
            The user's input is: "{user_input}"

            Determine the user's high-level intent based on their input and the current state.
            Choose ONE intent from the following list: {self.AVAILABLE_INTENTS}.
            Do NOT choose 'parse_openapi_spec' unless the input explicitly asks to parse something that might be a spec but wasn't caught by the initial check.

            Consider these goals:
            - If the user asks to create a workflow description or plan, choose 'plan_execution'.
            - If the user asks specifically about identifying APIs for a purpose, choose 'identify_apis'.
            - If the user asks about generating example data/payload *descriptions* for API calls, choose 'generate_payloads'.
            - If the user asks to generate the structure/diagram/graph of an API workflow description, choose 'generate_execution_graph'.
            - If the user asks to describe or explain the generated workflow graph description, choose 'describe_graph'.
            - If the user asks for the raw data/JSON of the generated graph description, choose 'get_graph_json'.
            - If the user asks a general question about the loaded OpenAPI spec, the identified APIs, the graph, or the plan description, choose 'answer_openapi_query'.
            - If the intent is unclear or doesn't fit the above, choose 'handle_unknown'.

            Current State Summary:
            - OpenAPI spec loaded: {'Yes' if state.openapi_schema else 'No'}
            - Execution graph description exists: {'Yes' if state.execution_graph else 'No'}
            - Execution plan description exists: {'Yes' if state.execution_plan else 'No'} ({len(state.execution_plan) if state.execution_plan else 0} steps)
            - Previous Intent: {previous_intent or 'None'}

            Output ONLY the chosen intent string from the list.

            Chosen Intent:
            """
            try:
                llm_response = llm_call_helper(self.router_llm, prompt)
                determined_intent = llm_response.strip().lower() # Ensure lowercase for comparison

                # Basic validation: check if the determined intent is in the allowed list
                if determined_intent not in self.AVAILABLE_INTENTS:
                    logger.warning(f"Router LLM returned invalid intent '{determined_intent}'. Defaulting to handle_unknown.")
                    determined_intent = "handle_unknown"
                    state.update_scratchpad_reason("router", f"LLM returned invalid intent '{llm_response.strip()}'. Defaulted to '{determined_intent}'.")
                # Prevent LLM from choosing parse_openapi_spec if heuristic didn't catch it
                elif determined_intent == "parse_openapi_spec":
                     logger.warning(f"Router LLM chose '{determined_intent}' unexpectedly. Overriding to handle_unknown.")
                     determined_intent = "handle_unknown"
                     state.update_scratchpad_reason("router", f"LLM chose '{determined_intent}' unexpectedly. Defaulted to '{determined_intent}'.")
                else:
                    logger.debug(f"Router LLM determined intent: {determined_intent}")
                    state.update_scratchpad_reason("router", f"LLM determined intent: '{determined_intent}'.")

            except Exception as e:
                logger.error(f"Error calling Router LLM: {e}", exc_info=True)
                determined_intent = "handle_unknown" # Fallback on error
                state.update_scratchpad_reason("router", f"LLM call failed: {e}. Defaulted to '{determined_intent}'.")
                # Set an intermediate response here if desired, although handle_unknown node will generate one
                # state.response = "An internal error occurred while trying to understand your request. Please try again."


        # --- Apply Loop Detection based on determined_intent ---
        if determined_intent == previous_intent and determined_intent not in ["handle_unknown", "handle_loop", "responder", "parse_openapi_spec"]: # Add parse_openapi_spec here - unlikely loop target
             # Increment loop counter if the same non-final, non-parse intent repeats
             loop_counter += 1
             logger.warning(f"Router detected repeated intent: {determined_intent}. Loop counter: {loop_counter}")
             if loop_counter >= 3:
                 logger.error(f"Router detected potential loop (intent '{determined_intent}' repeated {loop_counter} times). Routing to handle_loop.")
                 final_intent = "handle_loop"
                 # Reset counter after routing to handle_loop
                 state.loop_counter = 0
             else:
                 # Update counter but proceed with the determined intent for now
                 state.loop_counter = loop_counter
                 final_intent = determined_intent
        else:
            # Reset loop counter if intent changes or is final/parse state
            state.loop_counter = 0
            final_intent = determined_intent
            # logger.debug("Router: Intent changed or is final/parse state. Resetting loop counter.")


        # Update state for the next turn and for the graph conditional edge
        state.previous_intent = final_intent # Store the final decision
        state.intent = final_intent

        logger.info(f"Router routing to: {final_intent}")

        # Return the updated state instance
        return state
