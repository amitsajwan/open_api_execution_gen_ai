# router.py

import logging
from models import BotState
from typing import List, Any
# Assuming persistence is implemented for saving/loading state
# from persistence import save_state, load_state # your save/load utilities


# Add 'planner' as a possible intent the router can return.
# The planner will handle the more granular decisions about API execution or graph building.
AVAILABLE_INTENTS: List[str] = [
    "parse_openapi_spec",
    "identify_apis",
    "generate_payloads",
    "generate_execution_graph",
    "describe_graph",
    "get_graph_json",
    "handle_unknown",
    "handle_loop",
    "planner", # Added planner as a destination from the initial router
    # Add validate_graph, add_graph_edge if they are initial entry points from user command
    # "validate_graph",
    # "add_graph_edge",
]

logger = logging.getLogger(__name__)

class OpenAPIRouter:
    
    # Add 'planner' as a possible intent the router can return.
    # The planner will handle the more granular decisions about API execution or graph building.
    AVAILABLE_INTENTS: List[str] = [
        "parse_openapi_spec",
        "identify_apis",
        "generate_payloads",
        "generate_execution_graph",
        "describe_graph",
        "get_graph_json",
        "handle_unknown",
        "handle_loop",
        "planner", # Added planner as a destination from the initial router
        # Add validate_graph, add_graph_edge if they are initial entry points from user command
        # "validate_graph",
        # "add_graph_edge",
    ]

    def __init__(self, llm: Any):
        """
        Initializes the router component.

        Args:
            llm: The language model instance used for initial intent routing and parameter extraction.
        """
        if not hasattr(llm, 'invoke'):
             # Added a check similar to core_logic
             raise TypeError("Router LLM must have an 'invoke' method.")
        self.llm = llm
        logger.info("OpenAPIRouter initialized.")


    def _determine_intent(self, state: BotState) -> str:
        """
        Uses the router LLM to determine the high-level intent from user input.
        This LLM is for initial routing, not detailed API execution planning.
        It should decide if the user wants to:
        - Parse a spec? -> 'parse_openapi_spec'
        - Build/Generate a graph? -> 'generate_execution_graph'
        - Describe the graph? -> 'describe_graph'
        - Get graph JSON? -> 'get_graph_json'
        - Handle something else dynamically (most cases)? -> 'planner'
        - Is it an unknown command? -> 'handle_unknown'
        - Is it a loop condition? (handled by route method's guard) -> 'handle_loop'

        Sets state.intent before returning.
        """
        logger.debug("Router: Determining intent with LLM.")
        user_input = state.user_input

        if not user_input:
             logger.warning("Router: _determine_intent called with no user_input.")
             state.intent = "handle_unknown"
             return state.intent # Should be caught by the guard in route, but good practice

        # --- LLM Intent Determination Logic Placeholder ---
        # Prompt the self.llm (router_llm) with the user_input.
        # Provide it with the list of AVAILABLE_INTENTS (excluding 'planner' maybe,
        # or instructing it that 'planner' is the default for complex requests).
        # Instruct it to return *only* one of the AVAILABLE_INTENTS as a string.

        # Example Prompt Idea:
        # "Analyze the user query and determine the primary high-level intent.
        # Choose ONE from the following options: parse_openapi_spec, generate_execution_graph,
        # describe_graph, get_graph_json, handle_unknown. If the query is not a clear command
        # for one of these, or requires executing APIs from a graph, indicate 'planner'.
        # Return ONLY the chosen intent string."
        # Query: "{user_input}"
        # Options: [...]
        # Chosen Intent:

        llm_response_intent = "planner" # Simulate LLM returning 'planner' for most cases

        # Simulate some explicit command detection for initial routing
        lower_query = user_input.lower()
        if "parse spec" in lower_query or "load spec" in lower_query:
             llm_response_intent = "parse_openapi_spec"
        elif "generate graph" in lower_query or "build workflow" in lower_query:
             llm_response_intent = "generate_execution_graph"
        elif "describe graph" in lower_query:
             llm_response_intent = "describe_graph"
        elif "get graph json" in lower_query:
             llm_response_intent = "get_graph_json"
        # Add validate_graph, add_graph_edge if they are explicit initial commands
        # elif "validate graph" in lower_query:
        #      llm_response_intent = "validate_graph"
        # elif "add edge" in lower_query:
        #      llm_response_intent = "add_graph_edge"
        # Add handle_unknown as a possible explicit return if LLM determines it
        # elif "unknown" in llm_response_raw: # Example if LLM returns "unknown"
        #      llm_response_intent = "handle_unknown"


        # Validate the LLM's response against AVAILABLE_INTENTS
        if llm_response_intent not in AVAILABLE_INTENTS:
            logger.warning(f"Router LLM returned invalid intent: {llm_response_intent}. Defaulting to 'planner'.")
            state.intent = "planner" # Default to planner if LLM output is unexpected
        else:
            state.intent = llm_response_intent

        logger.debug(f"Router: Determined intent: {state.intent}")
        return state.intent


    def _extract_parameters(self, state: BotState) -> None:
        """
        Uses the router LLM to extract parameters relevant to the determined intent
        from the user input.
        Stores extracted parameters in state.extracted_params.
        This is less critical now as the planner will do more detailed parameter extraction
        for API execution, but might be needed for initial commands (e.g., parameters for add_edge).
        """
        logger.debug("Router: Extracting parameters with LLM.")
        user_input = state.user_input
        current_intent = state.intent # Use the intent already determined

        if not user_input or not current_intent or current_intent in ["handle_unknown", "handle_loop", "planner"]:
             # No need to extract parameters for these intents at this stage
             state.extracted_params = {}
             logger.debug("Router: No parameter extraction needed for current intent or input missing.")
             return

        # --- LLM Parameter Extraction Logic Placeholder ---
        # Prompt the self.llm (router_llm) with the user_input and the determined current_intent.
        # Instruct it to extract relevant parameters for that specific intent
        # and return them as a JSON object.
        # Example: If intent is 'add_graph_edge', extract 'from_node', 'to_node', 'description'.

        # Example Prompt Idea:
        # "Analyze the user query and extract parameters relevant to the intent '{current_intent}'.
        # Return the parameters as a JSON object.
        # User Query: '{user_input}'
        # Parameters (JSON):"

        extracted_params_json_str = "{}" # Simulate LLM returning empty JSON by default

        try:
            # llm_response_params = llm_call_helper(self.llm, prompt_for_params) # Need a helper for this
            # extracted_params = parse_llm_json_output(llm_response_params) # Need a JSON parsing helper
            extracted_params = json.loads(extracted_params_json_str) # Placeholder parsing

            if not isinstance(extracted_params, dict):
                 logger.warning(f"Router LLM parameter extraction did not return a dict: {extracted_params_json_str}")
                 extracted_params = {} # Default to empty dict on failure

        except Exception as e:
            logger.error(f"Router: Error during parameter extraction LLM call: {e}", exc_info=True)
            extracted_params = {} # Default to empty dict on error

        state.extracted_params = extracted_params
        logger.debug(f"Router: Extracted parameters: {state.extracted_params}")


    # The main routing function used by LangGraph
    def route(self, state: BotState) -> str:
        """
        1) Guard: if no user_input -> handle_unknown
        2) Cycle-detection: break if same intent repeats (using loop_counter)
        3) Determine next intent and extract params using LLM
        4) Persist state, then return intent (which is the next node name)
        """
        logger.debug(f"--- Router Start (Session: {state.session_id}) ---")
        # logger.debug(f"Initial State in Router: {state.model_dump()}") # Log full state for debugging

        # 1. Guard: require user_input for initial routing
        if not state.user_input:
            logger.warning("Router called with no user_input in state.")
            state.intent = "handle_unknown"
            # optional: record reason in scratchpad
            state.update_scratchpad_reason("Router", "No user input provided.")
            # save_state(state) # Save state before exiting router if needed
            logger.debug(f"--- Router End (Returning: {state.intent}) ---")
            return state.intent # Route to handle_unknown node

        # 2. Cycle detection
        # Check if the determined intent is the same as the previous one
        # Note: This checks intent returned by _determine_intent, not the node that just ran.
        # It might be better to check the *node* that just completed vs the *new* intent.
        # For now, let's stick to checking determined intent vs previous determined intent.
        # The planner node also has a loop check based on loop_counter.
        determined_intent = self._determine_intent(state) # Determine intent first

        if determined_intent == state.previous_intent and determined_intent not in ["planner", "executor"]:
             # Increment loop counter only if the determined intent repeats AND it's not the planner/executor
             # We allow planner/executor to appear consecutively as they manage internal loops/steps.
             state.loop_counter += 1
             logger.debug(f"Router: Intent '{determined_intent}' repeated. Loop counter: {state.loop_counter}")
        else:
             state.loop_counter = 0 # Reset counter if intent changes
             logger.debug(f"Router: Intent changed to '{determined_intent}'. Loop counter reset.")

        state.previous_intent = determined_intent # Update previous intent

        # break out if looping too much on non-planner/executor nodes
        if state.loop_counter >= 2: # Threshold can be adjusted
            logger.info(f"Router: Loop detected for intent '{determined_intent}'; routing to handle_loop")
            state.intent = "handle_loop" # Force intent to handle_loop
            # save_state(state) # Save state if needed before returning
            logger.debug(f"--- Router End (Returning: {state.intent}) ---")
            return state.intent

        # 3. Normal routing - intent was already determined by _determine_intent
        # Extract parameters based on the determined intent
        self._extract_parameters(state)

        # The determined intent is the name of the next node to return
        next_node_name = state.intent

        # 4. Persist and return
        # save_state(state) # Save state if needed before returning

        logger.debug(f"--- Router End (Returning: {next_node_name}) ---")
        return next_node_name # Return the name of the next node

