import json
import logging
from typing import Any, Dict, List, Optional, Tuple

# Assuming models.py defines BotState and intent parameter models
from models import (
    BotState, AddEdgeParams, GeneratePayloadsParams, GenerateGraphParams
) 
# Assuming utils.py defines helpers
from utils import llm_call_helper, save_state, load_state, parse_llm_json_output

# Module-level logger
logger = logging.getLogger(__name__)

# Define the available tools/intents the router can choose from
# These should match the method names in OpenAPICoreLogic + 'router'
AVAILABLE_INTENTS = [
    "parse_openapi_spec",
    "identify_apis",
    "generate_payloads",
    "generate_execution_graph",
    "add_graph_edge",
    "validate_graph",
    "describe_graph",
    "get_graph_json",
    "handle_unknown", # Fallback
]

# Intents that might require parameter extraction
INTENTS_REQUIRING_PARAMS = {
    "add_graph_edge": AddEdgeParams,
    "generate_payloads": GeneratePayloadsParams,
    "generate_execution_graph": GenerateGraphParams,
}


class OpenAPIRouter:
    """
    Routes user input to the appropriate tool/action using a router LLM.
    Handles state loading/saving and parameter extraction.
    """
    def __init__(self, router_llm: Any):
        """
        Initializes the router.

        Args:
            router_llm: The language model instance dedicated to routing and parameter extraction.
        """
        if not hasattr(router_llm, 'invoke'):
             raise TypeError("router_llm must have an 'invoke' method.")
        self.router_llm = router_llm
        logger.info("OpenAPIRouter initialized.")

    def _determine_intent(self, state: BotState) -> str:
        """Uses the router LLM to determine the next action/intent."""
        # Prepare context for the router LLM
        state_summary = {
            "has_spec_text": bool(state.openapi_spec_text),
            "has_schema": bool(state.openapi_schema),
            "has_identified_apis": bool(state.identified_apis),
            "has_payloads": bool(state.generated_payloads),
            "has_graph": bool(state.execution_graph),
            "last_response": state.response[-200:] if state.response else None, # Last 200 chars
            "scratchpad_summary": state.scratchpad.get('reasoning_log', '')[-500:] # Last 500 chars of reasoning
        }
        state_summary_str = json.dumps(state_summary, indent=2)
        
        intent_list_str = ", ".join(AVAILABLE_INTENTS)

        prompt = f"""
        Given the current state summary and the user's latest input, determine the most appropriate next action (intent) from the available list.

        Current State Summary:
        ```json
        {state_summary_str}
        ```

        User Input: "{state.user_input}"

        Available Intents: [{intent_list_str}]

        Consider the state:
        - If no spec is loaded, 'parse_openapi_spec' might be needed if the input looks like a spec.
        - If a spec is loaded but no graph exists, 'generate_execution_graph' or 'identify_apis' might be appropriate.
        - If a graph exists, actions like 'add_graph_edge', 'validate_graph', 'describe_graph', 'get_graph_json', or 'generate_payloads' might be relevant.
        - If the input is unclear or doesn't match an action, choose 'handle_unknown'.

        Output ONLY the chosen intent string (e.g., "generate_execution_graph").

        Chosen Intent:
        """
        try:
            llm_response = llm_call_helper(self.router_llm, prompt).strip()
            print(" llm_response ", llm_response)
            # Basic validation: check if the response is one of the available intents
            if llm_response in AVAILABLE_INTENTS:
                chosen_intent = llm_response
                logger.info(f"Router determined intent: {chosen_intent}")
            else:
                logger.warning(f"Router LLM returned invalid intent '{llm_response}'. Defaulting to 'handle_unknown'.")
                chosen_intent = "handle_unknown"
                
        except Exception as e:
            logger.error(f"Router LLM failed during intent determination: {e}", exc_info=True)
            chosen_intent = "handle_unknown" # Fallback on error

        state.intent = chosen_intent
        state.update_scratchpad_reason("Router (Intent)", f"Input: '{state.user_input}'. Determined Intent: {chosen_intent}")
        return chosen_intent

    def _extract_parameters(self, state: BotState) -> None:
        """
        If the chosen intent requires parameters, use the router LLM to extract them
        from the user input and potentially the conversation history/state.
        """
        intent = state.intent
        if not intent or intent not in INTENTS_REQUIRING_PARAMS:
            state.extracted_params = None # Clear params if not needed
            logger.debug(f"No parameter extraction needed for intent: {intent}")
            return

        param_model = INTENTS_REQUIRING_PARAMS[intent]
        model_schema = param_model.model_json_schema(ref_template="#/components/schemas/{model}")
        
        # Extract relevant parts of the schema for context
        components = model_schema.pop("components", None) # Extract components if they exist
        schema_str = json.dumps(model_schema, indent=2)
        components_str = json.dumps(components, indent=2) if components else "{}"

        # Use scratchpad for recent history context
        history = state.scratchpad.get('reasoning_log', '')[-1000:] # Last 1000 chars

        prompt = f"""
        Extract parameters from the user's input based on the required JSON schema for the intent '{intent}'.
        Consider the recent conversation history for context if needed.

        User Input: "{state.user_input}"

        Required Parameter JSON Schema (excluding components):
        ```json
        {schema_str}
        ```
        
        Referenced Components (if any):
        ```json
        {components_str}
        ```

        Recent Conversation History (summary):
        {history}
        
        Instructions:
        - Analyze the user input and history.
        - Populate the fields defined in the JSON schema with information extracted from the input/history.
        - If a required field cannot be found, you may need to omit it or use a sensible default if appropriate, but prioritize extracting from the user's text.
        - Output ONLY the populated JSON object containing the extracted parameters.

        Extracted Parameters (JSON Object):
        """
        try:
            llm_response = llm_call_helper(self.router_llm, prompt)
            extracted_data = parse_llm_json_output(llm_response) # Expecting dict

            if extracted_data and isinstance(extracted_data, dict):
                 # Validate against the Pydantic model (optional but recommended)
                 try:
                      # Validate raw dict before assigning
                      param_model.model_validate(extracted_data) 
                      state.extracted_params = extracted_data
                      logger.info(f"Successfully extracted parameters for intent '{intent}': {extracted_data}")
                      state.update_scratchpad_reason("Router (Params)", f"Extracted for '{intent}': {extracted_data}")
                 except Exception as ve: # Catches Pydantic's ValidationError and others
                      logger.error(f"Extracted parameters failed validation for {intent} ({param_model.__name__}): {ve}\nRaw JSON: {extracted_data}", exc_info=True)
                      state.extracted_params = None # Discard invalid params
                      state.update_scratchpad_reason("Router (Params)", f"Param validation failed for '{intent}': {ve}")
            else:
                 logger.warning(f"LLM did not return valid JSON for parameter extraction for intent '{intent}'. Raw response: {llm_response[:500]}...")
                 state.extracted_params = None
                 state.update_scratchpad_reason("Router (Params)", f"Param extraction failed for '{intent}'. No valid JSON.")

        except Exception as e:
            logger.error(f"Router LLM failed during parameter extraction for intent '{intent}': {e}", exc_info=True)
            state.extracted_params = None # Clear params on error
            state.update_scratchpad_reason("Router (Params)", f"Param extraction LLM call failed for '{intent}': {e}")


    def route(self, state: BotState) -> str:
        """
        The main routing logic called by LangGraph.
        Loads state, determines intent, extracts parameters, saves state,
        and returns the name of the next node (intent) to execute.
        """
        logger.debug(f"--- Router Start (Session: {state.session_id}) ---")
        
        # 1. Load potentially persisted state (optional, depending on graph setup)
        # If using MemorySaver checkpointer, LangGraph handles state loading implicitly.
        # If managing state manually per session:
        # loaded = load_state(state.session_id)
        # if loaded:
        #     state = loaded # Overwrite incoming state with persisted one
        #     logger.info(f"Router loaded persisted state for session {state.session_id}")
        # else:
        #     logger.info(f"Router starting with fresh state for session {state.session_id}")
            
        # Ensure user input is present
        if not state.user_input:
             logger.warning("Router called with no user input in state.")
             # Decide how to handle this - maybe default to a help/unknown intent?
             state.intent = "handle_unknown" 
             state.update_scratchpad_reason("Router", "No user input provided.")
             save_state(state) # Save state even if input missing
             logger.debug(f"--- Router End (Returning: {state.intent}) ---")
             return state.intent

        # 2. Determine Intent
        next_intent = self._determine_intent(state) # Updates state.intent

        # 3. Extract Parameters (if needed)
        self._extract_parameters(state) # Updates state.extracted_params

        # 4. Persist State
        save_state(state)

        logger.debug(f"--- Router End (Returning: {next_intent}) ---")
        # 5. Return the determined intent string for LangGraph conditional routing
        return next_intent

