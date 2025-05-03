# router.py

import logging
from models import BotState
from typing import List, Any
from persistence import save_state   # your save/load utilities

logger = logging.getLogger(__name__)

# Add the special loop‑break intent
AVAILABLE_INTENTS: List[str] = [
    "parse_openapi_spec",
    "identify_apis",
    "generate_payloads",
    "generate_execution_graph",
    "describe_graph",
    "get_graph_json",
    "handle_unknown",
    "handle_loop"
]

class OpenAPIRouter:
    def __init__(self, llm: Any):
        self.llm = llm

    

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
        1) Guard: if no user_input → handle_unknown
        2) Cycle‑detection: break if same intent repeats
        3) Determine next intent and extract params
        4) Persist state, then return intent
        """
        logger.debug(f"--- Router Start (Session: {state.session_id}) ---")
    
        # 1. Guard: require user_input
        if not state.user_input:
            logger.warning("Router called with no user_input in state.")
            state.intent = "handle_unknown"
            # optional: record reason in scratchpad
            state.update_scratchpad_reason("Router", "No user input provided.")
            save_state(state)
            logger.debug(f"--- Router End (Returning: {state.intent}) ---")
            return state.intent
    
        # 2. Cycle detection
        if state.intent == state.previous_intent:
            state.loop_counter += 1
        else:
            state.loop_counter = 0
        state.previous_intent = state.intent
    
        # break out if looping too much
        if state.loop_counter >= 2:
            logger.info("Loop detected; routing to handle_loop")
            state.intent = "handle_loop"
            save_state(state)
            return state.intent
    
        # 3. Normal routing
        next_intent = self._determine_intent(state)
        self._extract_parameters(state)
    
        # 4. Persist and return
        save_state(state)
        logger.debug(f"--- Router End (Returning: {next_intent}) ---")
        return next_intent
    
