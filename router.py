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
        # your existing LLM or rule‑based intent detection
        # must set state.intent before returning
        intent = ... 
        state.intent = intent
        return intent

    def _extract_parameters(self, state: BotState) -> None:
        # your existing param extraction, e.g.:
        # state.extracted_params = ...
        ...

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
