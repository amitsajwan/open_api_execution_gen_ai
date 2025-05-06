# filename: router.py
import logging
from typing import Dict, Any, List, Optional
import re # Import regex for spec detection

# Assuming models.py defines BotState
from models import BotState

# Assuming utils.py has llm_call_helper and parse_llm_json_output_with_model
from utils import llm_call_helper # Removed unused import: parse_llm_json_output_with_model


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
    AVAILABLE_INTENTS = [
        "parse_openapi_spec", # User provides a spec
        "process_schema_completely", # NEW INTENT: Process the schema completely after parsing
        "plan_execution", # User asks to plan a workflow description
        "identify_apis", # User asks to identify relevant APIs
        "generate_payloads", # User asks to generate payload descriptions
        "generate_execution_graph", # User asks to generate the graph description
        "describe_graph", # User asks to describe the graph description
        "get_graph_json", # User asks for the graph description JSON
        "answer_openapi_query", # User asks a general question about spec/plan
        "handle_unknown", # Intent could not be determined
        "handle_loop", # Detected potential loop in routing
        "responder", # Router might route directly to responder
    ]

    # Regex to quickly check if input looks like JSON or YAML spec start
    SPEC_START_REGEX = re.compile(r"^\s*(\"openapi\":|\'openapi\':|openapi:|swagger:|{|-)")
    # Simple commands to check when schema exists
    SCHEMA_LOADED_COMMANDS = {
        "describe graph": "describe_graph",
        "show graph": "describe_graph",
        "get graph json": "get_graph_json",
        "show graph json": "get_graph_json",
        "generate graph": "generate_execution_graph", # Explicit command even if graph exists
        "generate plan": "plan_execution",
        "create plan": "plan_execution",
        "list apis": "answer_openapi_query", # Treat common queries like commands
        "identify apis": "identify_apis",
        "what endpoints": "answer_openapi_query",
        "generate payloads": "generate_payloads",
    }

    def __init__(self, router_llm: Any):
        if not hasattr(router_llm, 'invoke'):
             raise TypeError("router_llm must have an 'invoke' method.")
        self.router_llm = router_llm
        logger.info("OpenAPIRouter initialized (without execution capabilities).")


    def route(self, state: BotState) -> Dict[str, Any]:
        user_input = state.user_input
        user_input_lower = user_input.lower() if user_input else ""
        previous_intent = state.previous_intent
        loop_counter = state.loop_counter

        determined_intent: Optional[str] = None
        updates: Dict[str, Any] = {}
        updates['input_is_spec'] = False
        updates['extracted_params'] = None

        if not user_input:
            logger.warning("Router received empty user input. Routing to handle_unknown.")
            determined_intent = "handle_unknown"
            state.update_scratchpad_reason("router", "Empty input, routing to handle_unknown.")
            # Loop detection applied below

        # --- Heuristic Check for NEW OpenAPI Spec ---
        # Check first, as providing a spec should override other interpretations
        # Use length check AND regex. Don't trigger for short inputs.
        if user_input and len(user_input) > 150 and self.SPEC_START_REGEX.search(user_input):
             logger.info("Router heuristic detected potential NEW OpenAPI spec input.")
             state.update_scratchpad_reason("router", "Heuristic detected potential new spec. Routing to parse_openapi_spec.")
             determined_intent = "parse_openapi_spec"
             updates['input_is_spec'] = True # Set the flag in updates
             # Skip further checks if it looks like a spec

        # --- Special check if we just parsed a spec successfully (to trigger full processing) ---
        elif previous_intent == "parse_openapi_spec" and state.openapi_schema and not state.schema_summary:
            logger.info("Router detected newly parsed schema without processing. Routing to process_schema_completely.")
            state.update_scratchpad_reason("router", "Schema parsed but not processed. Routing to complete processing pipeline.")
            determined_intent = "process_schema_completely"
            # Add a response to the user that we're processing their schema
            updates['response'] = "I've successfully loaded your OpenAPI specification. I'll now analyze it completely to prepare summaries, identify endpoints, generate payload descriptions, and construct an execution graph. This will help me answer your questions more effectively."

        # --- Handle state if schema is ALREADY loaded ---
        elif determined_intent is None and state.openapi_schema:
            logger.debug("Schema exists. Checking specific commands or if input is a query.")
            state.update_scratchpad_reason("router", "Schema exists. Checking for commands/query.")

            # 1. Check for exact commands first
            for command, intent in self.SCHEMA_LOADED_COMMANDS.items():
                # Use simple matching (adjust if more complex matching is needed)
                if command in user_input_lower:
                    logger.info(f"Router detected specific command '{command}' with schema loaded. Routing to {intent}.")
                    determined_intent = intent
                    state.update_scratchpad_reason("router", f"Specific command '{command}' matched. Routing to {intent}.")
                    break # Stop checking commands once one matches

            # 2. If no command matched, use LLM to classify if it's a general query
            if determined_intent is None:
                query_classification_prompt = f"""
                An OpenAPI specification is loaded. The user's input is:
                "{user_input}"

                Consider the input. Is it primarily asking a question seeking information *about* the loaded OpenAPI specification, the identified APIs, payload descriptions, the described execution graph, or the described plan?
                Examples of questions: "What endpoints are available?", "Explain the create user API.", "How does the graph flow?", "What parameters does getUser take?".
                Examples of NOT questions: "generate the graph", "parse this spec: ...", "plan how to create a user".

                Based ONLY on the user input and the distinction between asking for information versus commanding an action, answer "YES" if it's a question about the loaded state/artifacts, or "NO" otherwise.

                Current State Summary (for context only, base decision on user input):
                - OpenAPI spec loaded: Yes
                - Schema Summary: {state.schema_summary[:500] + '...' if state.schema_summary else 'None'}
                - Identified APIs: {'Yes' if state.identified_apis else 'No'} ({len(state.identified_apis) if state.identified_apis else 0} found)
                - Execution graph description exists: {'Yes' if state.execution_graph else 'No'}
                - Execution plan description exists: {'Yes' if state.execution_plan else 'No'} ({len(state.execution_plan) if state.execution_plan else 0} steps)

                Is the input a question about the loaded state/artifacts? (YES/NO):
                """
                try:
                    llm_response = llm_call_helper(self.router_llm, query_classification_prompt)
                    classification = llm_response.strip().upper()
                    logger.debug(f"Router LLM query classification result: {classification}")

                    if classification == "YES":
                        logger.info("Router LLM classified input as a query about the loaded state. Routing to answer_openapi_query.")
                        determined_intent = "answer_openapi_query"
                        state.update_scratchpad_reason("router", "LLM classified as query about loaded state -> answer_openapi_query.")
                    else:
                        # If LLM says NO (it's not a query), and it wasn't a specific command,
                        # it might be an unknown command or something else. Let the general LLM below handle it cautiously.
                        logger.debug("Router LLM classified input NOT as a query about loaded state. Will proceed to general intent check.")
                        state.update_scratchpad_reason("router", "LLM classified NOT as query. Proceeding to general intent check.")
                        # determined_intent remains None

                except Exception as e:
                    logger.error(f"Error calling Router LLM for query classification: {e}", exc_info=True)
                    logger.warning("Router LLM query classification failed. Proceeding to general intent determination.")
                    state.update_scratchpad_reason("router", f"LLM query classification failed: {e}. Proceeding to general intent check.")
                    # determined_intent remains None

        # --- General Intent Determination (if still None) ---
        # This runs if:
        # - No schema was loaded initially.
        # - Schema was loaded, but input wasn't a new spec, wasn't a specific command, and wasn't classified as a query.
        if determined_intent is None:
            logger.debug("Determining intent using general LLM prompt.")
            state.update_scratchpad_reason("router", "Using general LLM prompt for intent.")

            # Construct a more careful prompt, especially when schema is loaded
            schema_loaded_context = "No OpenAPI spec is currently loaded."
            if state.openapi_schema:
                schema_loaded_context = f"""An OpenAPI spec IS currently loaded.
                - Schema Summary (first 500 chars): {state.schema_summary[:500] + '...' if state.schema_summary else 'None'}
                - Identified APIs: {'Yes' if state.identified_apis else 'No'} ({len(state.identified_apis) if state.identified_apis else 0} found)
                - Graph description exists: {'Yes' if state.execution_graph else 'No'}
                - Plan description exists: {'Yes' if state.execution_plan else 'No'} ({len(state.execution_plan) if state.execution_plan else 0} steps)
                IMPORTANT: Since a spec is loaded, DO NOT choose 'parse_openapi_spec' unless the input explicitly contains a new spec. Prioritize 'answer_openapi_query' for questions. If the input is a command not handled above (like 'generate payloads'), choose the corresponding action. If unsure, choose 'handle_unknown'."""

            prompt = f"""
            Determine the user's high-level intent from the list: {self.AVAILABLE_INTENTS}.

            User Input: "{user_input}"

            Current State Context:
            {schema_loaded_context}
            - Previous Intent: {previous_intent or 'None'}

            Instructions:
            1. Analyze the User Input.
            2. Consider the Current State Context. If a spec is loaded, be very careful about choosing `parse_openapi_spec`. Prefer `answer_openapi_query` for informational requests about the loaded spec/artifacts.
            3. Choose the *single best matching intent* from the list: {self.AVAILABLE_INTENTS}.
            4. If the intent is unclear or doesn't fit well, choose `handle_unknown`.
            5. Output ONLY the chosen intent string.

            Chosen Intent:
            """
            try:
                llm_response = llm_call_helper(self.router_llm, prompt)
                determined_intent_llm = llm_response.strip().lower() # Ensure lowercase

                # Validation
                if determined_intent_llm not in self.AVAILABLE_INTENTS:
                    logger.warning(f"Router LLM returned invalid intent '{determined_intent_llm}'. Defaulting to handle_unknown.")
                    determined_intent = "handle_unknown"
                    state.update_scratchpad_reason("router", f"General LLM returned invalid intent '{llm_response.strip()}'. Defaulted to handle_unknown.")
                # Prevent LLM from choosing parse_openapi_spec if schema exists (heuristic should catch specs)
                elif determined_intent_llm == "parse_openapi_spec" and state.openapi_schema:
                    logger.warning(f"Router LLM chose '{determined_intent_llm}' when schema already exists. Overriding to handle_unknown.")
                    determined_intent = "handle_unknown"
                    state.update_scratchpad_reason("router", f"General LLM chose '{determined_intent_llm}' with existing schema. Defaulted to handle_unknown.")
                # Don't allow LLM to choose process_schema_completely directly
                elif determined_intent_llm == "process_schema_completely":
                    logger.warning(f"Router LLM chose '{determined_intent_llm}' which should only be triggered internally. Overriding to handle_unknown.")
                    determined_intent = "handle_unknown"
                    state.update_scratchpad_reason("router", f"General LLM chose '{determined_intent_llm}' which is internal only. Defaulted to handle_unknown.")
                else:
                    determined_intent = determined_intent_llm
                    logger.debug(f"Router LLM determined general intent: {determined_intent}")
                    state.update_scratchpad_reason("router", f"General LLM determined intent: '{determined_intent}'.")

            except Exception as e:
                logger.error(f"Error calling Router LLM for general intent: {e}", exc_info=True)
                determined_intent = "handle_unknown"
                state.update_scratchpad_reason("router", f"General LLM call failed: {e}. Defaulted to handle_unknown.")

        # --- Final Fallback ---
        if determined_intent is None:
             logger.error("Router failed to determine intent after all checks. Defaulting to handle_unknown.")
             determined_intent = "handle_unknown"
             state.update_scratchpad_reason("router", "Failed to determine intent after all checks. Defaulted to handle_unknown.")

        # --- Apply Loop Detection ---
        final_intent = determined_intent
        # Check if the same valid, non-final intent is repeating
        if determined_intent == previous_intent and determined_intent not in ["handle_unknown", "handle_loop", "parse_openapi_spec", "process_schema_completely", "responder"]:
             loop_counter += 1
             updates['loop_counter'] = loop_counter
             logger.warning(f"Router detected repeated intent: {determined_intent}. Loop counter: {loop_counter}")
             if loop_counter >= 3:
                 logger.error(f"Router detected potential loop. Routing to handle_loop.")
                 final_intent = "handle_loop"
                 updates['loop_counter'] = 0 # Reset counter
             # else: proceed with determined_intent, counter updated
        else:
            # Reset loop counter if intent changes or is a final/parse state
            updates['loop_counter'] = 0 # Reset counter

        # --- Set Final Updates ---
        updates['previous_intent'] = final_intent
        updates['intent'] = final_intent
        updates['__next__'] = final_intent # Set the actual next node

        logger.info(f"Router routing to: {final_intent}")
        return updates
