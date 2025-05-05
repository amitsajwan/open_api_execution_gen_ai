# filename: router.py
import logging
from typing import Dict, Any, List, Optional
import re # Import regex for spec detection

# Assuming models.py defines BotState
from models import BotState

# Assuming utils.py has llm_call_helper and parse_llm_json_output_with_model
from utils import llm_call_helper, parse_llm_json_output_with_model # Corrected import name


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


    def route(self, state: BotState) -> str: # Return type hint is str for the node name
        """
        Determines the user's high-level intent and returns the name of the
        next node in the graph to transition to.
        Updates state.intent, state.previous_intent, state.loop_counter, and state.input_is_spec.
        Returns the name of the next node.
        """
        user_input = state.user_input
        previous_intent = state.previous_intent
        loop_counter = state.loop_counter

        # Initialize determined_intent to None at the start
        determined_intent: Optional[str] = None # <-- INITIALIZED HERE

        # Reset the input_is_spec flag at the start of routing
        state.input_is_spec = False
        # Clear extracted params from previous turn
        state.extracted_params = None


        if not user_input:
            logger.warning("Router received empty user input. Routing to handle_unknown.")
            determined_intent = "handle_unknown"
            state.update_scratchpad_reason("router", "Empty input, routing to handle_unknown.")
            # Loop detection will be applied below

        # --- Prioritize routing based on existing state ---
        # If a schema is already loaded, check if the user is asking a question about it
        elif state.openapi_schema:
            logger.debug("Schema exists. Checking if input is a query about the loaded spec/artifacts.")
            state.update_scratchpad_reason("router", "Schema exists. Checking for query intent.")

            # Use LLM specifically to classify if the input is a question about the loaded state
            query_classification_prompt = f"""
            An OpenAPI specification has already been loaded and processed.
            The user's input is: "{user_input}"

            Is this input a question about the loaded OpenAPI specification, the identified APIs, the generated payload descriptions, the execution graph description, or the execution plan description?
            Answer ONLY with "YES" or "NO".

            Current State Summary:
            - OpenAPI spec loaded: Yes
            - Schema Summary: {state.schema_summary[:500] + '...' if state.schema_summary else 'None'}
            - Identified APIs: {'Yes' if state.identified_apis else 'No'} ({len(state.identified_apis) if state.identified_apis else 0} found)
            - Execution graph description exists: {'Yes' if state.execution_graph else 'No'}
            - Execution plan description exists: {'Yes' if state.execution_plan else 'No'} ({len(state.execution_plan) if state.execution_plan else 0} steps)

            Answer (YES/NO):
            """
            try:
                llm_response = llm_call_helper(self.router_llm, query_classification_prompt)
                classification = llm_response.strip().upper()

                if classification == "YES":
                    logger.info("Router LLM classified input as a query about the loaded state. Routing to answer_openapi_query.")
                    determined_intent = "answer_openapi_query"
                    state.update_scratchpad_reason("router", "LLM classified input as query about loaded state. Routing to answer_openapi_query.")
                else:
                    logger.debug("Router LLM classified input NOT as a query about the loaded state. Proceeding to general intent determination.")
                    # If not a query about the state, proceed to the general intent determination below
                    # determined_intent remains None, which triggers the next block

            except Exception as e:
                logger.error(f"Error calling Router LLM for query classification: {e}", exc_info=True)
                # If classification fails, fall back to general intent determination
                logger.warning("Router LLM query classification failed. Proceeding to general intent determination.")
                # determined_intent remains None
                state.update_scratchpad_reason("router", f"LLM query classification failed: {e}. Proceeding to general intent determination.")


        # --- General Intent Determination (if not already routed by state or is a new spec) ---
        # This block runs if determined_intent is still None (meaning it wasn't a query about loaded state)
        # or if there was no schema loaded initially.
        if determined_intent is None: # <-- Check if intent is still None
            # --- Heuristic Check for NEW OpenAPI Spec ---
            # Only check for spec heuristic if no schema is loaded OR if the input is very long
            # (indicating it might be a new spec even if one is loaded)
            # Adding a length check prevents short questions from accidentally triggering the heuristic
            if not state.openapi_schema or len(user_input) > 200: # Example length threshold (adjust as needed)
                 if self.SPEC_START_REGEX.search(user_input):
                      logger.info("Router heuristic detected potential NEW OpenAPI spec input.")
                      state.update_scratchpad_reason("router", "Heuristic detected potential new spec. Routing to parse_openapi_spec.")
                      determined_intent = "parse_openapi_spec"
                      state.input_is_spec = True # Set the flag for the parser node
                 else:
                      logger.debug("Input not detected as spec by heuristic or schema exists and input is short. Using LLM for general intent.")
                      # Fall through to LLM intent determination if not a spec by heuristic or short input with existing schema
                      # determined_intent remains None, which triggers the LLM call below
            else:
                 # Schema exists and input is short and not classified as a query about state.
                 # This is likely a command or a different type of request. Use LLM for general intent.
                 logger.debug("Schema exists, input is short, and not classified as query. Using LLM for general intent.")
                 # determined_intent remains None


            # --- Call LLM for General Intent (if determined_intent is still None) ---
            if determined_intent is None: # <-- Check if intent is still None before calling LLM
                prompt = f"""
                The user's input is: "{user_input}"

                Determine the user's high-level intent from the following list, considering the current state: {self.AVAILABLE_INTENTS}.
                Do NOT choose 'parse_openapi_spec' unless the user is explicitly providing new spec text that clearly looks like a spec.
                Prioritize intents related to analyzing or describing the loaded OpenAPI spec and generated artifacts if they exist.

                Current State Summary:
                - OpenAPI spec loaded: {'Yes' if state.openapi_schema else 'No'}
                - Schema Summary: {state.schema_summary[:500] + '...' if state.schema_summary else 'None'}
                - Identified APIs: {'Yes' if state.identified_apis else 'No'} ({len(state.identified_apis) if state.identified_apis else 0} found)
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
                    # Prevent LLM from choosing parse_openapi_spec if heuristic didn't catch it and schema exists
                    # or if the input is short. The heuristic is the primary way to detect new specs.
                    elif determined_intent == "parse_openapi_spec" and (state.openapi_schema or len(user_input) < 200):
                         logger.warning(f"Router LLM chose '{determined_intent}' unexpectedly (schema exists or input is short). Overriding to handle_unknown.")
                         determined_intent = "handle_unknown"
                         state.update_scratchpad_reason("router", f"LLM chose '{determined_intent}' unexpectedly. Defaulted to '{determined_intent}'.")
                    else:
                        logger.debug(f"Router LLM determined general intent: {determined_intent}")
                        state.update_scratchpad_reason("router", f"LLM determined general intent: '{determined_intent}'.")

                except Exception as e:
                    logger.error(f"Error calling Router LLM for general intent: {e}", exc_info=True)
                    # If LLM call fails, default to handle_unknown
                    determined_intent = "handle_unknown"
                    state.update_scratchpad_reason("router", f"LLM call failed: {e}. Defaulted to '{determined_intent}'.")
                    # Set an intermediate response here if desired, although handle_unknown node will generate one
                    # state.response = "An internal error occurred while trying to understand your request. Please try again."

            # If determined_intent is still None after all checks, it means something went wrong.
            if determined_intent is None: # <-- Final fallback check
                 logger.error("Router failed to determine intent after all checks. Defaulting to handle_unknown.")
                 determined_intent = "handle_unknown"
                 state.update_scratchpad_reason("router", "Failed to determine intent after all checks. Defaulted to handle_unknown.")


        # --- Apply Loop Detection based on determined_intent ---
        # Only apply loop detection if the intent is not handle_unknown or handle_loop itself
        # and not parse_openapi_spec (which might be a valid repeat if user provides multiple specs)
        if determined_intent == previous_intent and determined_intent not in ["handle_unknown", "handle_loop", "parse_openapi_spec", "responder"]:
             # Increment loop counter if the same non-final, non-parse intent repeats
             loop_counter += 1
             logger.warning(f"Router detected repeated intent: {determined_intent}. Loop counter: {loop_counter}")
             if loop_counter >= 3: # Threshold for loop detection
                 logger.error(f"Router detected potential loop (intent '{determined_intent}' repeated {loop_counter} times). Routing to handle_loop.")
                 final_intent = "handle_loop"
                 # Reset counter after routing to handle_loop
                 state.loop_counter = 0
             else:
                 # Update counter but proceed with the determined intent for now
                 state.loop_counter = loop_counter
                 final_intent = determined_intent
        else:
            # Reset loop counter if intent changes or is a final/parse state
            state.loop_counter = 0
            final_intent = determined_intent
            # logger.debug("Router: Intent changed or is final/parse state. Resetting loop counter.")


        # Update state for the next turn and for the graph conditional edge
        state.previous_intent = final_intent # Store the final decision
        state.intent = final_intent

        logger.info(f"Router routing to: {final_intent}")

        # Return the name of the next node
        return final_intent
