# filename: core_logic.py
import json
import logging
from typing import Any, Dict, List, Optional

# Assuming models.py defines BotState, GraphOutput, Node, Edge, InputMapping etc.
from models import (
    BotState, GraphOutput, Node, Edge, AddEdgeParams,
    GeneratePayloadsParams, GenerateGraphParams, InputMapping
)
# Assuming utils.py defines helpers
from utils import (
    llm_call_helper, load_cached_schema, save_schema_to_cache,
    get_cache_key, check_for_cycles, parse_llm_json_output
)

# Module-level logger
logger = logging.getLogger(__name__)

class OpenAPICoreLogic:
    """
    Handles the core tasks of parsing OpenAPI specs, generating payload
    descriptions, creating execution graph descriptions, and managing
    graph descriptions using a worker LLM.
    These methods are designed to be used as nodes in the LangGraph.
    Actual API execution logic is NOT included.
    """
    def __init__(self, worker_llm: Any):
        """
        Initializes the core logic component.

        Args:
            worker_llm: The language model instance dedicated to performing tasks.
        """
        if not hasattr(worker_llm, 'invoke'):
             raise TypeError("worker_llm must have an 'invoke' method.")
        self.worker_llm = worker_llm
        logger.info("OpenAPICoreLogic initialized (without execution).")

    def _generate_llm_schema_summary(self, schema: Dict[str, Any], max_length: int = 8000) -> str:
        """
        Generates a comprehensive, LLM-friendly text summary of the OpenAPI schema
        by using the worker LLM to process the full schema JSON.
        Prioritizes key information like paths, operations, summaries, parameters,
        request bodies, and responses.
        """
        if not schema:
            return "No OpenAPI schema available."

        logger.debug("Generating LLM-based schema summary.")

        # Convert the full schema dictionary to a JSON string for the LLM.
        full_schema_json = json.dumps(schema, indent=2)
        # Truncate the input JSON if it's excessively large.
        truncated_input_json = full_schema_json[:15000] # Example limit
        if len(full_schema_json) > 15000:
             truncated_input_json += "\n... (OpenAPI schema JSON truncated for prompt)"


        prompt = f"""
        Analyze the following OpenAPI schema provided as a JSON object.
        Generate a detailed, human-readable text summary that highlights the key aspects relevant for understanding and using the API.
        Focus on:
        - The API Title and Version.
        - A list of all available Paths and the HTTP Methods for each path.
        - For each operation (method on a path), include its `operationId`, `summary`, and a brief mention of its parameters (e.g., path, query, request body) and response codes.
        - Briefly describe the structure of significant request bodies and responses if possible and concise, noting key field names.
        - Do NOT include the full JSON schema in your output summary.
        - Format the summary clearly with headings and bullet points.

        OpenAPI Schema JSON:
        ```json
        {truncated_input_json}
        ```

        Detailed OpenAPI Schema Summary:
        """

        try:
            # Use the worker_llm to generate the summary
            llm_response = llm_call_helper(self.worker_llm, prompt)
            llm_generated_summary = llm_response.strip()

            logger.debug(f"LLM generated schema summary (length: {len(llm_generated_summary)}).")

            # Apply truncation to the LLM's output as a safeguard
            if len(llm_generated_summary) > max_length:
                logger.warning(f"LLM-generated schema summary truncated to {max_length} characters.")
                truncated_summary = llm_generated_summary[:max_length]
                last_newline = truncated_summary.rfind('\n')
                if last_newline != -1:
                     truncated_summary = truncated_summary[:last_newline] + "\n... (summary truncated)"
                else:
                     truncated_summary = truncated_summary + "... (summary truncated)"
                return truncated_summary
            return llm_generated_summary

        except Exception as e:
            logger.error(f"Error calling LLM for schema summary: {e}", exc_info=True)
            # Fallback to a basic programmatic summary if LLM call fails
            logger.warning("Falling back to basic programmatic schema summary due to LLM error.")
            fallback_summary = ["Error generating LLM summary. Basic schema info:"]
            info = schema.get('info', {})
            fallback_summary.append(f"API Title: {info.get('title', 'Untitled API')}")
            fallback_summary.append(f"Version: {info.get('version', 'N/A')}")
            paths = schema.get('paths', {})
            if paths:
                 fallback_summary.append(f"Paths defined: {len(paths)}")
                 # Add first few paths as example
                 for i, (path, methods) in enumerate(paths.items()):
                     if i >= 5: break # Limit fallback detail
                     fallback_summary.append(f"- {path}: {', '.join(methods.keys()).upper()}")
            else:
                 fallback_summary.append("No paths defined.")

            fallback_text = "\n".join(fallback_summary)
            # Truncate fallback summary if needed
            if len(fallback_text) > max_length:
                 return fallback_text[:max_length] + "\n... (fallback summary truncated)"
            return fallback_text

    # --- Tool Methods (Designed as Graph Nodes) ---

    def parse_openapi_spec(self, state: BotState) -> BotState:
        """
        Parses the raw OpenAPI spec text provided in the state.
        Relies on state.input_is_spec flag set by the router to confirm input is likely a spec.
        Uses caching to avoid re-parsing identical specs.
        Relies on the LLM to perform the parsing/resolution.
        Updates state.openapi_schema, state.schema_summary, and state.response.
        Clears the input_is_spec flag after processing.
        """
        tool_name = "parse_openapi_spec"
        state.update_scratchpad_reason(tool_name, "Starting OpenAPI spec parsing.")
        logger.debug("Executing parse_openapi_spec node.")

        # Check if the router flagged the input as a spec
        if not state.input_is_spec:
            state.response = "Error: The previous input was not identified as an OpenAPI specification. Please provide the spec text."
            state.update_scratchpad_reason(tool_name, "Failed: Input not flagged as spec by router.")
            logger.error("parse_openapi_spec called but input_is_spec flag is false.")
            # Clear potentially stale spec info
            state.openapi_spec_text = None
            state.openapi_schema = None
            state.schema_summary = None
            state.schema_cache_key = None
            # Reset flag (though it was already false)
            state.input_is_spec = False
            return state

        # Get the spec text from user_input (router should have placed it there)
        spec_text_to_parse = state.user_input

        if not spec_text_to_parse:
            state.response = "Error: No OpenAPI specification text found in the user input to parse, even though it was flagged as a spec."
            state.update_scratchpad_reason(tool_name, "Failed: No spec text found in user_input despite flag.")
            logger.error("parse_openapi_spec: input_is_spec is true but user_input is empty.")
            state.openapi_spec_text = None
            state.openapi_schema = None
            state.schema_summary = None
            state.schema_cache_key = None
            state.input_is_spec = False # Reset flag
            return state

        # Store the raw spec text in the dedicated field
        state.openapi_spec_text = spec_text_to_parse
        logger.debug("Using spec text from state.user_input based on input_is_spec flag.")

        # Reset the flag now that we've consumed the input
        state.input_is_spec = False

        # Generate cache key based on the raw text
        cache_key = get_cache_key(spec_text_to_parse)
        state.schema_cache_key = cache_key

        # Try loading from cache first
        cached_schema = load_cached_schema(cache_key)
        if cached_schema:
            state.openapi_schema = cached_schema
            # If schema is loaded from cache, generate or retrieve the summary as well
            # For simplicity, we'll regenerate the summary from the cached schema.
            state.schema_summary = self._generate_llm_schema_summary(state.openapi_schema)
            state.response = "Successfully loaded parsed OpenAPI schema from cache and generated summary."
            state.update_scratchpad_reason(tool_name, f"Loaded schema from cache (key: {cache_key}). Generated summary.")
            logger.info("Loaded OpenAPI schema from cache and generated summary.")
            return state

        # If not cached, use LLM to parse
        state.update_scratchpad_reason(tool_name, "Schema not found in cache. Using LLM to parse.")
        logger.info("Parsing OpenAPI schema using LLM.")
        prompt = f"""
        Parse the following OpenAPI specification text (which can be in YAML or JSON format) into a fully resolved JSON object.
        Ensure all internal `$ref` links are resolved if possible, embedding the referenced schema objects directly.
        Handle external references (`$ref` to URLs or local files) by including a note or placeholder if full resolution is not possible, but prioritize resolving internal references.
        Output ONLY the resulting JSON object, without any surrounding text, markdown formatting, or conversational remarks.

        OpenAPI Specification Text:
        ```
        {spec_text_to_parse[:15000]}
        ```
        { "... (Input specification truncated)" if len(spec_text_to_parse) > 15000 else "" }

        Parsed and Resolved JSON Object:
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            parsed_schema = parse_llm_json_output(llm_response) # Basic JSON parsing

            if parsed_schema and isinstance(parsed_schema, dict):
                state.openapi_schema = parsed_schema
                # Successfully parsed, now generate the summary using the LLM
                state.schema_summary = self._generate_llm_schema_summary(state.openapi_schema)
                state.response = "Successfully parsed OpenAPI specification and generated summary."
                state.update_scratchpad_reason(tool_name, f"LLM parsed schema and generated summary. Keys: {list(parsed_schema.keys())}")
                logger.info("LLM successfully parsed OpenAPI schema and generated summary.")
                # Save the newly parsed schema to cache
                save_schema_to_cache(cache_key, parsed_schema)
            else:
                state.response = "Error: LLM did not return a valid JSON object for the OpenAPI schema. Please check the format of your specification."
                state.update_scratchpad_reason(tool_name, f"LLM parsing failed. Raw response: {llm_response[:500]}...")
                logger.error(f"LLM failed to parse OpenAPI spec into valid JSON. Response: {llm_response[:500]}")
                state.openapi_schema = None
                state.schema_summary = None # Ensure summary is cleared on parsing failure
                state.schema_cache_key = None # Clear cache key as parsing failed

        except Exception as e:
            state.response = f"Error during OpenAPI parsing LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for OpenAPI parsing: {e}", exc_info=True)
            state.openapi_schema = None
            state.schema_summary = None # Ensure summary is cleared on error
            state.schema_cache_key = None # Clear cache key on error

        return state


    def plan_execution(self, state: BotState) -> BotState:
        """
        Planner node: break the user's high-level goal and current state
        into an ordered list of tool-names (or operationIds) describing a plan.
        This node does NOT execute the plan, only generates its description.
        Sets state.execution_plan.
        """
        tool_name = "plan_execution"
        state.update_scratchpad_reason(tool_name, "Starting execution planning (description only).")
        logger.debug("Executing plan_execution node.")

        if not state.openapi_schema:
            state.response = "Error: Cannot plan execution without a parsed OpenAPI schema."
            state.update_scratchpad_reason(tool_name, "Failed: Schema missing for planning.")
            logger.error("plan_execution called without openapi_schema.")
            state.execution_plan = []
            state.current_plan_step = 0
            return state

        schema_summary = state.schema_summary
        user_goal = state.user_input or "Generate a typical workflow description." # Default goal for planning

        prompt = f"""
        Analyze the user's goal and the available API operations from the OpenAPI schema summary to create an ordered list of API operationIds that *would* be executed to achieve the goal.
        This is a planning step; you are describing the sequence, not executing it.

        Current State Summary:
        - OpenAPI spec loaded: Yes
        - Schema Summary: {schema_summary[:1000] + '...' if schema_summary else 'None'}
        - Identified APIs: {'Yes' if state.identified_apis else 'No'} ({len(state.identified_apis) if state.identified_apis else 0} found)
        - User Goal: "{user_goal}"

        Instructions:
        1. Identify the most relevant API operations based on the user's goal and the schema summary.
        2. Determine a logical sequence of `operationId`s to achieve the goal.
        3. If the user's goal implies creating, reading, updating, or deleting resources, plan the standard CRUD sequence description if applicable (e.g., createItem -> getItem -> updateItem -> deleteItem).
        4. Consider dependencies: if one operation produces data (like an ID) needed by another, place the producer before the consumer in the sequence description.
        5. Output ONLY a JSON list of strings, where each string is an `operationId` from the OpenAPI schema summary.

        Example: If the user says "describe a plan to create a new user and get their details", and the schema has operationIds "createUser" and "getUser", the plan description might be `["createUser", "getUser"]`.

        Based on the user goal and state, generate the sequence of operationIds (JSON list):
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            steps = parse_llm_json_output(llm_response)

            # Basic validation: ensure steps is a list of strings
            if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
                 # Further validation: Check if operationIds are valid
                 valid_plan = True
                 valid_op_ids = {op['operationId'] for path, methods in state.openapi_schema.get('paths', {}).items() for method, op in methods.items() if isinstance(op, dict) and 'operationId' in op}
                 validated_steps = [step for step in steps if step in valid_op_ids]
                 if len(validated_steps) != len(steps):
                      invalid_steps = set(steps) - valid_op_ids
                      logger.warning(f"Planner generated invalid operationIds: {list(invalid_steps)}. Filtering plan description.")
                      steps = validated_steps # Filter out invalid steps
                      if not steps: valid_plan = False # If filtering results in empty plan

                 if valid_plan and steps: # Ensure plan is not empty after validation
                      state.execution_plan = steps
                      state.current_plan_step = 0 # Reset step counter for a new plan description
                      state.response = f"Based on your goal, I've described a potential plan with these steps: {state.execution_plan}"
                      state.update_scratchpad_reason(tool_name, f"LLM generated plan description: {state.execution_plan}")
                      logger.info(f"Generated execution plan (description): {state.execution_plan}")
                 else:
                      state.response = "Could not generate a valid execution plan description based on the available APIs or the plan was empty after validation."
                      state.update_scratchpad_reason(tool_name, f"LLM generated invalid or empty plan description. Original steps: {steps}, Validated: {validated_steps}")
                      logger.error(f"LLM generated invalid or empty plan description. Raw LLM response: {llm_response}")
                      state.execution_plan = [] # Clear plan on failure
                      state.current_plan_step = 0

            else:
                 # Fallback if LLM output is invalid format
                 state.response = "Could not generate a valid plan description. The LLM did not return a JSON list of strings."
                 state.update_scratchpad_reason(tool_name, f"LLM returned invalid plan format: {llm_response[:500]}...")
                 logger.error(f"LLM returned invalid plan format: {llm_response}")
                 state.execution_plan = [] # Clear plan on failure
                 state.current_plan_step = 0

        except Exception as e:
            state.response = f"Error during planning description LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for planning description: {e}", exc_info=True)
            state.execution_plan = [] # Clear plan on error
            state.current_plan_step = 0

        return state

    def identify_apis(self, state: BotState) -> BotState:
        """
        Identifies relevant API endpoints from the parsed schema based on user goal or general analysis.
        Uses the worker LLM. Updates state.identified_apis and state.response.
        """
        tool_name = "identify_apis"
        state.update_scratchpad_reason(tool_name, "Starting API identification.")
        logger.debug("Executing identify_apis node.")

        if not state.openapi_schema:
            state.response = "Error: Cannot identify APIs without a parsed OpenAPI schema. Please provide/parse a spec first."
            state.update_scratchpad_reason(tool_name, "Failed: Schema missing.")
            logger.error("identify_apis called without openapi_schema.")
            return state
        if not state.schema_summary:
             state.response = "Error: Cannot identify APIs without a schema summary. Please parse the spec again."
             state.update_scratchpad_reason(tool_name, "Failed: Schema summary missing.")
             logger.error("identify_apis called without schema_summary.")
             return state

        schema_summary = state.schema_summary
        # Use graph_generation_instructions or user_input as context for identification
        user_context = state.graph_generation_instructions or state.user_input or "general analysis of the API"

        prompt = f"""
        Analyze the following detailed OpenAPI schema summary and the user's goal/context to identify the key API endpoints (operations) that are most relevant.
        User Goal/Context: "{user_context}"
        For each identified API, extract its 'operationId', 'summary', HTTP 'method', and 'path'.
        Output ONLY a JSON list of objects, where each object represents an identified API endpoint.
        Example format: `[ {{"operationId": "getUser", "summary": "Get user details", "method": "get", "path": "/users/{{userId}}"}}, ... ]`

        Detailed OpenAPI Schema Summary:
 
