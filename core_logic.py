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
    get_cache_key, check_for_cycles, parse_llm_json_output_with_model
)

# Module-level logger
logger = logging.getLogger(__name__)

class OpenAPICoreLogic:
    # ... (keep __init__, _generate_llm_schema_summary as is)
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
        # Set an intermediate response message
        state.response = "Parsing OpenAPI specification..."


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
            # Update response message
            state.response = "Successfully loaded parsed OpenAPI schema from cache and generated summary."
            state.update_scratchpad_reason(tool_name, f"Loaded schema from cache (key: {cache_key}). Generated summary.")
            logger.info("Loaded OpenAPI schema from cache and generated summary.")
            return state

        # If not cached, use LLM to parse
        state.update_scratchpad_reason(tool_name, "Schema not found in cache. Using LLM to parse.")
        logger.info("Parsing OpenAPI schema using LLM.")
        # Update response message
        state.response = "Schema not found in cache. Using LLM to parse..."

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
            # Use parse_llm_json_output without model for initial schema parsing
            parsed_schema = parse_llm_json_output_with_model(llm_response) # Basic JSON parsing

            if parsed_schema and isinstance(parsed_schema, dict):
                state.openapi_schema = parsed_schema
                # Successfully parsed, now generate the summary using the LLM
                state.schema_summary = self._generate_llm_schema_summary(state.openapi_schema)
                # Update response message
                state.response = "Successfully parsed OpenAPI specification and generated summary."
                state.update_scratchpad_reason(tool_name, f"LLM parsed schema and generated summary. Keys: {list(parsed_schema.keys())}")
                logger.info("LLM successfully parsed OpenAPI schema and generated summary.")
                # Save the newly parsed schema to cache
                save_schema_to_cache(cache_key, parsed_schema)
            else:
                # Update response message
                state.response = "Error: LLM did not return a valid JSON object for the OpenAPI schema. Please check the format of your specification."
                state.update_scratchpad_reason(tool_name, f"LLM parsing failed. Raw response: {llm_response[:500]}...")
                logger.error(f"LLM failed to parse OpenAPI spec into valid JSON. Response: {llm_response[:500]}")
                state.openapi_schema = None
                state.schema_summary = None # Ensure summary is cleared on parsing failure
                state.schema_cache_key = None # Clear cache key as parsing failed

        except Exception as e:
            # Update response message
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
        state.response = "Generating execution plan description..."


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
            # Use parse_llm_json_output without model for list of strings
            steps = parse_llm_json_output_with_model(llm_response)

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
        state.response = "Identifying relevant APIs..."


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
        ```
        {schema_summary}
        ```

        Identified APIs (JSON List):
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            # Use parse_llm_json_output without model for list of dicts
            identified_apis = parse_llm_json_output_with_model(llm_response) # Expecting a list

            if identified_apis and isinstance(identified_apis, list):
                # Basic validation of list items (check for required keys)
                valid_apis = []
                seen_op_ids = set()
                for api_info in identified_apis:
                     if isinstance(api_info, dict) and all(key in api_info for key in ['operationId', 'method', 'path']):
                          op_id = api_info['operationId']
                          if op_id not in seen_op_ids:
                              valid_apis.append(api_info)
                              seen_op_ids.add(op_id)
                          else:
                              logger.warning(f"LLM returned duplicate operationId '{op_id}' in identified APIs. Skipping duplicate.")
                     else:
                          logger.warning(f"LLM returned invalid API info format: {api_info}")

                state.identified_apis = valid_apis
                state.response = f"Identified {len(valid_apis)} potentially relevant API endpoints based on the context."
                state.update_scratchpad_reason(tool_name, f"LLM identified {len(valid_apis)} APIs. First few: {valid_apis[:3]}")
                logger.info(f"Identified APIs: {valid_apis}")
            else:
                state.response = "Error: LLM did not return a valid JSON list of identified APIs."
                state.update_scratchpad_reason(tool_name, f"LLM API identification failed. Raw response: {llm_response[:500]}...")
                logger.error(f"LLM failed to identify APIs in valid JSON list format. Response: {llm_response[:500]}")
                state.identified_apis = None # Clear on failure

        except Exception as e:
            state.response = f"Error during API identification LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for API identification: {e}", exc_info=True)
            state.identified_apis = None # Clear on error

        return state


    def generate_payloads(self, state: BotState) -> BotState:
        """
        Generates example payload descriptions for identified API operations using the LLM.
        Considers user instructions if provided via state.payload_generation_instructions
        or state.extracted_params. Updates state.payload_descriptions and state.response.
        These are NOT actual payloads for execution, just descriptions/examples.
        """
        tool_name = "generate_payloads"
        state.update_scratchpad_reason(tool_name, "Starting payload generation (description only).")
        logger.debug("Executing generate_payloads node (description only).")
        state.response = "Generating example payload descriptions..."


        if not state.openapi_schema:
            state.response = "Error: Cannot generate payload descriptions without a parsed OpenAPI schema."
            state.update_scratchpad_reason(tool_name, "Failed: Schema missing.")
            logger.error("generate_payloads called without openapi_schema.")
            return state
        if not state.schema_summary:
             state.response = "Error: Cannot generate payload descriptions without a schema summary. Please parse the spec again."
             state.update_scratchpad_reason(tool_name, "Failed: Schema summary missing.")
             logger.error("generate_payloads called without schema_summary.")
             return state

        apis_to_process = state.identified_apis
        # Check for specific instructions or target APIs from parameters (extracted by router or planner)
        params: Optional[GeneratePayloadsParams] = None
        instructions = state.payload_generation_instructions or "Describe typical example values."# Use instructions captured earlier or a default
        target_apis_filter: Optional[List[str]] = None

        if state.extracted_params:
            try:
                params = GeneratePayloadsParams.model_validate(state.extracted_params)
                if params.instructions:
                     instructions = params.instructions # Override if new instructions provided
                if params.target_apis:
                     target_apis_filter = params.target_apis
                     logger.info(f"Targeting payload description for specific APIs from params: {target_apis_filter}")
            except Exception as e: # Catch Pydantic validation errors etc.
                logger.warning(f"Could not parse GeneratePayloadsParams from extracted_params: {e}. Using state fields/defaults.")
                # Fallback to instructions potentially set by planner or default
                instructions = state.payload_generation_instructions or state.extracted_params.get("instructions", instructions)

        # Apply target_apis_filter if specified
        if target_apis_filter and apis_to_process:
             filtered_apis = [api for api in apis_to_process if api.get('operationId') in target_apis_filter]
             if not filtered_apis:
                  state.response = f"Warning: Could not find details for target APIs: {target_apis_filter} among identified APIs. Generating descriptions for all identified instead."
                  logger.warning(f"Target APIs not found among identified: {target_apis_filter}")
                  # Proceed with all identified APIs if filter yields nothing
             else:
                  apis_to_process = filtered_apis
                  logger.debug(f"Processing filtered APIs for payload descriptions: {apis_to_process}")

        if not apis_to_process:
             state.response = "Error: No relevant API operations found or specified to generate payload descriptions for. Try identifying APIs first."
             state.update_scratchpad_reason(tool_name, "Failed: No APIs to process for payload descriptions.")
             logger.error("generate_payloads called with no APIs to process.")
             state.payload_descriptions = None # Ensure cleared
             return state

        schema_summary = state.schema_summary
        api_list_str = json.dumps(apis_to_process, indent=2)
        # Use the renamed state field
        payloads_desc_str = json.dumps(state.payload_descriptions or "No payload descriptions generated yet", indent=2)

        prompt = f"""
        Based on the detailed OpenAPI schema summary and the list of target API operations below, describe example request payloads.
        Follow these instructions: {instructions}
        For each API operation in the list, determine its required parameters (path, query, header, body) from the schema summary.
        Describe a plausible JSON payload for the request body if applicable (e.g., for POST, PUT, PATCH). Use realistic example values in your description (e.g., "name": "Example User", "email": "user@example.com").
        Output ONLY a single JSON object where keys are the 'operationId' from the input list, and values are the *description* of the example payloads (e.g., "JSON object with fields 'name' (string, e.g., 'Example User'), 'email' (string, e.g., 'user@example.com')").

        Detailed OpenAPI Schema Summary:
        ```
        {schema_summary}
        ```

        Target API Operations:
        ```json
        {api_list_str}
        ```

        Instructions for Payload Description: {instructions}

        Described Payloads (JSON Object mapping operationId to payload description string):
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            # Use parse_llm_json_output without model for dict
            described_payloads = parse_llm_json_output_with_model(llm_response) # Expecting a dict

            if described_payloads and isinstance(described_payloads, dict):
                # Store the descriptions in the renamed state field
                state.payload_descriptions = described_payloads
                state.response = f"Generated descriptions for example payloads for {len(described_payloads)} API operations."
                state.update_scratchpad_reason(tool_name, f"LLM generated payload descriptions for keys: {list(described_payloads.keys())}")
                logger.info(f"Generated Payload Descriptions: {described_payloads}")
            else:
                state.response = "Error: LLM did not return a valid JSON object mapping operationIds to payload descriptions."
                state.update_scratchpad_reason(tool_name, f"LLM payload description failed. Raw response: {llm_response[:500]}...")
                logger.error(f"LLM failed to generate payload descriptions in valid JSON object format. Response: {llm_response[:500]}")
                state.payload_descriptions = None # Clear on failure

        except Exception as e:
            state.response = f"Error during payload description LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for payload description: {e}", exc_info=True)
            state.payload_descriptions = None # Clear on error

        return state

    def generate_execution_graph(self, state: BotState) -> BotState:
        """
        Generates a directed acyclic graph (DAG) describing the execution flow
        of identified APIs, considering dependencies and user goals/instructions.
        Updates state.execution_graph (description) and state.response.
        Includes explicit input mapping descriptions in the graph nodes.
        This node does NOT prepare for actual execution.
        """
        tool_name = "generate_execution_graph"
        state.update_scratchpad_reason(tool_name, "Starting execution graph generation (description only).")
        logger.debug("Executing generate_execution_graph node (description only).")
        state.response = "Generating execution graph description..."


        if not state.openapi_schema:
            state.response = "Error: Cannot generate execution graph description without a parsed OpenAPI schema."
            state.update_scratchpad_reason(tool_name, "Failed: Schema missing.")
            logger.error("generate_execution_graph called without openapi_schema.")
            return state
        if not state.schema_summary:
             state.response = "Error: Cannot generate execution graph description without a schema summary. Please parse the spec again."
             state.update_scratchpad_reason(tool_name, "Failed: Schema summary missing.")
             logger.error("generate_execution_graph called without schema_summary.")
             return state
        if not state.identified_apis:
            state.response = "Warning: No specific APIs identified. Attempting to generate a graph description based on the full schema summary and goal, but it might be less focused. Try identifying APIs first."
            state.update_scratchpad_reason(tool_name, "Warning: No APIs identified. Proceeding with full schema context.")
            logger.warning("generate_execution_graph called without identified_apis. Using full schema context.")
            # Allow proceeding, but the result might be less useful


        # Check for specific instructions or goals from parameters (extracted by router or planner)
        params: Optional[GenerateGraphParams] = None
        user_goal = state.user_input or "Describe a logical execution flow based on the API." # Default goal
        graph_instructions = "Describe typical CRUD dependencies (Create->Read->Update->Read->Delete) and data flow (e.g., an ID created in one step is used in the next)." # Default instructions

        # Use graph_generation_instructions if already set by planner, otherwise check extracted_params
        if state.graph_generation_instructions:
             logger.debug("Using graph generation instructions from state.")
             # Assume instructions already incorporate the goal
             combined_instructions = state.graph_generation_instructions
        elif state.extracted_params:
            try:
                params = GenerateGraphParams.model_validate(state.extracted_params)
                if params.goal: user_goal = params.goal
                if params.instructions: graph_instructions = params.instructions
                combined_instructions = f"Goal: {user_goal}\nInstructions: {graph_instructions}"
                state.graph_generation_instructions = combined_instructions # Store combined
                logger.debug(f"Graph description params from extracted_params: Goal='{user_goal}', Instructions='{graph_instructions}'")
            except Exception as e:
                logger.warning(f"Could not parse GenerateGraphParams from extracted_params: {e}. Using default behavior.")
                user_goal = state.extracted_params.get("goal", user_goal) or state.user_input
                graph_instructions = state.extracted_params.get("instructions", graph_instructions) or "Describe typical dependencies and data flow."
                combined_instructions = f"Goal: {user_goal}\nInstructions: {graph_instructions}"
                state.graph_generation_instructions = combined_instructions
        else:
            # Use defaults if nothing else provided
            combined_instructions = f"Goal: {user_goal}\nInstructions: {graph_instructions}"
            state.graph_generation_instructions = combined_instructions


        schema_summary = state.schema_summary
        # Use identified APIs if available, otherwise indicate all APIs are considered
        api_list_str = json.dumps(state.identified_apis, indent=2) if state.identified_apis else '"All APIs in schema summary"'
        # Referencing renamed state field payload_descriptions
        payloads_desc_str = json.dumps(state.payload_descriptions or "No payload descriptions generated yet", indent=2)


        prompt = f"""
        Task: Generate a description of an API execution workflow graph as a Directed Acyclic Graph (DAG).
        Crucially, for each node in the graph description, identify how its input parameters *would* be derived from the described results of preceding nodes and specify this using explicit input mappings *descriptions*.

        Context:
        1. User Goal & Instructions: {combined_instructions}
        2. Detailed OpenAPI Schema Summary:
            ```
            {schema_summary}
            ```
        3. Identified Relevant APIs (or 'All APIs in schema summary'):
            ```json
            {api_list_str}
            ```
        4. Described Example Payloads:
            ```json
            {payloads_desc_str}
            ```

        Instructions for Graph Description Generation:
        1. Analyze the relationships and potential dependencies between the API operations based on the schema, common workflow patterns (like CRUD), and the user's goal/instructions.
        2. Determine a logical sequence of API calls to achieve the goal. Think step-by-step about data flow (e.g., needing an ID from a 'create' response description for a subsequent 'get' or 'update' call description).
        3. Represent this workflow description as a DAG (no circular dependencies). Emphasize creating a valid DAG. If a potential cycle exists (e.g., repeatedly checking status), represent it logically as a sequence in the description or note the pattern in the graph description text, but DO NOT create a circular graph structure in the JSON output.
        4. Nodes: Each node should correspond to an API operation. Include its 'operationId', 'summary', and optionally the 'payload_description' if available and relevant.
           **IMPORTANT:** If an API operation is called multiple times in the workflow description, generate a unique `display_name` for each instance (e.g., "getUserDetails_afterCreate", "getUserDetails_afterUpdate") in addition to the `operationId`. Use the `display_name` as the unique identifier for the node in edges and input mappings if provided, otherwise use the `operationId`.
           For each node that would require input parameters derived from previous described steps, include an `input_mappings` list. Each item in this list should be an object with:
           - `source_operation_id`: The `operationId` OR `display_name` of the previous node whose described result *would* contain the required data. Use the unique identifier of the source node instance.
           - `source_data_path`: A JSONPath expression description (e.g., `$.id`, `$.items[0].name`, `$.data.token`) to describe how the specific data field *would* be extracted from the source node's *described result*.
           - `target_parameter_name`: The name of the parameter in the current node's API call description that this data maps to.
           - `target_parameter_in`: The location of the target parameter (`path`, `query`, `header`, `cookie`).
           - `transformation`: (Optional) A brief description if the data would need transformation (e.g., "convert to string", "format as date").

        5. Edges: Each edge should represent a dependency or sequential step description. Define `from_node` and `to_node` using the unique identifier of the source and target nodes (the `display_name` if present, otherwise the `operationId`). Add a brief `description` for the edge if the dependency reason is clear (e.g., "Data dependency: describes using ID from create response"). Ensure edges align with the `input_mappings` descriptions and represent a DAG.
        6. Provide a brief natural language `description` of the overall workflow described by the graph.

        Output Format:
        Output ONLY a single JSON object matching the GraphOutput structure (including `nodes`, `edges`, `description`). The content should be descriptions, not instructions for live execution. Ensure nodes use the `payload_description` field. Use the `display_name` field in nodes if multiple instances of the same operation are used. Reference nodes in edges and mappings using their `display_name` if provided, otherwise their `operationId`.
        ```json
        {{
          "nodes": [
            {{
              "operationId": "...",
              "display_name": "...", // Include if operationId is repeated
              "summary": "...",
              "payload_description": "...", // This should be the string description
              "input_mappings": [
                {{
                  "source_operation_id": "...", // operationId or display_name of source node
                  "source_data_path": "...",
                  "target_parameter_name": "...",
                  "target_parameter_in": "...",
                  "transformation": "..." // Optional
                }},
                ...
              ]
            }},
            ...
          ],
          "edges": [ {{ "from_node": "nodeId1", "to_node": "nodeId2", "description": "..." }}, ... ], // nodeId is display_name or operationId
          "description": "Overall workflow description..."
        }}
        ```
        Ensure all node identifiers used in nodes, edges, and input mappings are unique within the graph and correspond to a defined node.
        Ensure `source_operation_id` in mappings refers to a preceding node instance in the planned sequence description using its unique identifier.
        Ensure `target_parameter_name` and `target_parameter_in` match parameters expected by the target operation according to the schema summary.

        Generated Execution Graph Description (JSON Object):
        """

        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            # Use parse_llm_json_output with the expected model for validation
            # The model validation will now expect payload_description to be a string
            graph_output = parse_llm_json_output_with_model(llm_response, expected_model=GraphOutput)

            if graph_output and isinstance(graph_output, GraphOutput):
                # Validate graph structure and check for cycles using effective_id
                node_ids = {node.effective_id for node in graph_output.nodes} # Use effective_id
                valid_graph_structure = True
                error_details = []

                # Validate edges using effective_id
                for edge in graph_output.edges:
                     if edge.from_node not in node_ids or edge.to_node not in node_ids:
                          error_details.append(f"Edge references non-existent node: {edge.from_node} -> {edge.to_node}")
                          valid_graph_structure = False

                # Validate mappings (basic checks) using effective_id
                for node in graph_output.nodes:
                    for mapping in node.input_mappings:
                        if mapping.source_operation_id not in node_ids: # Check against effective_ids
                             error_details.append(f"Mapping in node '{node.effective_id}' references non-existent source node: '{mapping.source_operation_id}'")
                             valid_graph_structure = False
                        # More detailed validation could check target_parameter_name/in against schema here

                # Pass GraphOutput directly to check_for_cycles, which is updated to use effective_id
                is_acyclic, cycle_msg = check_for_cycles(graph_output)

                if not is_acyclic:
                    state.response = f"Error: LLM generated a graph description with cycles. {cycle_msg} Cannot accept cyclic graph description."
                    state.update_scratchpad_reason(tool_name, f"Graph description failed: Cycle detected. {cycle_msg}")
                    logger.error(f"LLM generated cyclic graph description: {cycle_msg}")
                    state.execution_graph = None
                elif not valid_graph_structure:
                    error_msg = f"Error: LLM generated a graph description with structural errors (invalid edges or mappings). Details: {'; '.join(error_details)}"
                    state.response = error_msg
                    state.update_scratchpad_reason(tool_name, f"Graph description failed: Structural errors. {error_details}")
                    logger.error(f"LLM generated graph description with structural errors: {error_details}")
                    state.execution_graph = None
                else:
                    state.execution_graph = graph_output
                    state.response = f"Successfully generated execution graph description with {len(graph_output.nodes)} nodes and {len(graph_output.edges)} dependencies."
                    if graph_output.description:
                         state.response += f" Workflow Description: {graph_output.description}"
                    state.update_scratchpad_reason(tool_name, f"LLM generated graph description. Nodes: {len(graph_output.nodes)}, Edges: {len(graph_output.edges)}. Desc: {graph_output.description}")
                    logger.info(f"Generated Graph Description: {graph_output.model_dump_json(indent=2)}")

            else:
                state.response = "Error: LLM did not return a valid JSON object matching the GraphOutput structure for the graph description."
                state.update_scratchpad_reason(tool_name, f"LLM graph generation failed. Raw response: {llm_response[:500]}...")
                logger.error(f"LLM failed to generate graph description in valid GraphOutput format. Response: {llm_response[:500]}")
                state.execution_graph = None

        except Exception as e:
            state.response = f"Error during graph generation LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for graph generation: {e}", exc_info=True)
            state.execution_graph = None

        return state

    # ... (keep other methods like describe_graph, get_graph_json, handle_unknown, handle_loop, answer_openapi_query as is,
    #      but potentially add state.response = "..." at the beginning of each if you want intermediate messages for those too)
    def describe_graph(self, state: BotState) -> BotState:
        """Generates a natural language description of the current execution graph description using the LLM (if not already described)."""
        tool_name = "describe_graph"
        state.update_scratchpad_reason(tool_name, "Starting graph description.")
        logger.debug("Executing describe_graph node.")
        state.response = "Describing the generated graph..." # Added intermediate message

        if not state.execution_graph:
            state.response = "Error: No execution graph description exists to describe. Try generating one first."
            state.update_scratchpad_reason(tool_name, "Failed: Graph missing.")
            logger.error("describe_graph called without execution_graph.")
            return state

        # Use existing description if available and seems reasonable
        if state.execution_graph.description and len(state.execution_graph.description) > 20:
             # Keep the intermediate message briefly before potentially overwriting with final desc
             original_response = state.response
             state.response = state.execution_graph.description
             state.update_scratchpad_reason(tool_name, "Used existing graph description.")
             logger.info("Using existing graph description.")
             # If you want *both* intermediate and final, append:
             # state.response = original_response + "\nFinal Description: " + state.execution_graph.description
             return state

        # Generate description if missing or too short
        graph_json = state.execution_graph.model_dump_json(indent=2)
        prompt = f"""
        Based on the following API execution graph description (JSON format), provide a concise, natural language description of the workflow it represents.
        Focus on the sequence of described actions and potential data flow implied by the nodes, edges, and input mappings descriptions.
        Emphasize that this is a *plan* or *description* of a workflow, not an executed result. Use the node `display_name` if available, otherwise the `operationId`, to refer to steps.

        Execution Graph Description JSON:
        ```json
        {graph_json}
        ```

        Workflow Description:
        """
        state.update_scratchpad_reason(tool_name, "Asking LLM to generate graph description.")
        logger.info("Asking LLM to generate graph description.")
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            description = llm_response.strip()
            state.response = description # Set the generated description as the final response
            state.execution_graph.description = description # Store the generated description
            state.update_scratchpad_reason(tool_name, f"LLM generated graph description (length {len(description)}).")
            logger.info(f"Generated graph description: {description}")

        except Exception as e:
            state.response = f"Error during graph description LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for graph description: {e}", exc_info=True)
            # Provide a basic fallback response if LLM fails
            state.response = f"Could not generate a detailed description via LLM, but the graph description contains {len(state.execution_graph.nodes)} steps and {len(state.execution_graph.edges)} dependencies."

        return state

    def get_graph_json(self, state: BotState) -> BotState:
        """Outputs the current execution graph description as a JSON string."""
        tool_name = "get_graph_json"
        state.update_scratchpad_reason(tool_name, "Starting get graph JSON.")
        logger.debug("Executing get_graph_json node.")
        state.response = "Formatting graph description as JSON..." # Added intermediate message


        if not state.execution_graph:
            state.response = "Error: No execution graph description exists to output. Try generating one first."
            state.update_scratchpad_reason(tool_name, "Failed: Graph missing.")
            logger.error("get_graph_json called without execution_graph.")
            return state

        try:
            # Output JSON with indentation for readability
            graph_json = state.execution_graph.model_dump_json(indent=2)
            # Prepend with markdown code block formatting for better display in some UIs
            state.response = f"```json\n{graph_json}\n```"
            state.update_scratchpad_reason(tool_name, "Outputted graph JSON.")
            logger.info("Outputted graph JSON.")
        except Exception as e:
            state.response = f"Error serializing graph description to JSON: {e}"
            state.update_scratchpad_reason(tool_name, f"JSON serialization error: {e}")
            logger.error(f"Error serializing graph description: {e}", exc_info=True)

        return state

    def handle_unknown(self, state: BotState) -> BotState:
        """Handles cases where the user's intent is unclear or not supported."""
        tool_name = "handle_unknown"
        state.update_scratchpad_reason(tool_name, f"Handling unknown intent for input: {state.user_input}")
        logger.debug("Executing handle_unknown node.")
        state.response = "Trying to understand your request..." # Added intermediate message


        prompt = f"""
        The user said: "{state.user_input}"
        My current state includes:
        - OpenAPI spec loaded: {'Yes' if state.openapi_schema else 'No'}
        - Execution graph description exists: {'Yes' if state.execution_graph else 'No'}
        - Execution graph description summary: {state.execution_graph.description if state.execution_graph and state.execution_graph.description else 'None generated'}
        - Execution plan description exists: {'Yes' if state.execution_plan else 'No'} ({len(state.execution_plan) if state.execution_plan else 0} steps)

        I couldn't determine a specific action (like 'parse spec', 'generate graph description', 'describe plan', etc.) from the user's input, or the current state is not ready for the requested action.
        Please formulate a polite response acknowledging the situation. Explain what I *can* do:
        - Parse an OpenAPI specification (JSON or YAML).
        - Summarize the API described in the spec.
        - Identify relevant API endpoints based on a goal.
        - Describe example payloads for APIs.
        - Describe potential API workflows as a graph.
        - Answer questions about the loaded spec or described artifacts.
        Emphasize that I **cannot actually execute API calls**. Suggest the user rephrase or ask for one of the supported actions.

        Response to user:
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            state.response = llm_response.strip()
            logger.info("LLM generated handle_unknown response.")
        except Exception as e:
             logger.error(f"Error calling LLM for unknown intent handling: {e}", exc_info=True)
             state.response = "I'm sorry, I didn't quite understand that request or I'm not ready to perform it in the current state. Could you please rephrase? I can help analyze OpenAPI specs, describe potential API workflows, and answer questions about the spec, but I cannot execute API calls."

        state.update_scratchpad_reason(tool_name, "Provided clarification response (no execution).")
        return state


    def handle_loop(self, state: BotState) -> BotState:
        """
        Called when the router detects the same intent repeating.
        Place a message into state to ask the user how to proceed.
        """
        tool_name = "handle_loop"
        state.update_scratchpad_reason(tool_name, f"Handling detected loop for previous intent: {state.previous_intent}")
        logger.debug("Executing handle_loop node.")
        state.response = "It seems we might be in a loop..." # Added intermediate message


        prompt = f"""
        It seems we might be repeating the same step based on your last input: "{state.user_input}".
        My previous action was related to: {state.previous_intent}.
        I cannot actually execute API calls. Please formulate a message for the user acknowledging the situation and suggesting options related to analyzing specs or describing plans.

        Suggest options like:
        - Rephrasing their last request.
        - Starting over by providing a new OpenAPI spec.
        - Asking to describe the current graph description (if one exists).
        - Asking to describe the current execution plan description (if one exists).
        - Asking a specific question about the loaded spec.

        Response to user about the loop (no execution):
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            state.response = llm_response.strip()
            logger.info("LLM generated handle_loop response.")
        except Exception as e:
             logger.error(f"Error calling LLM for handle_loop: {e}", exc_info=True)
             state.response = "It looks like we might be stuck in a loop. Could you please try rephrasing your request or tell me what you'd like to do next? For example, you could ask me to describe the current plan or graph, ask a question about the spec, or provide a new spec. Remember, I can only describe API specs and potential workflows, not execute them."

        state.update_scratchpad_reason(tool_name, "Provided loop handling response (no execution).")
        return state

    def answer_openapi_query(self, state: BotState) -> BotState:
        """
        Answers general informational queries about the loaded OpenAPI specification,
        identified APIs, generated payload descriptions, or the execution graph description
        using the worker LLM and available state data.
        Emphasizes that these are descriptions/plans, not execution results.
        """
        tool_name = "answer_openapi_query"
        state.update_scratchpad_reason(tool_name, f"Starting to answer general query about spec/plan: {state.user_input}")
        logger.debug("Executing answer_openapi_query node (no execution).")
        state.response = "Answering your question based on the loaded spec and generated artifacts..." # Added intermediate message


        query = state.user_input
        schema_loaded = state.openapi_schema is not None
        graph_exists = state.execution_graph is not None
        apis_identified = state.identified_apis is not None
        # Use renamed state field
        payloads_described = state.payload_descriptions is not None
        plan_exists = state.execution_plan is not None and len(state.execution_plan) > 0

        # Check if there's *any* context to answer from
        if not schema_loaded and not graph_exists and not plan_exists and not apis_identified:
             state.response = "I don't have an OpenAPI specification loaded or any described artifacts (like identified APIs, plans, or graphs) yet. Please provide a spec or ask me to generate something first."
             state.update_scratchpad_reason(tool_name, "Cannot answer query: No schema or artifacts available.")
             logger.info("Cannot answer OpenAPI query: No schema or generated artifacts.")
             return state

        # Construct a prompt for the LLM using available state information
        prompt_parts = [f"The user is asking a question: \"{query}\""]
        prompt_parts.append("\nAnswer the question based *only* on the following context about the currently loaded OpenAPI specification and described artifacts. If the context doesn't contain the answer, say you don't have that information in the current context.")

        if schema_loaded and state.schema_summary:
             prompt_parts.append("\nAvailable OpenAPI Schema Summary:")
             prompt_parts.append("```")
             # Limit summary length in prompt to avoid excessive token usage
             prompt_parts.append(state.schema_summary[:3000] + ("..." if len(state.schema_summary) > 3000 else ""))
             prompt_parts.append("```")
        elif schema_loaded:
             prompt_parts.append("\nContext: An OpenAPI Schema is loaded, but its summary is not available or too long to include fully.")

        if apis_identified:
             prompt_parts.append(f"\nIdentified APIs ({len(state.identified_apis)} found):")
             # Include a summary or list of identified APIs
             api_summaries = [f"- {api.get('operationId', 'N/A')}: {api.get('summary', 'No summary')} ({api.get('method','').upper()} {api.get('path','')})" for api in state.identified_apis[:15]] # Limit list size
             prompt_parts.append("\n".join(api_summaries))
             if len(state.identified_apis) > 15:
                 prompt_parts.append(f"... and {len(state.identified_apis) - 15} more.")

        if graph_exists:
             prompt_parts.append("\nExecution Graph Description Exists:")
             prompt_parts.append(f"- Description: {state.execution_graph.description or 'No overall description provided.'}")
             prompt_parts.append(f"- Nodes (Steps): {len(state.execution_graph.nodes)}")
             # Use effective_id for logging/display
             node_ids_list = [node.effective_id for node in state.execution_graph.nodes]
             prompt_parts.append(f"- Edges (Dependencies): {len(state.execution_graph.edges)}")
             # Include node list if short
             if len(node_ids_list) < 15: # Example limit
                 node_list = ", ".join(node_ids_list)
                 prompt_parts.append(f"  Node Sequence (approx): {node_list}")
             else:
                 prompt_parts.append(f"  Node IDs: {', '.join(node_ids_list[:15])}...")


        if plan_exists:
             prompt_parts.append(f"\nExecution Plan Description Exists:")
             prompt_parts.append(f"- Steps: {len(state.execution_plan)}")
             # Include the plan steps
             prompt_parts.append(f"  Plan Sequence: {state.execution_plan}")

        # Use renamed state field
        if payloads_described:
             prompt_parts.append(f"\nExample Payload Descriptions Generated for {len(state.payload_descriptions)} operations.")
             # Include keys and brief values of generated payload descriptions
             payload_desc_previews = [f"- {op_id}: {desc[:60]}..." for op_id, desc in list(state.payload_descriptions.items())[:10]] # Limit list size and description preview
             prompt_parts.append("  Example Descriptions:")
             prompt_parts.append("\n".join(payload_desc_previews))
             if len(state.payload_descriptions) > 10:
                 prompt_parts.append(f"... and descriptions for {len(state.payload_descriptions) - 10} more operations.")


        prompt_parts.append("\nBased *only* on the context above, answer the user's question clearly and concisely.")
        prompt_parts.append("If the answer isn't in the context, state that clearly.")
        prompt_parts.append("Crucially, remember and subtly remind the user if relevant that I can only analyze the spec and describe workflows/plans; I cannot actually execute API calls.")
        prompt_parts.append("\nAnswer to user:")

        full_prompt = "\n".join(prompt_parts)

        try:
            llm_response = llm_call_helper(self.worker_llm, full_prompt)
            state.response = llm_response.strip()
            state.update_scratchpad_reason(tool_name, "LLM generated response to general query based on context (no execution).")
            logger.info("LLM generated response for general OpenAPI query (no execution).")
        except Exception as e:
            state.response = f"I encountered an error while trying to answer your question about the OpenAPI specification: {e}. Please try rephrasing. Remember, I can only describe specs and plans, not execute them."
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for answer_openapi_query: {e}", exc_info=True)

        return state

# Note: The AddEdgeParams, GeneratePayloadsParams, GenerateGraphParams models
# are defined in models.py and used here via type hints for clarity,
# but their validation/parsing from state.extracted_params happens within
# the respective methods (generate_payloads, generate_execution_graph).
