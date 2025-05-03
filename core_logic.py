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
    Handles the core tasks of parsing OpenAPI specs, generating payloads,
    creating execution graphs, and managing graph modifications using a worker LLM.
    These methods are designed to be used as nodes in the LangGraph.
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
        logger.info("OpenAPICoreLogic initialized.")

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

    def parse_openapi_spec(self, state: BotState) -> BotState: # Return BotState instance
        """
        Parses the raw OpenAPI spec text provided in the state.
        Handles spec text from either openapi_spec_text (for subsequent turns)
        or user_input (for the first turn).
        Uses caching to avoid re-parsing identical specs.
        Relies on the LLM to perform the parsing/resolution.
        Updates state.openapi_schema, state.schema_summary, and state.response.
        """
        tool_name = "parse_openapi_spec"
        state.update_scratchpad_reason(tool_name, "Starting OpenAPI spec parsing.")
        logger.debug("Executing parse_openapi_spec node.")

        spec_text_to_parse = None

        # Check if spec text is already in the dedicated field (subsequent turns)
        if state.openapi_spec_text:
            spec_text_to_parse = state.openapi_spec_text
            logger.debug("Using spec text from state.openapi_spec_text.")
        # If not, check if it's in the user input (first turn)
        elif state.user_input:
            # Heuristic: Assume if openapi_spec_text is empty, the user_input *is* the spec text
            # for this node execution. A more robust approach might involve
            # the router/planner explicitly moving the spec text from user_input.
            spec_text_to_parse = state.user_input
            # Optionally, move it to the dedicated field for future turns
            state.openapi_spec_text = spec_text_to_parse
            logger.debug("Using spec text from state.user_input and moving to state.openapi_spec_text.")

        if not spec_text_to_parse:
            state.response = "Error: No OpenAPI specification text found in the state or user input to parse."
            state.update_scratchpad_reason(tool_name, "Failed: No spec text provided.")
            logger.error("parse_openapi_spec called without spec text in state or user input.")
            state.openapi_spec_text = None
            state.openapi_schema = None
            state.schema_summary = None # Ensure summary is also cleared
            state.schema_cache_key = None
            return state # Return state instance

        # Generate cache key based on the raw text
        cache_key = get_cache_key(spec_text_to_parse)
        state.schema_cache_key = cache_key

        # Try loading from cache first
        cached_schema = load_cached_schema(cache_key)
        if cached_schema:
            state.openapi_schema = cached_schema
            # If schema is loaded from cache, generate or retrieve the summary as well
            # For simplicity, we'll regenerate the summary from the cached schema.
            # A more advanced cache could store the summary too.
            state.schema_summary = self._generate_llm_schema_summary(state.openapi_schema)
            state.response = "Successfully loaded parsed OpenAPI schema from cache."
            state.update_scratchpad_reason(tool_name, f"Loaded schema from cache (key: {cache_key}). Generated summary.")
            logger.info("Loaded OpenAPI schema from cache and generated summary.")
            return state # Return state instance

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
                state.response = "Error: LLM did not return a valid JSON object for the OpenAPI schema."
                state.update_scratchpad_reason(tool_name, f"LLM parsing failed. Raw response: {llm_response[:500]}...")
                logger.error(f"LLM failed to parse OpenAPI spec into valid JSON. Response: {llm_response[:500]}")
                state.openapi_schema = None
                state.schema_summary = None # Ensure summary is cleared on parsing failure
                state.schema_cache_key = None

        except Exception as e:
            state.response = f"Error during OpenAPI parsing LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for OpenAPI parsing: {e}", exc_info=True)
            state.openapi_schema = None
            state.schema_summary = None # Ensure summary is cleared on error
            state.schema_cache_key = None

        return state # Return state instance

    def identify_apis(self, state: BotState) -> BotState: # Return BotState instance
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
            return state # Return state instance
        if not state.schema_summary:
             # This should ideally not happen if parse_openapi_spec ran successfully
             state.response = "Error: Cannot identify APIs without a schema summary. Please parse the spec again."
             state.update_scratchpad_reason(tool_name, "Failed: Schema summary missing.")
             logger.error("identify_apis called without schema_summary.")
             return state # Return state instance


        # Use the schema summary directly from the state
        schema_summary = state.schema_summary

        # Use graph_generation_instructions or user_input as context for identification
        user_context = state.graph_generation_instructions or state.user_input or "general analysis"

        # Prompt for identifying APIs based on the LLM-generated summary
        prompt = f"""
        Analyze the following detailed OpenAPI schema summary and the user's goal to identify the key API endpoints (operations) that are most relevant.
        Consider the user's goal if provided: "{user_context}"
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
            identified_apis = parse_llm_json_output(llm_response) # Expecting a list

            if identified_apis and isinstance(identified_apis, list):
                # Basic validation of list items (check for required keys)
                valid_apis = []
                for api_info in identified_apis:
                     if isinstance(api_info, dict) and all(key in api_info for key in ['operationId', 'method', 'path']):
                          valid_apis.append(api_info)
                     else:
                          logger.warning(f"LLM returned invalid API info format: {api_info}")

                state.identified_apis = valid_apis
                state.response = f"Identified {len(valid_apis)} potentially relevant API endpoints."
                state.update_scratchpad_reason(tool_name, f"LLM identified {len(valid_apis)} APIs. First few: {valid_apis[:3]}")
                logger.info(f"Identified APIs: {valid_apis}")
            else:
                state.response = "Error: LLM did not return a valid JSON list of identified APIs."
                state.update_scratchpad_reason(tool_name, f"LLM API identification failed. Raw response: {llm_response[:500]}...")
                logger.error(f"LLM failed to identify APIs in valid JSON list format. Response: {llm_response[:500]}")
                state.identified_apis = None

        except Exception as e:
            state.response = f"Error during API identification LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for API identification: {e}", exc_info=True)
            state.identified_apis = None

        return state # Return state instance


    def generate_payloads(self, state: BotState) -> BotState: # Return BotState instance
        """
        Generates example payloads for identified API operations using the LLM.
        Considers user instructions if provided via state.payload_generation_instructions
        or state.extracted_params. Updates state.generated_payloads and state.response.
        """
        tool_name = "generate_payloads"
        state.update_scratchpad_reason(tool_name, "Starting payload generation.")
        logger.debug("Executing generate_payloads node.")

        if not state.openapi_schema:
            state.response = "Error: Cannot generate payloads without a parsed OpenAPI schema."
            state.update_scratchpad_reason(tool_name, "Failed: Schema missing.")
            logger.error("generate_payloads called without openapi_schema.")
            return state # Return state instance
        if not state.schema_summary:
             # This should ideally not happen if parse_openapi_spec ran successfully
             state.response = "Error: Cannot generate payloads without a schema summary. Please parse the spec again."
             state.update_scratchpad_reason(tool_name, "Failed: Schema summary missing.")
             logger.error("generate_payloads called without schema_summary.")
             return state # Return state instance


        apis_to_process = state.identified_apis
        # Check for specific instructions or target APIs from parameters (extracted by router or planner)
        params: Optional[GeneratePayloadsParams] = None
        instructions = state.payload_generation_instructions # Use instructions captured earlier
        target_apis_filter: Optional[List[str]] = None

        if state.extracted_params:
            try:
                params = GeneratePayloadsParams.model_validate(state.extracted_params)
                if params.instructions:
                     instructions = params.instructions # Override if new instructions provided
                if params.target_apis:
                     target_apis_filter = params.target_apis
                     logger.info(f"Targeting payload generation for specific APIs from params: {target_apis_filter}")
            except Exception as e: # Catch Pydantic validation errors etc.
                logger.warning(f"Could not parse GeneratePayloadsParams from extracted_params: {e}. Using state fields.")
                # Fallback to instructions potentially set by planner
                instructions = state.payload_generation_instructions or state.extracted_params.get("instructions", instructions)


        # Apply target_apis_filter if specified
        if target_apis_filter and apis_to_process:
             filtered_apis = [api for api in apis_to_process if api.get('operationId') in target_apis_filter]
             if not filtered_apis:
                  state.response = f"Warning: Could not find details for target APIs: {target_apis_filter} among identified APIs. Generating for all identified instead."
                  logger.warning(f"Target APIs not found among identified: {target_apis_filter}")
                  # Proceed with all identified APIs if filter yields nothing
             else:
                  apis_to_process = filtered_apis
                  logger.debug(f"Processing filtered APIs for payloads: {apis_to_process}")


        if not apis_to_process:
             # If no identified APIs and no target_apis_filter, maybe try to get all operations from schema?
             # For now, require identified_apis or a successful target_apis filter.
             state.response = "Error: No relevant API operations found or specified to generate payloads for. Try identifying APIs first."
             state.update_scratchpad_reason(tool_name, "Failed: No APIs to process for payloads.")
             logger.error("generate_payloads called with no APIs to process.")
             return state # Return state instance

        # Use the schema summary directly from the state
        schema_summary = state.schema_summary

        api_list_str = json.dumps(apis_to_process, indent=2)
        payloads_str = json.dumps(state.generated_payloads or "No payloads generated yet", indent=2)

        prompt = f"""
        Based on the detailed OpenAPI schema summary and the list of target API operations below, generate example request payloads.
        Follow these instructions: {instructions}
        For each API operation in the list, determine its required parameters (path, query, header, body) from the schema summary.
        Generate a plausible JSON payload for the request body if applicable (e.g., for POST, PUT, PATCH). Use realistic example values.
        Output ONLY a single JSON object where keys are the 'operationId' from the input list, and values are the generated example payloads (or null if no payload is applicable/needed, e.g., for simple GETs with no body).

        Detailed OpenAPI Schema Summary:
        ```
        {schema_summary}
        ```

        Target API Operations:
        ```json
        {api_list_str}
        ```

        Instructions: {instructions}

        Generated Payloads (JSON Object mapping operationId to payload):
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            generated_payloads = parse_llm_json_output(llm_response) # Expecting a dict

            if generated_payloads and isinstance(generated_payloads, dict):
                state.generated_payloads = generated_payloads
                state.response = f"Generated example payloads for {len(generated_payloads)} API operations."
                state.update_scratchpad_reason(tool_name, f"LLM generated payloads for keys: {list(generated_payloads.keys())}")
                logger.info(f"Generated Payloads: {generated_payloads}")
            else:
                state.response = "Error: LLM did not return a valid JSON object mapping operationIds to payloads."
                state.update_scratchpad_reason(tool_name, f"LLM payload generation failed. Raw response: {llm_response[:500]}...")
                logger.error(f"LLM failed to generate payloads in valid JSON object format. Response: {llm_response[:500]}")
                state.generated_payloads = None # Clear on failure

        except Exception as e:
            state.response = f"Error during payload generation LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for payload generation: {e}", exc_info=True)
            state.generated_payloads = None # Clear on error

        return state # Return state instance

    def generate_execution_graph(self, state: BotState) -> BotState: # Return BotState instance
        """
        Generates a directed acyclic graph (DAG) representing the execution flow
        of identified APIs, considering dependencies and user goals/instructions.
        This is a core "thought process" using the worker LLM.
        Updates state.execution_graph and state.response.
        Includes explicit input mapping instructions in the graph nodes.
        """
        tool_name = "generate_execution_graph"
        state.update_scratchpad_reason(tool_name, "Starting execution graph generation.")
        logger.debug("Executing generate_execution_graph node.")

        if not state.openapi_schema:
            state.response = "Error: Cannot generate execution graph without a parsed OpenAPI schema."
            state.update_scratchpad_reason(tool_name, "Failed: Schema missing.")
            logger.error("generate_execution_graph called without openapi_schema.")
            return state # Return state instance
        if not state.schema_summary:
             # This should ideally not happen if parse_openapi_spec ran successfully
             state.response = "Error: Cannot generate execution graph without a schema summary. Please parse the spec again."
             state.update_scratchpad_reason(tool_name, "Failed: Schema summary missing.")
             logger.error("generate_execution_graph called without schema_summary.")
             return state # Return state instance


        # Check for specific instructions or goals from parameters (extracted by router or planner)
        params: Optional[GenerateGraphParams] = None
        user_goal = state.user_input or "Determine a logical execution flow." # Default goal
        graph_instructions = "Consider typical CRUD dependencies (Create->Read->Update->Read->Delete) and data flow (e.g., an ID created in one step is used in the next)." # Default instructions

        # Use graph_generation_instructions if already set by planner, otherwise check extracted_params
        if state.graph_generation_instructions:
             logger.debug("Using graph generation instructions from state.")
             # For now, assume state.graph_generation_instructions is the primary instruction source
             # A more robust approach might parse it into goal/instructions fields
             # If graph_generation_instructions is a string, use it as combined goal/instructions
             user_goal = state.graph_generation_instructions # Use the captured query as the goal
             graph_instructions = "Based on the user's goal, determine the logical API call sequence and explicit data mappings."

        elif state.extracted_params:
            try:
                params = GenerateGraphParams.model_validate(state.extracted_params)
                if params.goal: user_goal = params.goal
                if params.instructions: graph_instructions = params.instructions
                state.graph_generation_instructions = f"Goal: {user_goal}\nInstructions: {graph_instructions}" # Store combined
                logger.debug(f"Graph generation params from extracted_params: Goal='{user_goal}', Instructions='{graph_instructions}'")
            except Exception as e:
                logger.warning(f"Could not parse GenerateGraphParams from extracted_params: {e}. Using default behavior.")
                user_goal = state.extracted_params.get("goal", user_goal) or state.user_input
                graph_instructions = state.extracted_params.get("instructions", graph_instructions) or "Consider typical dependencies and data flow."
                state.graph_generation_instructions = f"Goal: {user_goal}\nInstructions: {graph_instructions}"


        # Use the schema summary directly from the state
        schema_summary = state.schema_summary

        api_list_str = json.dumps(state.identified_apis or "All APIs in schema", indent=2)
        payloads_str = json.dumps(state.generated_payloads or "No payloads generated yet", indent=2)

        prompt = f"""
        Task: Generate an API execution workflow graph as a Directed Acyclic Graph (DAG).
        Crucially, for each node in the graph, identify how its input parameters are derived from the results of preceding nodes and specify this using explicit input mappings.

        Context:
        1. User Goal/Task: {user_goal}
        2. Specific Instructions: {graph_instructions}
        3. Detailed OpenAPI Schema Summary:
            ```
            {schema_summary}
            ```
        4. Identified Relevant APIs (or 'All APIs in schema'):
            ```json
            {api_list_str}
            ```
        5. Generated Example Payloads (or 'No payloads generated yet'):
            ```json
            {payloads_str}
            ```

        Instructions for Graph Generation:
        1. Analyze the relationships and potential dependencies between the API operations based on the schema, common workflow patterns (like CRUD), and the user's goal.
        2. Determine a logical sequence of API calls to achieve the goal. Think step-by-step about data flow (e.g., needing an ID from a 'create' response for a subsequent 'get' or 'update' call).
        3. Represent this workflow as a graph with nodes and edges.
        4. Nodes: Each node should correspond to an API operation. Include its 'operationId', 'summary', and optionally the generated 'example_payload' if available and relevant. Use operationIds found in the schema.
           **IMPORTANT:** For each node that requires input parameters derived from previous steps, include an `input_mappings` list. Each item in this list should be an object with:
           - `source_operation_id`: The `operationId` of the node that produces the required data.
           - `source_data_path`: A JSONPath expression (e.g., `$.id`, `$.items[0].name`, `$.data.token`) to extract the specific data field from the source node's *result*. Assume results are the parsed JSON response body.
           - `target_parameter_name`: The name of the parameter in the current node's API call that this data maps to.
           - `target_parameter_in`: The location of the target parameter (`path`, `query`, `header`, `cookie`).
           - `transformation`: (Optional) A brief description if the data needs transformation (e.g., "convert to string", "format as date").

        5. Edges: Each edge should represent a dependency or sequential step. Define `from_node` and `to_node` using the operationIds. Add a brief `description` for the edge if the dependency reason is clear (e.g., "Data dependency: uses ID from create response"). Ensure edges align with the `input_mappings`.
        6. Ensure the graph is a DAG (no circular dependencies). If a potential cycle exists (e.g., repeatedly checking status), represent it logically or note the pattern in the graph description.
        7. Provide a brief natural language `description` of the overall workflow represented by the graph.

        Output Format:
        Output ONLY a single JSON object matching this structure:
        ```json
        {{
          "nodes": [
            {{
              "operationId": "...",
              "summary": "...",
              "example_payload": {{...}} or null,
              "input_mappings": [
                {{
                  "source_operation_id": "...",
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
          "edges": [ {{ "from_node": "opId1", "to_node": "opId2", "description": "..." }}, ... ],
          "description": "Overall workflow description..."
        }}
        ```
        Ensure all `operationId`s used in nodes and edges exist in the schema summary.
        Ensure `source_operation_id` in mappings refers to a preceding node in the planned sequence.
        Ensure `target_parameter_name` and `target_parameter_in` match parameters expected by the target operation according to the schema summary.

        Generated Execution Graph (JSON Object):
        """

        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            # Use parse_llm_json_output with the expected model for validation
            graph_output = parse_llm_json_output(llm_response, expected_model=GraphOutput)

            if graph_output and isinstance(graph_output, GraphOutput):
                node_ids = {node.operationId for node in graph_output.nodes}
                valid_graph_structure = True
                invalid_edge_msgs = []
                invalid_mapping_msgs = []

                # Validate edges
                for edge in graph_output.edges:
                     if edge.from_node not in node_ids or edge.to_node not in node_ids:
                          invalid_edge_msgs.append(f"Edge references non-existent node: {edge.from_node} -> {edge.to_node}")
                          valid_graph_structure = False

                # Validate mappings (basic checks - more detailed validation might require schema analysis)
                for node in graph_output.nodes:
                    for mapping in node.input_mappings:
                        if mapping.source_operation_id not in node_ids:
                             invalid_mapping_msgs.append(f"Mapping in node '{node.operationId}' references non-existent source node: '{mapping.source_operation_id}'")
                             valid_graph_structure = False
                        # Could add checks here to see if target_parameter_name/in exist in the schema for this node's operationId

                is_acyclic, cycle_msg = check_for_cycles(graph_output)

                if not is_acyclic:
                    state.response = f"Error: LLM generated a graph with cycles. {cycle_msg} Cannot accept cyclic graph."
                    state.update_scratchpad_reason(tool_name, f"Graph generation failed: Cycle detected. {cycle_msg}")
                    logger.error(f"LLM generated cyclic graph: {cycle_msg}")
                    state.execution_graph = None
                elif not valid_graph_structure:
                    error_details = "; ".join(invalid_edge_msgs + invalid_mapping_msgs)
                    state.response = f"Error: LLM generated a graph with structural errors (invalid edges or mappings). Details: {error_details}"
                    state.update_scratchpad_reason(tool_name, f"Graph generation failed: Structural errors. {error_details}")
                    logger.error(f"LLM generated graph with structural errors: {error_details}")
                    state.execution_graph = None
                else:
                    state.execution_graph = graph_output
                    state.response = f"Successfully generated execution graph with {len(graph_output.nodes)} nodes and {len(graph_output.edges)} edges."
                    if graph_output.description:
                         state.response += f" Description: {graph_output.description}"
                    state.update_scratchpad_reason(tool_name, f"LLM generated graph. Nodes: {len(graph_output.nodes)}, Edges: {len(graph_output.edges)}. Desc: {graph_output.description}")
                    logger.info(f"Generated Graph: {graph_output.model_dump_json(indent=2)}")

            else:
                state.response = "Error: LLM did not return a valid JSON object matching the GraphOutput structure."
                state.update_scratchpad_reason(tool_name, f"LLM graph generation failed. Raw response: {llm_response[:500]}...")
                logger.error(f"LLM failed to generate graph in valid GraphOutput format. Response: {llm_response[:500]}")
                state.execution_graph = None

        except Exception as e:
            state.response = f"Error during graph generation LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for graph generation: {e}", exc_info=True)
            state.execution_graph = None

        return state # Return state instance


    def describe_graph(self, state: BotState) -> BotState: # Return BotState instance
        """Generates a natural language description of the current execution graph using the LLM."""
        tool_name = "describe_graph"
        state.update_scratchpad_reason(tool_name, "Starting graph description.")
        logger.debug("Executing describe_graph node.")

        if not state.execution_graph:
            state.response = "Error: No execution graph exists to describe."
            state.update_scratchpad_reason(tool_name, "Failed: Graph missing.")
            logger.error("describe_graph called without execution_graph.")
            return state # Return state instance

        if state.execution_graph.description:
             state.response = state.execution_graph.description
             state.update_scratchpad_reason(tool_name, "Used existing graph description.")
             logger.info("Using existing graph description.")
             return state # Return state instance

        graph_json = state.execution_graph.model_dump_json(indent=2)
        prompt = f"""
        Based on the following API execution graph (JSON format), provide a concise, natural language description of the workflow it represents.
        Focus on the sequence of actions and potential data flow implied by the nodes, edges, and input mappings.

        Execution Graph JSON:
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
            state.response = description
            state.execution_graph.description = description
            state.update_scratchpad_reason(tool_name, f"LLM generated graph description (length {len(description)}).")
            logger.info(f"Generated graph description: {description}")

        except Exception as e:
            state.response = f"Error during graph description LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for graph description: {e}", exc_info=True)
            state.response = f"The graph contains {len(state.execution_graph.nodes)} steps and {len(state.execution_graph.edges)} dependencies."

        return state # Return state instance

    def get_graph_json(self, state: BotState) -> BotState: # Return BotState instance
        """Outputs the current execution graph as a JSON string."""
        tool_name = "get_graph_json"
        state.update_scratchpad_reason(tool_name, "Starting get graph JSON.")
        logger.debug("Executing get_graph_json node.")

        if not state.execution_graph:
            state.response = "Error: No execution graph exists to output."
            state.update_scratchpad_reason(tool_name, "Failed: Graph missing.")
            logger.error("get_graph_json called without execution_graph.")
            return state # Return state instance

        try:
            graph_json = state.execution_graph.model_dump_json(indent=2)
            state.response = graph_json
            state.update_scratchpad_reason(tool_name, "Outputted graph JSON.")
            logger.info("Outputted graph JSON.")
        except Exception as e:
            state.response = f"Error serializing graph to JSON: {e}"
            state.update_scratchpad_reason(tool_name, f"JSON serialization error: {e}")
            logger.error(f"Error serializing graph: {e}", exc_info=True)

        return state # Return state instance

    def handle_unknown(self, state: BotState) -> BotState: # Return BotState instance
        """Handles cases where the user's intent is unclear or not supported."""
        tool_name = "handle_unknown"
        state.update_scratchpad_reason(tool_name, f"Handling unknown intent for input: {state.user_input}")
        logger.debug("Executing handle_unknown node.")

        prompt = f"""
        The user said: "{state.user_input}"
        My current state includes:
        - OpenAPI spec loaded: {'Yes' if state.openapi_schema else 'No'}
        - Execution graph exists: {'Yes' if state.execution_graph else 'No'}
        - Execution graph summary: {state.execution_graph.description if state.execution_graph else 'None generated'}

        I couldn't determine a specific action (like 'parse spec', 'generate graph', 'execute workflow', etc.) from the user's input or the current state is not ready for the requested action.
        Please formulate a polite response acknowledging the situation and either:
        a) Explain briefly what actions you *can* perform given the current state (e.g., if no spec, ask for one; if spec loaded, mention identifying APIs or building graph; if graph exists, mention execution or description).
        b) Ask for clarification on what the user wants to do.

        Response to user:
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            state.response = llm_response.strip()
            logger.info("LLM generated handle_unknown response.")
        except Exception as e:
             logger.error(f"Error calling LLM for unknown intent handling: {e}", exc_info=True)
             state.response = "I'm sorry, I didn't understand that request or I'm not ready to perform it. Could you please rephrase? I can help with managing OpenAPI specs and execution graphs."

        state.update_scratchpad_reason(tool_name, "Provided clarification response.")
        return state # Return state instance


    def handle_loop(self, state: BotState) -> BotState: # Return BotState instance
        """
        Called when the router detects the same intent repeating.
        Place a message into state to ask the user how to proceed.
        """
        tool_name = "handle_loop"
        state.update_scratchpad_reason(tool_name, f"Handling detected loop for previous intent: {state.previous_intent}")
        logger.debug("Executing handle_loop node.")

        prompt = f"""
        It seems we are repeating the same step or getting stuck based on your last input: "{state.user_input}".
        My previous action was related to: {state.previous_intent}.
        Please formulate a message for the user acknowledging the situation and suggesting options.
        Options could include:
        - Rephrasing their last request.
        - Starting over with a new OpenAPI spec.
        - Describing the current graph.
        - Asking for help.

        Response to user about the loop:
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            state.response = llm_response.strip()
            logger.info("LLM generated handle_loop response.")
        except Exception as e:
             logger.error(f"Error calling LLM for handle_loop: {e}", exc_info=True)
             state.response = "It looks like we're getting stuck. Could you please try rephrasing your request or tell me what you'd like to do next?"

        state.update_scratchpad_reason(tool_name, "Provided loop handling response.")
        return state # Return state instance

    def answer_openapi_query(self, state: BotState) -> BotState: # Return BotState instance
        """
        Answers general informational queries about the loaded OpenAPI specification
        or the generated graph/payloads using the worker LLM and available state data.
        """
        tool_name = "answer_openapi_query"
        state.update_scratchpad_reason(tool_name, f"Starting to answer general query: {state.user_input}")
        logger.debug("Executing answer_openapi_query node.")

        query = state.user_input
        schema_loaded = state.openapi_schema is not None
        graph_exists = state.execution_graph is not None
        apis_identified = state.identified_apis is not None
        payloads_generated = state.generated_payloads is not None

        if not schema_loaded and not graph_exists:
             state.response = "I don't have an OpenAPI specification loaded or an execution graph generated yet. Please provide a spec first."
             state.update_scratchpad_reason(tool_name, "Cannot answer query: No schema or graph.")
             logger.info("Cannot answer OpenAPI query: No schema or graph.")
             return state # Return state instance

        # Construct a prompt for the LLM using available state information
        prompt_parts = [f"The user is asking a question about the loaded OpenAPI specification or the generated workflow artifacts: \"{query}\""]

        if schema_loaded and state.schema_summary:
             prompt_parts.append("\nAvailable OpenAPI Schema Summary:")
             prompt_parts.append("```")
             prompt_parts.append(state.schema_summary)
             prompt_parts.append("```")
        elif schema_loaded:
             prompt_parts.append("\nOpenAPI Schema is loaded, but summary is not available.")

        if apis_identified:
             prompt_parts.append(f"\nIdentified APIs ({len(state.identified_apis)} found):")
             # Include a summary or list of identified APIs, but maybe not the full JSON if very long
             api_summaries = [f"- {api.get('operationId', 'N/A')}: {api.get('summary', 'No summary')}" for api in state.identified_apis[:10]] # Limit list size
             prompt_parts.append("\n".join(api_summaries))
             if len(state.identified_apis) > 10:
                 prompt_parts.append(f"... and {len(state.identified_apis) - 10} more.")

        if graph_exists:
             prompt_parts.append("\nExecution Graph Exists:")
             prompt_parts.append(f"- Description: {state.execution_graph.description or 'No description available.'}")
             prompt_parts.append(f"- Nodes: {len(state.execution_graph.nodes)}")
             prompt_parts.append(f"- Edges: {len(state.execution_graph.edges)}")
             # Optionally include a summary of graph nodes/edges if concise
             if len(state.execution_graph.nodes) < 10: # Example limit
                 node_list = ", ".join([node.operationId for node in state.execution_graph.nodes])
                 prompt_parts.append(f"  Nodes: {node_list}")


        if payloads_generated:
             prompt_parts.append(f"\nExample Payloads Generated for {len(state.generated_payloads)} operations.")
             # Optionally include keys of generated payloads
             if len(state.generated_payloads) < 10: # Example limit
                 payload_keys = ", ".join(state.generated_payloads.keys())
                 prompt_parts.append(f"  Payloads for: {payload_keys}")


        prompt_parts.append("\nBased on the information above, please answer the user's question clearly and concisely.")
        prompt_parts.append("Answer to user:")

        full_prompt = "\n".join(prompt_parts)

        try:
            llm_response = llm_call_helper(self.worker_llm, full_prompt)
            state.response = llm_response.strip()
            state.update_scratchpad_reason(tool_name, "LLM generated response to general query.")
            logger.info("LLM generated response for general OpenAPI query.")
        except Exception as e:
            state.response = f"I encountered an error while trying to answer your question about the OpenAPI specification: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for answer_openapi_query: {e}", exc_info=True)

        return state # Return state instance


