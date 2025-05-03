import json

import logging
from typing import Any, Dict, List, Optional

# Assuming models.py defines BotState, GraphOutput, Node, Edge, etc.
from models import (
    BotState, GraphOutput, Node, Edge, AddEdgeParams,
    GeneratePayloadsParams, GenerateGraphParams
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

    def _get_relevant_schema_text(self, schema: Dict[str, Any], max_length: int = 8000) -> str:
        """
        Generates a structured, human-readable summary of the OpenAPI schema
        for use in LLM prompts. Prioritizes key information like title, version,
        paths, operations, summaries, and parameters.
        """
        if not schema:
            return "No OpenAPI schema available."

        summary_parts = []

        # Add basic info
        info = schema.get('info', {})
        title = info.get('title', 'Untitled API')
        version = info.get('version', 'N/A')
        summary_parts.append(f"API Title: {title}")
        summary_parts.append(f"Version: {version}\n")

        # Summarize paths and operations
        paths = schema.get('paths', {})
        if not paths:
            summary_parts.append("No paths defined in the schema.")
        else:
            summary_parts.append("Available Paths and Operations:")
            for path, methods in paths.items():
                summary_parts.append(f"- Path: {path}")
                # Iterate through HTTP methods for this path
                for method, operation in methods.items():
                    if isinstance(operation, dict): # Ensure it's an operation object
                        operation_id = operation.get('operationId', 'N/A')
                        op_summary = operation.get('summary', 'No summary available.')
                        description = operation.get('description', '')

                        op_line = f"  - {method.upper()}: {operation_id} - {op_summary}"
                        summary_parts.append(op_line)

                        # Briefly mention parameters
                        parameters = operation.get('parameters')
                        if parameters:
                             param_types = []
                             for param in parameters:
                                 if isinstance(param, dict) and 'in' in param:
                                     param_types.append(param['in'])
                             if param_types:
                                 summary_parts.append(f"    Parameters: {', '.join(set(param_types))}") # Use set to get unique types

                        # Briefly mention request body
                        request_body = operation.get('requestBody')
                        if request_body:
                            summary_parts.append("    Request Body: Yes")

                        # Briefly mention responses
                        responses = operation.get('responses')
                        if responses:
                            response_codes = ", ".join(responses.keys())
                            summary_parts.append(f"    Responses: {response_codes}")

                        # Add description if concise
                        if description and len(description) < 100: # Limit description length
                             summary_parts.append(f"    Description: {description}")


        # Join parts and truncate if necessary
        full_summary = "\n".join(summary_parts)

        if len(full_summary) > max_length:
            logger.warning(f"Schema summary truncated to {max_length} characters.")
            # Truncate gracefully, trying not to cut in the middle of a line
            truncated_summary = full_summary[:max_length]
            last_newline = truncated_summary.rfind('\n')
            if last_newline != -1:
                 truncated_summary = truncated_summary[:last_newline] + "\n... (truncated summary)"
            else:
                 truncated_summary = truncated_summary + "... (truncated summary)"
            return truncated_summary
        return full_summary


    # --- Tool Methods (Designed as Graph Nodes) ---

    def parse_openapi_spec(self, state: BotState) -> BotState: # Return BotState instance
        """
        Parses the raw OpenAPI spec text provided in the state.
        Uses caching to avoid re-parsing identical specs.
        Relies on the LLM to perform the parsing/resolution.
        Updates state.openapi_schema and state.response.
        """
        tool_name = "parse_openapi_spec"
        state.update_scratchpad_reason(tool_name, "Starting OpenAPI spec parsing.")
        logger.debug("Executing parse_openapi_spec node.")

        if not state.openapi_spec_text:
            state.response = "Error: No OpenAPI specification text found in the state to parse."
            state.update_scratchpad_reason(tool_name, "Failed: No spec text provided.")
            logger.error("parse_openapi_spec called without openapi_spec_text.")
            return state # Return state instance

        # Generate cache key based on the raw text
        cache_key = get_cache_key(state.openapi_spec_text)
        state.schema_cache_key = cache_key

        # Try loading from cache first
        cached_schema = load_cached_schema(cache_key)
        if cached_schema:
            state.openapi_schema = cached_schema
            state.response = "Successfully loaded parsed OpenAPI schema from cache."
            state.update_scratchpad_reason(tool_name, f"Loaded schema from cache (key: {cache_key}).")
            logger.info("Loaded OpenAPI schema from cache.")
            return state # Return state instance

        # If not cached, use LLM to parse
        state.update_scratchpad_reason(tool_name, "Schema not found in cache. Using LLM to parse.")
        logger.info("Parsing OpenAPI schema using LLM.")
        # Instruct LLM to parse and resolve, outputting ONLY JSON
        prompt = f"""
        Parse the following OpenAPI specification text (which can be in YAML or JSON format) into a fully resolved JSON object.
        Ensure all internal `$ref` links are resolved if possible, embedding the referenced schema objects directly.
        Handle external references (`$ref` to URLs or local files) by including a note or placeholder if full resolution is not possible, but prioritize resolving internal references.
        Output ONLY the resulting JSON object, without any surrounding text, markdown formatting, or conversational remarks.

        OpenAPI Specification Text:
        ```
        {state.openapi_spec_text[:15000]}
        ```
        { "... (Input specification truncated)" if len(state.openapi_spec_text) > 15000 else "" }

        Parsed and Resolved JSON Object:
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            parsed_schema = parse_llm_json_output(llm_response) # Basic JSON parsing

            if parsed_schema and isinstance(parsed_schema, dict):
                state.openapi_schema = parsed_schema
                state.response = "Successfully parsed OpenAPI specification using LLM."
                state.update_scratchpad_reason(tool_name, f"LLM parsed schema. Keys: {list(parsed_schema.keys())}")
                logger.info("LLM successfully parsed OpenAPI schema.")
                # Save the newly parsed schema to cache
                save_schema_to_cache(cache_key, parsed_schema)
            else:
                state.response = "Error: LLM did not return a valid JSON object for the OpenAPI schema."
                state.update_scratchpad_reason(tool_name, f"LLM parsing failed. Raw response: {llm_response[:500]}...")
                logger.error(f"LLM failed to parse OpenAPI spec into valid JSON. Response: {llm_response[:500]}")
                # Keep the raw text, but clear the schema field and cache key if parsing failed
                state.openapi_schema = None
                state.schema_cache_key = None # Clear cache key if parsing failed

        except Exception as e:
            state.response = f"Error during OpenAPI parsing LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for OpenAPI parsing: {e}", exc_info=True)
            state.openapi_schema = None
            state.schema_cache_key = None # Clear cache key if parsing failed

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

        # Use the enhanced schema summary
        schema_summary = self._get_relevant_schema_text(state.openapi_schema)
        # Use graph_generation_instructions or user_input as context for identification
        user_context = state.graph_generation_instructions or state.user_input or "general analysis"

        prompt = f"""
        Analyze the following OpenAPI schema summary and identify the key API endpoints (operations) that are potentially relevant to the user's goal or for general understanding.
        Consider the user's goal if provided: "{user_context}"
        For each identified API, extract its 'operationId', 'summary', HTTP 'method', and 'path'.
        Output ONLY a JSON list of objects, where each object represents an identified API endpoint.
        Example format: `[ {{"operationId": "getUser", "summary": "Get user details", "method": "get", "path": "/users/{{userId}}"}}, ... ]`

        OpenAPI Schema Summary:
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

        # Use the enhanced schema summary
        schema_summary = self._get_relevant_schema_text(state.openapi_schema)
        api_list_str = json.dumps(apis_to_process, indent=2)
        payloads_str = json.dumps(state.generated_payloads or "No payloads generated yet", indent=2)

        prompt = f"""
        Based on the OpenAPI schema summary and the list of target API operations below, generate example request payloads.
        Follow these instructions: {instructions}
        For each API operation in the list, determine its required parameters (path, query, header, body) from the schema summary.
        Generate a plausible JSON payload for the request body if applicable (e.g., for POST, PUT, PATCH). Use realistic example values.
        Output ONLY a single JSON object where keys are the 'operationId' from the input list, and values are the generated example payloads (or null if no payload is applicable/needed, e.g., for simple GETs with no body).

        OpenAPI Schema Summary:
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
        """
        tool_name = "generate_execution_graph"
        state.update_scratchpad_reason(tool_name, "Starting execution graph generation.")
        logger.debug("Executing generate_execution_graph node.")

        if not state.openapi_schema:
            state.response = "Error: Cannot generate execution graph without a parsed OpenAPI schema."
            state.update_scratchpad_reason(tool_name, "Failed: Schema missing.")
            logger.error("generate_execution_graph called without openapi_schema.")
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
             pass
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


        # Use the enhanced schema summary
        schema_summary = self._get_relevant_schema_text(state.openapi_schema)
        api_list_str = json.dumps(state.identified_apis or "All APIs in schema", indent=2)
        payloads_str = json.dumps(state.generated_payloads or "No payloads generated yet", indent=2)

        prompt = f"""
        Task: Generate an API execution workflow graph as a Directed Acyclic Graph (DAG).

        Context:
        1. User Goal/Task: {user_goal}
        2. Specific Instructions: {graph_instructions}
        3. OpenAPI Schema Summary:
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
        5. Edges: Each edge should represent a dependency or sequential step. Define `from_node` and `to_node` using the operationIds. Add a brief `description` for the edge if the dependency reason is clear (e.g., "Uses ID from create response").
        6. Ensure the graph is a DAG (no circular dependencies). If a potential cycle exists (e.g., repeatedly checking status), represent it logically or note the pattern in the graph description.
        7. Provide a brief natural language `description` of the overall workflow represented by the graph.

        Output Format:
        Output ONLY a single JSON object matching this structure:
        ```json
        {{
          "nodes": [ {{ "operationId": "...", "summary": "...", "example_payload": {{...}} or null }}, ... ],
          "edges": [ {{ "from_node": "opId1", "to_node": "opId2", "description": "..." }}, ... ],
          "description": "Overall workflow description..."
        }}
        ```

        Generated Execution Graph (JSON Object):
        """

        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            graph_output = parse_llm_json_output(llm_response, expected_model=GraphOutput)

            if graph_output and isinstance(graph_output, GraphOutput):
                node_ids = {node.operationId for node in graph_output.nodes}
                valid_graph_structure = True
                invalid_edge_msgs = []
                for edge in graph_output.edges:
                     if edge.from_node not in node_ids or edge.to_node not in node_ids:
                          invalid_edge_msgs.append(f"Edge references non-existent node: {edge.from_node} -> {edge.to_node}")
                          valid_graph_structure = False

                is_acyclic, cycle_msg = check_for_cycles(graph_output)

                if not is_acyclic:
                    state.response = f"Error: LLM generated a graph with cycles. {cycle_msg} Cannot accept cyclic graph."
                    state.update_scratchpad_reason(tool_name, f"Graph generation failed: Cycle detected. {cycle_msg}")
                    logger.error(f"LLM generated cyclic graph: {cycle_msg}")
                    state.execution_graph = None
                elif not valid_graph_structure:
                    state.response = f"Error: LLM generated a graph with invalid edge references. Details: {'; '.join(invalid_edge_msgs)}"
                    state.update_scratchpad_reason(tool_name, f"Graph generation failed: Invalid edges. {'; '.join(invalid_edge_msgs)}")
                    logger.error(f"LLM generated graph with invalid edges: {'; '.join(invalid_edge_msgs)}")
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
        Focus on the sequence of actions and potential data flow.

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

