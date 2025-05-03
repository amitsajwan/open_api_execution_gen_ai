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
        """Extracts relevant parts of the schema, truncating if necessary."""
        # Simple truncation for now, could be smarter (e.g., only paths/components)
        schema_str = json.dumps(schema, indent=2)
        if len(schema_str) > max_length:
            logger.warning(f"Schema text truncated to {max_length} characters.")
            # Try to keep basic structure
            truncated_str = schema_str[:max_length]
            # Find the last complete line
            last_newline = truncated_str.rfind('\n')
            if last_newline != -1:
                 truncated_str = truncated_str[:last_newline] + "\n... (truncated)"
            else:
                 truncated_str = truncated_str + "... (truncated)"
            return truncated_str
        return schema_str

    # --- Tool Methods ---

    def parse_openapi_spec(self, state: BotState) -> Dict[str, Any]:
        """
        Parses the raw OpenAPI spec text provided in the state.
        Uses caching to avoid re-parsing identical specs.
        Relies on the LLM to perform the parsing/resolution.
        """
        tool_name = "parse_openapi_spec"
        state.update_scratchpad_reason(tool_name, "Starting OpenAPI spec parsing.")

        if not state.openapi_spec_text:
            state.response = "Error: No OpenAPI specification text found in the state to parse."
            state.update_scratchpad_reason(tool_name, "Failed: No spec text provided.")
            logger.error("parse_openapi_spec called without openapi_spec_text.")
            return state.model_dump()

        # Generate cache key based on the raw text
        cache_key = get_cache_key(state.openapi_spec_text)
        state.schema_cache_key = cache_key

        # Try loading from cache first
        cached_schema = load_cached_schema(cache_key)
        if cached_schema:
            state.openapi_schema = cached_schema
            state.response = "Successfully loaded parsed OpenAPI schema from cache."
            state.update_scratchpad_reason(tool_name, f"Loaded schema from cache (key: {cache_key}).")
            return state.model_dump()

        # If not cached, use LLM to parse
        state.update_scratchpad_reason(tool_name, "Schema not found in cache. Using LLM to parse.")
        prompt = f"""
        Parse the following OpenAPI specification text into a resolved JSON object.
        Ensure all internal `$ref` links are resolved if possible.
        Output ONLY the resulting JSON object, without any surrounding text or markdown formatting.

        OpenAPI Specification Text:
        ```yaml_or_json
        {state.openapi_spec_text[:15000]} 
        ```
        { "... (Input specification truncated)" if len(state.openapi_spec_text) > 15000 else "" }

        Parsed JSON Object:
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            parsed_schema = parse_llm_json_output(llm_response) # Basic JSON parsing

            if parsed_schema and isinstance(parsed_schema, dict):
                state.openapi_schema = parsed_schema
                state.response = "Successfully parsed OpenAPI specification using LLM."
                state.update_scratchpad_reason(tool_name, f"LLM parsed schema. Keys: {list(parsed_schema.keys())}")
                # Save the newly parsed schema to cache
                save_schema_to_cache(cache_key, parsed_schema)
            else:
                state.response = "Error: LLM did not return a valid JSON object for the OpenAPI schema."
                state.update_scratchpad_reason(tool_name, f"LLM parsing failed. Raw response: {llm_response[:500]}...")
                logger.error(f"LLM failed to parse OpenAPI spec into valid JSON. Response: {llm_response[:500]}")
                # Keep the raw text, but clear the schema field
                state.openapi_schema = None

        except Exception as e:
            state.response = f"Error during OpenAPI parsing LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for OpenAPI parsing: {e}", exc_info=True)
            state.openapi_schema = None

        return state.model_dump()

    def identify_apis(self, state: BotState)-> Dict[str, Any]:
        """
        Identifies relevant API endpoints from the parsed schema based on user goal or general analysis.
        (Placeholder - LLM task)
        """
        tool_name = "identify_apis"
        state.update_scratchpad_reason(tool_name, "Starting API identification.")

        if not state.openapi_schema:
            state.response = "Error: Cannot identify APIs without a parsed OpenAPI schema. Please provide/parse a spec first."
            state.update_scratchpad_reason(tool_name, "Failed: Schema missing.")
            return state.model_dump()

        schema_summary = self._get_relevant_schema_text(state.openapi_schema)
        user_goal = state.graph_generation_instructions or state.user_input or "general analysis"

        prompt = f"""
        Analyze the following OpenAPI schema summary and identify the key API endpoints (operations).
        Consider the user's goal if provided: "{user_goal}"
        For each identified API, extract its 'operationId', 'summary', HTTP 'method', and 'path'.
        Output ONLY a JSON list of objects, where each object represents an identified API endpoint.
        Example format: `[ {{"operationId": "getUser", "summary": "Get user details", "method": "get", "path": "/users/{{userId}}"}}, ... ]`

        OpenAPI Schema Summary:
        ```json
        {schema_summary}
        ```

        Identified APIs (JSON List):
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            identified_apis = parse_llm_json_output(llm_response) # Expecting a list

            if identified_apis and isinstance(identified_apis, list):
                state.identified_apis = identified_apis
                state.response = f"Identified {len(identified_apis)} potentially relevant API endpoints."
                state.update_scratchpad_reason(tool_name, f"LLM identified {len(identified_apis)} APIs. First few: {identified_apis[:3]}")
                logger.info(f"Identified APIs: {identified_apis}")
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

        return state.model_dump()


    def generate_payloads(self, state: BotState)-> Dict[str, Any]:
        """
        Generates example payloads for identified API operations using the LLM.
        Considers user instructions if provided via extracted_params.
        """
        tool_name = "generate_payloads"
        state.update_scratchpad_reason(tool_name, "Starting payload generation.")

        if not state.openapi_schema:
            state.response = "Error: Cannot generate payloads without a parsed OpenAPI schema."
            state.update_scratchpad_reason(tool_name, "Failed: Schema missing.")
            return state.model_dump()
        
        apis_to_process = state.identified_apis
        if not apis_to_process:
             # Maybe try to get all operations if none were explicitly identified?
             # For now, require identify_apis first or target_apis param.
             apis_to_process = [] # Default to empty list

        # Check for specific instructions or target APIs from parameters
        params: Optional[GeneratePayloadsParams] = None
        if state.extracted_params:
            try:
                params = GeneratePayloadsParams.model_validate(state.extracted_params)
                state.payload_generation_instructions = params.instructions # Store instructions
                if params.target_apis:
                     # Filter identified APIs or fetch details for target APIs
                     # This part needs more robust handling based on how APIs are stored/retrieved
                     logger.info(f"Targeting payload generation for specific APIs: {params.target_apis}")
                     # Simplistic filter - assumes identified_apis has the necessary info
                     apis_to_process = [api for api in (state.identified_apis or []) if api.get('operationId') in params.target_apis]
                     if not apis_to_process:
                          state.response = f"Warning: Could not find details for target APIs: {params.target_apis}. Ensure they exist in the schema and were identified."
                          # Potentially try to fetch details directly from schema here
            except Exception as e: # Catch Pydantic validation errors etc.
                logger.warning(f"Could not parse GeneratePayloadsParams: {e}. Using default behavior.")
                state.payload_generation_instructions = state.extracted_params.get("instructions") # Fallback

        if not apis_to_process:
             state.response = "Error: No relevant API operations found or specified to generate payloads for. Try identifying APIs first."
             state.update_scratchpad_reason(tool_name, "Failed: No APIs to process.")
             return state.model_dump()

        schema_summary = self._get_relevant_schema_text(state.openapi_schema)
        api_list_str = json.dumps(apis_to_process, indent=2)
        instructions = state.payload_generation_instructions or "Generate typical example payloads."

        prompt = f"""
        Based on the OpenAPI schema summary and the list of target API operations below, generate example request payloads.
        Follow these instructions: {instructions}
        For each API operation in the list, determine its required parameters (path, query, header, body) from the schema summary.
        Generate a plausible JSON payload for the request body if applicable (e.g., for POST, PUT, PATCH). Use realistic example values.
        Output ONLY a single JSON object where keys are the 'operationId' from the input list, and values are the generated example payloads (or null if no payload is applicable/needed, e.g., for simple GETs with no body).

        OpenAPI Schema Summary:
        ```json
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
                state.generated_payloads = None

        except Exception as e:
            state.response = f"Error during payload generation LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for payload generation: {e}", exc_info=True)
            state.generated_payloads = None

        return state.model_dump()

    def generate_execution_graph(self, state: BotState)-> Dict[str, Any]:
        """
        Generates a directed acyclic graph (DAG) representing the execution flow
        of identified APIs, considering dependencies and user goals.
        This is the core "thought process" using the LLM.
        """
        tool_name = "generate_execution_graph"
        state.update_scratchpad_reason(tool_name, "Starting execution graph generation.")

        if not state.openapi_schema:
            state.response = "Error: Cannot generate execution graph without a parsed OpenAPI schema."
            state.update_scratchpad_reason(tool_name, "Failed: Schema missing.")
            return state.model_dump()

        if not state.identified_apis:
            state.response = "Warning: No APIs explicitly identified. Graph generation might be less accurate. Consider running 'identify_apis' first."
            state.update_scratchpad_reason(tool_name, "Warning: No identified APIs provided.")
            # Proceed, but the LLM might have less context

        # Check for specific instructions or goals from parameters
        params: Optional[GenerateGraphParams] = None
        user_goal = state.user_input or "Determine a logical execution flow."
        graph_instructions = "Consider typical CRUD dependencies (Create->Read->Update->Read->Delete) and data flow (e.g., an ID created in one step is used in the next)."

        if state.extracted_params:
            try:
                params = GenerateGraphParams.model_validate(state.extracted_params)
                if params.goal: user_goal = params.goal
                if params.instructions: graph_instructions = params.instructions
                state.graph_generation_instructions = f"Goal: {user_goal}\nInstructions: {graph_instructions}" # Store combined
            except Exception as e:
                logger.warning(f"Could not parse GenerateGraphParams: {e}. Using default behavior.")
                state.graph_generation_instructions = state.extracted_params.get("instructions") or state.extracted_params.get("goal") # Fallback

        schema_summary = self._get_relevant_schema_text(state.openapi_schema)
        api_list_str = json.dumps(state.identified_apis or "All APIs in schema", indent=2)
        payloads_str = json.dumps(state.generated_payloads or "No payloads generated yet", indent=2)

        prompt = f"""
        Task: Generate an API execution workflow graph as a Directed Acyclic Graph (DAG).

        Context:
        1. User Goal/Task: {user_goal}
        2. Specific Instructions: {graph_instructions}
        3. OpenAPI Schema Summary:
           ```json
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
            # Expecting a GraphOutput structure
            graph_output = parse_llm_json_output(llm_response, expected_model=GraphOutput)

            if graph_output and isinstance(graph_output, GraphOutput):
                 # Basic validation: Check if nodes and edges reference valid operationIds
                 # (More thorough validation could cross-reference with schema)
                 node_ids = {node.operationId for node in graph_output.nodes}
                 valid_graph = True
                 for edge in graph_output.edges:
                      if edge.from_node not in node_ids or edge.to_node not in node_ids:
                           logger.warning(f"LLM generated edge referencing non-existent node: {edge.from_node} -> {edge.to_node}")
                           # Decide whether to discard the edge or the whole graph
                           # For now, let's keep it but log warning. Cycle check might catch issues.
                           # valid_graph = False
                           # break
                 
                 # Check for cycles
                 is_acyclic, cycle_msg = check_for_cycles(graph_output)
                 if not is_acyclic:
                      state.response = f"Error: LLM generated a graph with cycles. {cycle_msg} Cannot accept cyclic graph."
                      state.update_scratchpad_reason(tool_name, f"Graph generation failed: Cycle detected. {cycle_msg}")
                      logger.error(f"LLM generated cyclic graph: {cycle_msg}")
                      state.execution_graph = None # Reject cyclic graph
                 elif not valid_graph:
                      state.response = "Error: LLM generated a graph with invalid edge references."
                      state.update_scratchpad_reason(tool_name, "Graph generation failed: Invalid edges.")
                      logger.error("LLM generated graph with invalid edges.")
                      state.execution_graph = None # Reject graph with invalid edges
                 else:
                      state.execution_graph = graph_output
                      state.response = f"Successfully generated execution graph with {len(graph_output.nodes)} nodes and {len(graph_output.edges)} edges."
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

        return state.model_dump()

    def add_graph_edge(self, state: BotState)-> Dict[str, Any]:
        """Adds an edge to the existing execution graph based on user parameters."""
        tool_name = "add_graph_edge"
        state.update_scratchpad_reason(tool_name, "Starting adding graph edge.")

        if not state.execution_graph:
            state.response = "Error: No execution graph exists to add an edge to. Please generate a graph first."
            state.update_scratchpad_reason(tool_name, "Failed: Graph missing.")
            return state.model_dump()

        if not state.extracted_params:
            state.response = "Error: Missing parameters (from_node, to_node) to add edge."
            state.update_scratchpad_reason(tool_name, "Failed: Parameters missing.")
            return state

        try:
            params = AddEdgeParams.model_validate(state.extracted_params)
            
            # Validate nodes exist
            node_ids = {node.operationId for node in state.execution_graph.nodes}
            if params.from_node not in node_ids:
                state.response = f"Error: Source node '{params.from_node}' not found in the graph."
                state.update_scratchpad_reason(tool_name, f"Failed: Source node {params.from_node} not found.")
                return state.model_dump()
            if params.to_node not in node_ids:
                state.response = f"Error: Target node '{params.to_node}' not found in the graph."
                state.update_scratchpad_reason(tool_name, f"Failed: Target node {params.to_node} not found.")
                return state.model_dump()

            new_edge = Edge(from_node=params.from_node, to_node=params.to_node, description=params.description)

            # Check if edge already exists
            if new_edge in state.execution_graph.edges:
                state.response = f"Edge from '{params.from_node}' to '{params.to_node}' already exists."
                state.update_scratchpad_reason(tool_name, "No-op: Edge already exists.")
                return state.model_dump()
            # Create a potential new graph and check for cycles
            potential_graph = state.execution_graph.model_copy(deep=True)
            potential_graph.edges.append(new_edge)

            is_acyclic, cycle_msg = check_for_cycles(potential_graph)
            if not is_acyclic:
                state.response = f"Error: Adding edge from '{params.from_node}' to '{params.to_node}' would create a cycle. {cycle_msg}"
                state.update_scratchpad_reason(tool_name, f"Failed: Cycle detected - {cycle_msg}")
                return state.model_dump()

            # If acyclic, update the graph
            state.execution_graph = potential_graph
            state.response = f"Successfully added edge from '{params.from_node}' to '{params.to_node}'."
            state.update_scratchpad_reason(tool_name, f"Added edge {params.from_node}->{params.to_node}.")
            logger.info(f"Added edge: {new_edge.model_dump_json()}")

        except Exception as e: # Catch Pydantic validation errors etc.
            state.response = f"Error processing add_edge parameters: {e}"
            state.update_scratchpad_reason(tool_name, f"Parameter processing error: {e}")
            logger.error(f"Error in add_graph_edge: {e}", exc_info=True)

        return state.model_dump()

    def validate_graph(self, state: BotState)-> Dict[str, Any]:
        """Validates the current execution graph, primarily checking for cycles."""
        tool_name = "validate_graph"
        state.update_scratchpad_reason(tool_name, "Starting graph validation.")

        if not state.execution_graph:
            state.response = "No execution graph exists to validate."
            state.update_scratchpad_reason(tool_name, "Skipped: Graph missing.")
            return state.model_dump()

        is_acyclic, message = check_for_cycles(state.execution_graph)

        if is_acyclic:
            state.response = f"Graph validation successful: The graph is a DAG (Directed Acyclic Graph). {message}"
            state.update_scratchpad_reason(tool_name, "Validation successful: DAG.")
        else:
            state.response = f"Graph validation failed: {message}"
            state.update_scratchpad_reason(tool_name, f"Validation failed: {message}")
        
        logger.info(f"Graph validation result: {state.response}")
        return state.model_dump()

    def describe_graph(self, state: BotState)-> Dict[str, Any]:
        """Generates a natural language description of the current execution graph using the LLM."""
        tool_name = "describe_graph"
        state.update_scratchpad_reason(tool_name, "Starting graph description.")

        if not state.execution_graph:
            state.response = "Error: No execution graph exists to describe."
            state.update_scratchpad_reason(tool_name, "Failed: Graph missing.")
            return state.model_dump()

        # Use the description already generated if available and seems good
        if state.execution_graph.description:
             state.response = state.execution_graph.description
             state.update_scratchpad_reason(tool_name, "Used existing graph description.")
             return state.model_dump()

        # Otherwise, ask LLM to generate one
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
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            # Clean up response slightly
            description = llm_response.strip()
            state.response = description
            # Optionally update the description in the graph object itself
            state.execution_graph.description = description
            state.update_scratchpad_reason(tool_name, f"LLM generated graph description (length {len(description)}).")
            logger.info(f"Generated graph description: {description}")

        except Exception as e:
            state.response = f"Error during graph description LLM call: {e}"
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for graph description: {e}", exc_info=True)
            # Fallback description
            state.response = f"The graph contains {len(state.execution_graph.nodes)} steps and {len(state.execution_graph.edges)} dependencies."


        return state.model_dump()

    def get_graph_json(self, state: BotState)-> Dict[str, Any]:
        """Outputs the current execution graph as a JSON string."""
        tool_name = "get_graph_json"
        state.update_scratchpad_reason(tool_name, "Starting get graph JSON.")

        if not state.execution_graph:
            state.response = "Error: No execution graph exists to output."
            state.update_scratchpad_reason(tool_name, "Failed: Graph missing.")
            return state.model_dump()

        try:
            graph_json = state.execution_graph.model_dump_json(indent=2)
            state.response = graph_json
            state.update_scratchpad_reason(tool_name, "Outputted graph JSON.")
            # No need to log the full JSON here, it's the response
        except Exception as e:
            state.response = f"Error serializing graph to JSON: {e}"
            state.update_scratchpad_reason(tool_name, f"JSON serialization error: {e}")
            logger.error(f"Error serializing graph: {e}", exc_info=True)

        return state.model_dump()

    def handle_unknown(self, state: BotState)-> Dict[str, Any]:
        """Handles cases where the user's intent is unclear or not supported."""
        tool_name = "handle_unknown"
        state.update_scratchpad_reason(tool_name, f"Handling unknown intent for input: {state.user_input}")
        
        # Ask LLM to formulate a polite refusal or clarification question
        prompt = f"""
        The user said: "{state.user_input}"
        My current understanding of the OpenAPI spec is based on schema keys: {list(state.openapi_schema.keys()) if state.openapi_schema else 'None loaded'}
        My generated execution graph looks like this (summary): {state.execution_graph.description if state.execution_graph else 'None generated'}

        I couldn't determine a specific action (like 'generate graph', 'add edge', 'parse spec') from the user's input. 
        Please formulate a polite response acknowledging the input and either:
        a) Explain briefly what actions you *can* perform (e.g., parse spec, generate/modify/describe graph, generate payloads).
        b) Ask for clarification on what the user wants to do.

        Response to user:
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            state.response = llm_response.strip()
        except Exception as e:
             logger.error(f"Error calling LLM for unknown intent handling: {e}", exc_info=True)
             # Fallback response
             state.response = "I'm sorry, I didn't understand that request. Could you please rephrase? I can help with parsing OpenAPI specs, generating execution graphs, adding edges, validating graphs, describing workflows, and generating payloads."
             
        state.update_scratchpad_reason(tool_name, "Provided clarification response.")
        return state.model_dump()

