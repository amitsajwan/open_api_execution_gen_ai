# filename: models.py
import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

# Module-level logger
logger = logging.getLogger(__name__)

# --- Graph Representation Models ---

# Define a model for input mapping instructions (still useful for describing the plan)
class InputMapping(BaseModel):
    """Defines how to map data from previous results to a parameter of this node (as described in a plan)."""
    source_operation_id: str = Field(..., description="The operationId of the previous node whose described result contains the source data.")
    source_data_path: str = Field(..., description="A path or expression (e.g., JSONPath) to extract the data from the source node's described result.")
    target_parameter_name: str = Field(..., description="The name of the parameter in the current node's operation that this data maps to.")
    # Optional: Add parameter 'in' (path, query, header, cookie) for clarity/validation
    target_parameter_in: Optional[str] = Field(None, description="The location of the target parameter (path, query, header, cookie).")
    # Optional: Add transformation instructions if needed (e.g., format date)
    transformation: Optional[str] = Field(None, description="Optional instructions for transforming the data before mapping.")


class Node(BaseModel):
    """Represents a node (an API call description) in the execution graph."""
    operationId: str = Field(..., description="Unique identifier for the API operation (from OpenAPI spec).")
    summary: Optional[str] = Field(None, description="Short summary of the operation (from OpenAPI spec).")
    description: Optional[str] = Field(None, description="Detailed description of the operation.")
    # FIX: Changed type from Optional[Dict[str, Any]] to Optional[str]
    payload_description: Optional[str] = Field(None, description="Description of an example payload for this API call.")
    input_mappings: List[InputMapping] = Field(default_factory=list, description="Instructions on how data would be mapped from previous described results.")


class Edge(BaseModel):
    """Represents a directed edge (dependency) in the execution graph description."""
    from_node: str = Field(..., description="The operationId of the source node.")
    to_node: str = Field(..., description="The operationId of the target node.")
    description: Optional[str] = Field(None, description="Optional description of why this dependency exists (e.g., data dependency).")

    # Make Edge hashable for use in sets
    def __hash__(self):
        return hash((self.from_node, self.to_node))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return NotImplemented
        return self.from_node == other.from_node and self.to_node == other.to_node

class GraphOutput(BaseModel):
    """Represents the generated API execution graph description."""
    nodes: List[Node] = Field(default_factory=list, description="List of API operations (nodes) in the graph description.")
    edges: List[Edge] = Field(default_factory=list, description="List of dependencies (edges) between nodes in the graph description.")
    description: Optional[str] = Field(None, description="Natural language description of the overall workflow.")

# --- Tool Parameter Models ---
# These models are used by the router or planner to parse parameters
# extracted from the user query for specific actions.

class AddEdgeParams(BaseModel):
    """Parameters required for the add_edge tool."""
    from_node: str = Field(..., description="The operationId of the source node.")
    to_node: str = Field(..., description="The operationId of the target node.")
    description: Optional[str] = Field(None, description="Optional description for the new edge.")

class GeneratePayloadsParams(BaseModel):
    """Parameters/Instructions for generating payloads (descriptions)."""
    instructions: Optional[str] = Field(None, description="Specific user instructions for how payloads should be described.")
    target_apis: Optional[List[str]] = Field(None, description="Optional list of specific operationIds to describe payloads for.")

class GenerateGraphParams(BaseModel):
    """Parameters/Instructions for generating the execution graph description."""
    goal: Optional[str] = Field(None, description="The overall user goal or task to accomplish with the described API workflow.")
    instructions: Optional[str] = Field(None, description="Specific user instructions for how the graph should be structured.")

# --- State Model ---

class BotState(BaseModel):
    """Represents the full state of the conversation and processing."""
    session_id: str = Field(..., description="Unique identifier for the current session.")
    user_input: Optional[str] = Field(None, description="The latest input from the user.")

    # OpenAPI Specification related fields
    openapi_spec_text: Optional[str] = Field(None, description="The raw OpenAPI specification text provided by the user.")
    openapi_schema: Optional[Dict[str, Any]] = Field(None, description="The parsed and resolved OpenAPI schema as a dictionary.")
    schema_cache_key: Optional[str] = Field(None, description="Cache key used for the current schema.")
    schema_summary: Optional[str] = Field(None, description="LLM-generated text summary of the OpenAPI schema.")
    # Flag to indicate if the input is likely a spec (set by router)
    input_is_spec: bool = Field(False, description="Flag indicating if the last user input was identified as an OpenAPI spec.")


    # API Identification and Payload Generation (Descriptions)
    identified_apis: Optional[List[Dict[str, Any]]] = Field(None, description="List of APIs identified as potentially relevant by the LLM.")
    # Renamed from generated_payloads to payload_descriptions for clarity
    payload_descriptions: Optional[Dict[str, Any]] = Field(None, description="Dictionary mapping operationId to generated example payload descriptions.")
    payload_generation_instructions: Optional[str] = Field(None, description="User instructions captured for payload description.")

    # Execution Graph Description
    execution_graph: Optional[GraphOutput] = Field(None, description="The generated API execution graph description.")
    graph_generation_instructions: Optional[str] = Field(None, description="User instructions captured for graph description.")

    # Plan-only Fields
    execution_plan: List[str] = Field(default_factory=list, description="Ordered list of operationIds or tool names describing steps of a plan.")
    current_plan_step: int = Field(0, description="Index of the current step in the execution_plan description (useful for tracking).")


    # Routing and Control Flow
    intent: Optional[str] = Field(None, description="The user's high-level intent as determined by the initial router LLM.")
    previous_intent: Optional[str] = None
    loop_counter: int = Field(0, description="Counter to detect potential loops in routing.")

    # Parameters extracted by the initial router or the planner
    extracted_params: Optional[Dict[str, Any]] = Field(None, description="Parameters extracted by the router or planner for the current action.")

    # --- Responder Fields ---
    # The 'results' field is removed as actual execution does not occur.
    final_response: str = Field("", description="The final, user-facing response generated by the responder.")

    # Output and Communication (Intermediate messages from core_logic nodes)
    response: Optional[str] = Field(None, description="Intermediate response message set by nodes (e.g., 'Schema parsed successfully'). Cleared by responder.")

    # Internal working memory
    scratchpad: Dict[str, Any] = Field(default_factory=dict, description="Persistent memory for intermediate results, reasoning, history, planner decisions etc.")

    class Config:
        # Allow extra fields in scratchpad without validation errors
        extra = 'allow'
        # Enforce type validation on assignment to fields
        # Helps catch errors if nodes try to assign incorrect data types
        validate_assignment = True


    def update_scratchpad_reason(self, tool_name: str, details: str):
        """Helper to append reasoning/details to the scratchpad."""
        current_reason = self.scratchpad.get('reasoning_log', '')
        new_entry = f"\n---\nTool: {tool_name}\nDetails: {details}\n---\n"
        # Keep log manageable, e.g., last 5000 chars
        combined = (current_reason + new_entry)[-5000:]
        self.scratchpad['reasoning_log'] = combined
        logger.debug(f"Scratchpad Updated by {tool_name}: {details}")


```

```python
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
                logger.error(f"LLM failed to parse 
