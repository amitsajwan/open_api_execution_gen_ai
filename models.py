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
    display_name: Optional[str] = Field(None, description="A unique display name for this node instance, useful for differentiating multiple calls to the same operation.")
    summary: Optional[str] = Field(None, description="Short summary of the operation (from OpenAPI spec).")
    description: Optional[str] = Field(None, description="Detailed description of the operation.")
    # Changed type from Dict[str, Any] to str for payload description
    payload_description: Optional[str] = Field(None, description="A string description of an example payload for this API call.")
    input_mappings: List[InputMapping] = Field(default_factory=list, description="Instructions on how data would be mapped from previous described results.")

    # Add a computed property or method to get the effective node ID for graph structure
    # This ensures edges/cycle checks use a unique identifier
    @property
    def effective_id(self) -> str:
        """Returns the unique identifier for this node instance in the graph."""
        return self.display_name if self.display_name else self.operationId


class Edge(BaseModel):
    """Represents a directed edge (dependency) in the execution graph description."""
    # Edges should now reference the effective_id (operationId or display_name)
    from_node: str = Field(..., description="The effective_id (operationId or display_name) of the source node.")
    to_node: str = Field(..., description="The effective_id (operationId or display_name) of the target node.")
    description: Optional[str] = Field(None, description="Optional description of why this dependency exists (e.g., data dependency).")

    # Make Edge hashable for use in sets (use effective_id)
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
    from_node: str = Field(..., description="The operationId or display_name of the source node.")
    to_node: str = Field(..., description="The operationId or display_name of the target node.")
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
    # This field stores a dictionary where keys are operationIds and values are the *string* descriptions
    payload_descriptions: Optional[Dict[str, str]] = Field(None, description="Dictionary mapping operationId to generated example payload descriptions (string).")
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

    # LangGraph internal key for routing - exclude from serialization
    __next__: Optional[str] = Field(None, exclude=True, description="LangGraph internal key indicating the next node.") # <-- ADDED THIS LINE


    # Internal working memory
    scratchpad: Dict[str, Any] = Field(default_factory=dict, description="Persistent memory for intermediate results, reasoning, history, planner decisions etc.")

    class Config:
        # Allow extra fields in scratchpad without validation errors
        extra = 'allow'
        # Enforce type validation on assignment to fields
        # Helps catch errors if nodes try to assign incorrect data types
        validate_assignment = True

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
    operationId: str = Field(..., description="The OpenAPI operationId for this step.")
    # Add other relevant fields like parameters, requestBody description, etc.
    # For a non-executable agent, these would be descriptions, not actual data.
    parameters: Optional[Dict[str, Any]] = Field(None, description="Description or example of parameters for this operation.")
    request_body: Optional[Dict[str, Any]] = Field(None, description="Description or example of the request body for this operation.")
    description: Optional[str] = Field(None, description="A natural language description of this step.")


class Edge(BaseModel):
    """Represents a data dependency (edge) in the execution graph."""
    source: str = Field(..., description="The operationId of the source node.")
    target: str = Field(..., description="The operationId of the target node.")
    description: Optional[str] = Field(None, description="A natural language description of the data dependency.")
    input_mapping: Optional[List[InputMapping]] = Field(None, description="List of instructions on how to map data from the source node's result to the target node's parameters.")


class GraphOutput(BaseModel):
    """Represents the description of an execution graph (DAG) generated by the agent."""
    nodes: List[Node] = Field(..., description="A list of nodes (API call descriptions) in the graph.")
    edges: List[Edge] = Field(default_factory=list, description="A list of edges (data dependencies) in the graph.")
    description: Optional[str] = Field(None, description="Overall natural language description of the execution graph.")
    # Add fields for error handling or validation results if needed


# --- Agent State Model ---

# Define the state for the LangGraph
# This state object will be passed between nodes and persisted by the checkpointer
class BotState(BaseModel):
    """
    Represents the state of the conversation and processing for a user session.
    This state object is passed between nodes in the LangGraph.
    """
    # Input from the user for the current turn
    current_step_input: str = Field(default="", description="The user's input for the current turn.")

    # --- OpenAPI Spec and Related Information ---
    # Store the raw spec string temporarily upon receiving it
    openapi_spec_string: Optional[str] = Field(None, description="The raw OpenAPI specification string (JSON or YAML) provided by the user in the current turn. Cleared after parsing attempt.")
    # Flag to indicate if the current input contains a spec string
    input_is_spec: bool = Field(default=False, description="Flag set to True if the current input is detected as an OpenAPI spec string.")

    # Store the parsed OpenAPI schema representation
    # This could be a Pydantic model representing the schema or a summary dict.
    # Using Dict[str, Any] for flexibility, but a dedicated model is recommended for complex apps.
    openapi_schema: Optional[Dict[str, Any]] = Field(None, description="The parsed and processed representation or summary of the OpenAPI specification.")

    # --- Identified APIs and Generated Payloads ---
    identified_apis: List[str] = Field(default_factory=list, description="List of operationIds identified as relevant to the user's query.")
    generated_payloads: Optional[Dict[str, Any]] = Field(None, description="Example request/response payloads generated based on identified APIs.")

    # --- Execution Graph Description ---
    # Store the generated graph description (as a Pydantic model)
    execution_graph: Optional[GraphOutput] = Field(None, description="A description of the planned API execution workflow as a DAG.")

    # --- Conversation History and Scratchpad ---
    # conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="History of user and agent messages.") # Optional: Store history in state if needed
    # Store reasoning and intermediate steps
    scratchpad: Dict[str, Any] = Field(default_factory=dict, description="Persistent memory for intermediate results, reasoning, history, planner decisions etc.")

    # --- Agent Output and Control ---
    # Final response to be sent back to the user
    final_response: Optional[str] = Field(None, description="The final, user-facing response generated by the responder.")

    # Output and Communication (Intermediate messages from core_logic nodes)
    response: Optional[str] = Field(None, description="Intermediate response message set by nodes (e.g., 'Schema parsed successfully'). Cleared by responder.")

    # LangGraph internal key for routing - exclude from serialization
    __next__: Optional[str] = Field(None, exclude=True, description="LangGraph internal key indicating the next node.") # <-- ADDED THIS LINE

    # Loop detection counter
    loop_counter: int = Field(default=0, description="Counter to detect consecutive identical intents (potential loops).")
    previous_intent: Optional[str] = Field(None, description="Stores the intent from the previous router step for loop detection.")
    intent: Optional[str] = Field(None, description="Stores the current determined intent.")


    class Config:
        # Allow extra fields in scratchpad without validation errors
        extra = 'allow'
        # Enforce type validation on assignment to fields
        # Helps catch errors if nodes try to assign incorrect data types
        validate_assignment = True
        # Example: Configure Pydantic to handle serialization/deserialization quirks
        # json_encoders = {
        #     SomeComplexType: lambda v: v.to_json()
        # }
        # json_decoders = {
        #    SomeComplexType: lambda v: SomeComplexType.from_json(v)
        # }


    def update_scratchpad_reason(self, tool_name: str, details: str):
        """Helper to append reasoning/details to the scratchpad."""
        current_reason_log = self.scratchpad.get('reasoning_log', [])
        timestamp = datetime.now().isoformat() # Requires importing datetime
        new_entry = {"timestamp": timestamp, "tool": tool_name, "details": details}
        current_reason_log.append(new_entry)
        # Keep log size manageable, e.g., last 50 entries
        self.scratchpad['reasoning_log'] = current_reason_log[-50:]
        # Optionally also store a simple string log for easier viewing
        current_reason_string = self.scratchpad.get('reasoning_log_string', '')
        new_string_entry = f"\n---\\n[{timestamp}] Tool: {tool_name}\\nDetails: {details}\\n---\\n"
        self.scratchpad['reasoning_log_string'] = (current_reason_string + new_string_entry)[-5000:] # Keep string log size manageable
        logger.debug(f"Scratchpad Updated by {tool_name}: {details}")

# Add datetime import for timestamping
from datetime import datetime
    def update_scratchpad_reason(self, tool_name: str, details: str):
        """Helper to append reasoning/details to the scratchpad."""
        current_reason = self.scratchpad.get('reasoning_log', '')
        new_entry = f"\n---\nTool: {tool_name}\nDetails: {details}\n---\n"
        # Keep log manageable, e.g., last 5000 chars
        combined = (current_reason + new_entry)[-5000:]
        self.scratchpad['reasoning_log'] = combined
        logger.debug(f"Scratchpad Updated by {tool_name}: {details}")
