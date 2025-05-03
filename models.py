import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

# Module-level logger
logger = logging.getLogger(__name__)

# --- Graph Representation Models ---

class Node(BaseModel):
    """Represents a node in the execution graph (an API call)."""
    operationId: str = Field(..., description="Unique identifier for the API operation (from OpenAPI spec).")
    summary: Optional[str] = Field(None, description="Short summary of the operation (from OpenAPI spec).")
    description: Optional[str] = Field(None, description="Detailed description of the operation.")
    # Example payload can be added here if generated per node
    example_payload: Optional[Dict[str, Any]] = Field(None, description="Example payload for this API call.")

class Edge(BaseModel):
    """Represents a directed edge in the execution graph."""
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
    """Represents the generated API execution graph."""
    nodes: List[Node] = Field(default_factory=list, description="List of API operations (nodes) in the graph.")
    edges: List[Edge] = Field(default_factory=list, description="List of dependencies (edges) between nodes.")
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
    """Parameters/Instructions for generating payloads."""
    instructions: Optional[str] = Field(None, description="Specific user instructions for how payloads should be generated.")
    target_apis: Optional[List[str]] = Field(None, description="Optional list of specific operationIds to generate payloads for.")

class GenerateGraphParams(BaseModel):
    """Parameters/Instructions for generating the execution graph."""
    goal: Optional[str] = Field(None, description="The overall user goal or task to accomplish with the API workflow.")
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

    # API Identification and Payload Generation
    identified_apis: Optional[List[Dict[str, Any]]] = Field(None, description="List of APIs identified as potentially relevant by the LLM.")
    generated_payloads: Optional[Dict[str, Any]] = Field(None, description="Dictionary mapping operationId to generated example payloads.")
    payload_generation_instructions: Optional[str] = Field(None, description="User instructions captured for payload generation.")

    # Execution Graph
    execution_graph: Optional[GraphOutput] = Field(None, description="The generated API execution graph.")
    graph_generation_instructions: Optional[str] = Field(None, description="User instructions captured for graph generation.")

    # Routing and Control Flow
    # The 'intent' field from the router can still be used for initial high-level routing
    intent: Optional[str] = Field(None, description="The user's high-level intent as determined by the initial router LLM.")
    previous_intent: Optional[str] = None        # last-run intent (from router)
    loop_counter: int = 0                        # repeat counter (from router)

    # Parameters extracted by the initial router or the planner
    extracted_params: Optional[Dict[str, Any]] = Field(None, description="Parameters extracted by the router or planner for the current action.")

    # --- Planner/Executor/Responder Fields ---
    # These fields are used by the new P/E/R nodes
    plan: List[str] = Field(default_factory=list, description="List of operationIds to execute in sequence (generated by planner).")
    current_step: int = Field(0, description="Index of the current step being executed in the plan.")
    results: List[Any] = Field(default_factory=list, description="List of results from executed API calls.")
    final_response: str = Field("", description="The final, user-facing response generated by the responder.")

    # Output and Communication (This field might become less central for final output,
    # with final_response being the primary user-facing message, but can be used
    # for intermediate messages from core_logic nodes)
    response: Optional[str] = Field(None, description="Intermediate response message set by nodes (e.g., 'Schema parsed successfully').")

    # Internal working memory
    scratchpad: Dict[str, Any] = Field(default_factory=dict, description="Persistent memory for intermediate results, reasoning, history, planner decisions etc.")

    class Config:
        # Allow extra fields in scratchpad without validation errors
        extra = 'allow'
        # Allow assignment to fields even if they are not in the initial model_validate call
        # This is useful for nodes modifying the state.
        # validate_assignment = True # Consider adding this for stricter validation


    def update_scratchpad_reason(self, tool_name: str, details: str):
        """Helper to append reasoning/details to the scratchpad."""
        current_reason = self.scratchpad.get('reasoning_log', '')
        new_entry = f"\n---\nTool: {tool_name}\nDetails: {details}\n---\n"
        # Keep log manageable, e.g., last 5000 chars
        combined = (current_reason + new_entry)[-5000:]
        self.scratchpad['reasoning_log'] = combined
        logger.debug(f"Scratchpad Updated by {tool_name}: {details}")

