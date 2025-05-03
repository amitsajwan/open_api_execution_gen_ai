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
    intent: Optional[str] = Field(None, description="The user's intent as determined by the router LLM.")
    extracted_params: Optional[Dict[str, Any]] = Field(None, description="Parameters extracted by the router LLM for the current intent.")
    
    # Output and Communication
    response: Optional[str] = Field(None, description="The final response to be shown to the user.")
    
    # Internal working memory
    scratchpad: Dict[str, Any] = Field(default_factory=dict, description="Persistent memory for intermediate results, reasoning, history etc.")

    class Config:
        # Allow extra fields in scratchpad without validation errors
        extra = 'allow'

    def update_scratchpad_reason(self, tool_name: str, details: str):
        """Helper to append reasoning/details to the scratchpad."""
        current_reason = self.scratchpad.get('reasoning_log', '')
        new_entry = f"\n---\nTool: {tool_name}\nDetails: {details}\n---\n"
        # Keep log manageable, e.g., last 5000 chars
        combined = (current_reason + new_entry)[-5000:]
        self.scratchpad['reasoning_log'] = combined
        logger.debug(f"Scratchpad Updated by {tool_name}: {details}")

