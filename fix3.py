import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.messages import (
    BaseMessage, AIMessage, ToolMessage, HumanMessage, SystemMessage
)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.chat_models.fake import FakeListChatModel # For runnable example
from langchain_core.runnables import RunnableConfig


# Helper for patch_config (remains the same)
def patch_config(config: Optional[RunnableConfig], **kwargs):
    if config is None:
        config = {}
    if 'recursion_limit' in kwargs:
        kwargs['recursion_limit'] = int(kwargs['recursion_limit'])
    config.update(kwargs)
    return config

# === 1. State Schema ===
class BotState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    openapi_spec: str
    api_details: Optional[Dict[str, Any]] # Populated by identify_apis
    execution_graph: Optional[Dict[str, Any]] # Populated by build_exec_graph
    # Example: a field for generated payload
    generated_payload_data: Optional[Dict[str, Any]]

    is_planner_done: bool # Flag for planner to signal completion
    feedback: Dict[str, Any] # For cycle checks or other tool feedback
    final_response: str

    # Control flow
    loop_counter: int
    max_loops: int
    initialised: bool # System init flag

# === 2. Dummy Tools defined with @tool ===
@tool("identify_apis_tool", description="Extracts API details from an OpenAPI spec.")
def identify_apis_tool(openapi_spec: str) -> Dict[str, Any]:
    """Dummy: Returns predefined API details based on spec content."""
    print(f"[Tool Called] identify_apis_tool with spec: '{openapi_spec[:30]}...'")
    if "cycle_spec" in openapi_spec:
        return {"api_details": {"data": "APIs with a potential cycle structure", "spec_type": "cycle"}}
    elif "valid_spec" in openapi_spec:
        return {"api_details": {"data": "Standard set of APIs", "endpoints": ["/users", "/orders"], "spec_type": "valid"}}
    return {"api_details": None, "error": "Unknown spec for identify_apis_tool"}

@tool("build_execution_graph_tool", description="Builds an API execution graph from API details.")
def build_execution_graph_tool(api_details: Dict[str, Any]) -> Dict[str, Any]:
    """Dummy: Returns a graph structure based on api_details."""
    print(f"[Tool Called] build_execution_graph_tool with details: '{api_details.get('spec_type', 'N/A')}'")
    if not api_details or api_details.get("data") is None:
        return {"execution_graph": None, "error": "Missing api_details for graph building"}
    if api_details.get("spec_type") == "cycle":
        return {"execution_graph": {"nodes": ["A->B", "B->A"], "is_cyclic": True}}
    return {"execution_graph": {"nodes": ["UserAPI", "OrderAPI"], "dependencies": {"OrderAPI": "UserAPI"}, "is_cyclic": False}}

@tool("check_graph_cycles_tool", description="Checks the execution graph for cycles.")
def check_graph_cycles_tool(execution_graph: Dict[str, Any]) -> Dict[str, Any]:
    """Dummy: Returns feedback based on the 'is_cyclic' flag in the graph."""
    print(f"[Tool Called] check_graph_cycles_tool")
    if not execution_graph:
        return {"feedback": {"has_cycle": False, "message": "Graph not available for cycle check."}}
    if execution_graph.get("is_cyclic"):
        return {"feedback": {"has_cycle": True, "message": "Cycle detected in the graph by dummy tool!", "suggestion": "Review dependencies."}}
    return {"feedback": {"has_cycle": False, "message": "No cycles found by dummy tool."}}

@tool("generate_api_payload_tool", description="Generates a sample payload for a given API endpoint.")
def generate_api_payload_tool(endpoint_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Dummy: Returns a generic payload."""
    print(f"[Tool Called] generate_api_payload_tool for endpoint: '{endpoint_name}' with params: {parameters}")
    return {"generated_payload_data": {"endpoint": endpoint_name, "payload": {"sampleValue": 123, "params_received": parameters}}}

# ToolNode that will execute the LLM's chosen tools
# The output of these tools should be dictionaries that can update BotState directly.
# LangGraph's ToolNode by default appends ToolMessages. To update state directly from tool output,
# tools should return dicts with keys matching BotState fields.
# The @tool decorator and ToolNode handle this well if tool outputs match state keys.
# If not, we'd need a custom node to map tool outputs to state.
# For simplicity, let's assume tools return dicts that can update relevant BotState fields
# e.g. identify_apis_tool returns {"api_details": ...}
# ToolNode will then produce ToolMessages. The crucial part is how these ToolMessages update the overall state.
# The `add_messages` reducer handles the messages list. For other state fields,
# if the tools return dicts with keys matching BotState fields, and if ToolNode is used in a way
# that its output directly updates the state (e.g. if the graph node returns tool_results),
# then state can be updated.
# With ToolNode, the primary output is adding ToolMessages.
# To update state like `api_details` from a tool, often the planner LLM needs to see the ToolMessage
# and then potentially call another tool to explicitly update state, or the graph structure ensures this.

# Let's make the tools return values directly, and the ToolNode will wrap them in ToolMessages.
# The PlannerAgentLLM will then "see" these ToolMessages in the history.
# If direct state update is desired from tools, this would typically be handled by how the
# `ToolNode().invoke(state)` output is merged back into the main state dictionary.
# For this skeleton, we'll rely on the LLM seeing tool outputs in messages.
# And for key structured data like `api_details`, `execution_graph`, we'll have the
# Planner LLM explicitly call tools and then "know" these fields *should* be updated
# based on successful tool execution. The dummy tools will return dicts, and we can
# simulate that these update the state for the *next* LLM turn.

# To ensure state fields like `api_details` are updated:
# We need a mechanism after ToolNode runs to explicitly take parts of ToolMessage content
# and put them into the correct state fields.
# Or, the tools themselves are not directly updating state fields, but their JSON output
# is parsed by the LLM.

# Let's simplify: The tools will return the data. The planner LLM will be "told" via prompt
# that successful tool execution implies certain state fields are now populated.
# The dummy tools will print, showing they were called.

# List of all tools for the ToolNode
available_tools = [
    identify_apis_tool, build_execution_graph_tool,
    check_graph_cycles_tool, generate_api_payload_tool
]
tool_executor = ToolNode(available_tools)

# === 3. LLM Instances (FakeLLM) ===
# Responses will be crafted to simulate a basic planning flow.
# Order of responses matters for FakeListChatModel
llm_responses_for_planner = [
    AIMessage(content="Okay, I need to understand the APIs first.", tool_calls=[{"name": "identify_apis_tool", "args": {"openapi_spec": "valid_spec"}, "id": "call_identify"}]),
    AIMessage(content="Now that I have API details, I'll build the execution graph.", tool_calls=[{"name": "build_execution_graph_tool", "args": {"api_details": {"data": "Standard set of APIs", "endpoints": ["/users", "/orders"], "spec_type": "valid"}}, "id": "call_buildgraph"}]), # LLM "sees" output from previous tool to form args
    AIMessage(content="Graph built. Let's check for cycles.", tool_calls=[{"name": "check_graph_cycles_tool", "args": {"execution_graph": {"nodes": ["UserAPI", "OrderAPI"], "is_cyclic": False}}, "id": "call_checkcycle"}]),
    AIMessage(content="Graph is clean. User wants a payload for /users.", tool_calls=[{"name": "generate_api_payload_tool", "args": {"endpoint_name": "/users", "parameters": {"userId": "123"}}, "id": "call_genpayload"}]),
    AIMessage(content="All actions taken. Ready to respond."), # Signals planner is done
]
# For a cycle scenario
llm_responses_for_cycle_planner = [
    AIMessage(content="Okay, I need to understand the APIs first (cycle spec).", tool_calls=[{"name": "identify_apis_tool", "args": {"openapi_spec": "cycle_spec"}, "id": "call_identify_cycle"}]),
    AIMessage(content="Now that I have API details (cycle spec), I'll build the execution graph.", tool_calls=[{"name": "build_execution_graph_tool", "args": {"api_details": {"data": "APIs with a potential cycle structure", "spec_type": "cycle"}}, "id": "call_buildgraph_cycle"}]),
    AIMessage(content="Graph built (cycle spec). Let's check for cycles.", tool_calls=[{"name": "check_graph_cycles_tool", "args": {"execution_graph": {"nodes": ["A->B", "B->A"], "is_cyclic": True}}, "id": "call_checkcycle_cycle"}]),
    AIMessage(content="Cycle detected in graph! I cannot proceed further with planning normal operations."), # Signals planner is done due to error
]

# We'll select which response list to use based on initial spec for demo
planner_llm = FakeListChatModel(responses=[]) # Will be populated based on scenario

responder_llm_response = AIMessage(content="Here is the information you requested, incorporating all tool actions and checks.")
responder_llm = FakeListChatModel(responses=[responder_llm_response])


# === 4. LLM-Powered Nodes ===
def create_llm_messages(state: BotState, system_prompt_content: str) -> List[BaseMessage]:
    """Helper to create message list for LLM call."""
    messages = [SystemMessage(content=system_prompt_content)]
    # Filter messages to provide a clean history for the LLM
    history = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.content == "Thinking...": # Skip placeholder
            continue
        history.append(msg)
    messages.extend(history)

    if state.get("feedback") and state["feedback"].get("message"):
        messages.append(SystemMessage(content=f"SYSTEM_FEEDBACK: {json.dumps(state['feedback'])}"))
    return messages

def planner_agent_node(state: BotState) -> Dict[str, Any]:
    """LLM-driven planner node."""
    print(f"\n[Node] PlannerAgentNode | Loop: {state['loop_counter']} | Initialised: {state['initialised']}")
    print(f"  State Preview: API Details: {'Present' if state.get('api_details') else 'Missing'}, Graph: {'Present' if state.get('execution_graph') else 'Missing'}")
    print(f"  Feedback: {state.get('feedback')}")

    if state["loop_counter"] >= state["max_loops"]:
        print("  Max loops reached. Forcing planner to be done.")
        return {"is_planner_done": True, "feedback": {"type": "error", "message": "Max loops reached."}}

    system_prompt = (
        "You are a meticulous AI planner. Your job is to use available tools to gather information "
        "and prepare for fulfilling the user's request. "
        "Available tools: identify_apis_tool, build_execution_graph_tool, check_graph_cycles_tool, generate_api_payload_tool.\n"
        "Process to follow generally:\n"
        "1. If API details are missing, call identify_apis_tool.\n"
        "2. If execution graph is missing, call build_execution_graph_tool (needs API details from step 1).\n"
        "3. If graph exists, call check_graph_cycles_tool.\n"
        "4. Based on user query and available graph, call other tools like generate_api_payload_tool if needed.\n"
        "5. If a cycle is detected or a critical error occurs, note it and conclude planning.\n"
        "If you believe all necessary information is gathered and preparatory actions are complete, respond with just the content 'PLANNING_COMPLETE'. "
        "Otherwise, make the next appropriate tool call."
        f"Current openapi_spec in state: {state.get('openapi_spec')}"
    )
    messages = create_llm_messages(state, system_prompt)

    # Simulate LLM call
    print(f"  LLM Input Messages (last 2): {messages[-2:]}")
    ai_response = planner_llm.invoke(messages) # FakeListChatModel pops from its list
    print(f"  LLM Raw Response: {ai_response}")

    is_done = ai_response.content == "PLANNING_COMPLETE" or not ai_response.tool_calls

    # Manual state updates based on dummy tool "outputs" for next LLM turn
    # This simulates the effect of ToolMessages being processed and state being updated.
    # In a real complex system, this logic would be more robust or an explicit state mapping node.
    updated_state_from_tools = {}
    if state["messages"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
            tool_name = available_tools[[t.name for t in available_tools].index(last_message.name)].name # Get the actual tool name
            # Dummy tool output parsing logic
            try:
                tool_output = json.loads(last_message.content) # Assuming tools return JSON strings
                if tool_name == "identify_apis_tool":
                    updated_state_from_tools["api_details"] = tool_output.get("api_details")
                elif tool_name == "build_execution_graph_tool":
                    updated_state_from_tools["execution_graph"] = tool_output.get("execution_graph")
                elif tool_name == "check_graph_cycles_tool":
                    updated_state_from_tools["feedback"] = tool_output.get("feedback", state.get("feedback"))
                elif tool_name == "generate_api_payload_tool":
                    updated_state_from_tools["generated_payload_data"] = tool_output.get("generated_payload_data")
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON from tool output: {last_message.content}")


    print(f"  Planner Decision: IsDone={is_done}, ToolCalls={ai_response.tool_calls}")
    return {
        "messages": [ai_response],
        "is_planner_done": is_done,
        "loop_counter": state["loop_counter"] + 1,
        **updated_state_from_tools # Apply updates from "parsed" tool outputs
    }

def answer_generator_node(state: BotState) -> Dict[str, Any]:
    """LLM-driven responder node."""
    print("\n[Node] AnswerGeneratorNode")
    print(f"  Final State Preview: API Details: {'Present' if state.get('api_details') else 'Missing'}, Graph: {'Present' if state.get('execution_graph') else 'Missing'}")
    print(f"  Feedback: {state.get('feedback')}")
    print(f"  Generated Payload: {state.get('generated_payload_data')}")

    system_prompt = (
        "You are a helpful AI assistant. Summarize the actions taken (from message history) and "
        "provide a response to the user. If there were errors (see feedback or tool messages), explain them."
    )
    messages = create_llm_messages(state, system_prompt)

    # Simulate LLM call
    ai_response = responder_llm.invoke(messages)
    print(f"  Responder LLM Output: {ai_response.content}")

    return {"final_response": ai_response.content, "messages": [ai_response]}

# === 5. System Initialisation Node ===
def initialise_system_node(state: BotState) -> Dict[str, Any]:
    print("\n[Node] initialise_system_node")
    # Select LLM response list based on spec for demo
    global planner_llm # pylint: disable=global-statement
    if state.get("openapi_spec") == "cycle_spec":
        planner_llm = FakeListChatModel(responses=llm_responses_for_cycle_planner.copy())
    else: # Default to valid spec flow
        planner_llm = FakeListChatModel(responses=llm_responses_for_planner.copy())

    return {
        "initialised": True,
        "feedback": {},
        "is_planner_done": False,
        "loop_counter": 0,
        "max_loops": state.get("max_loops", 7), # Default max loops
        "api_details": None, # Ensure these are reset for each run
        "execution_graph": None,
        "generated_payload_data": None,
    }

# === 6. Router Logic ===
def should_continue_planning(state: BotState) -> str:
    print(f"\n[Router] should_continue_planning? PlannerDone: {state.get('is_planner_done')}, Loop: {state.get('loop_counter')}/{state.get('max_loops')}")
    if state.get("is_planner_done"):
        print("  Decision: Planner is done -> Generate Answer")
        return "answer_generator_node"
    if state.get("loop_counter", 0) >= state.get("max_loops", 7):
        print("  Decision: Max loops reached -> Generate Answer (with potential error state)")
        return "answer_generator_node" # Force to answer if stuck in loop

    # Default is to continue planning (which might involve calling tools or thinking)
    # tools_condition checks if the last message was an AIMessage with tool_calls
    if tools_condition(state) == "tools":
         print("  Decision: Tools called by LLM -> Execute Tools")
         return "tool_executor" # LangGraph's key for ToolNode if AIMessage has tool_calls
    print("  Decision: Continue Planning -> Planner Agent")
    return "planner_agent_node"


# === 7. Assemble Graph ===
graph_builder = StateGraph(BotState)

graph_builder.add_node("initialise_system", initialise_system_node)
graph_builder.add_node("planner_agent_node", planner_agent_node)
graph_builder.add_node("tool_executor", tool_executor) # Standard ToolNode execution
graph_builder.add_node("answer_generator_node", answer_generator_node)

graph_builder.set_entry_point("initialise_system")

graph_builder.add_edge("initialise_system", "planner_agent_node")

graph_builder.add_conditional_edges(
    "planner_agent_node",
    tools_condition, # This checks if the last AIMessage has tool_calls
    {
        "tools": "tool_executor", # If tool_calls present, execute them
        "end": "answer_generator_node"  # If no tool_calls (LLM signaled PLANNING_COMPLETE or error)
                                        # The "end" key from tools_condition means no tools were called by the LLM.
                                        # We need our own conditional logic here based on "is_planner_done"
    }
)
# Custom conditional edge from planner if tools_condition leads to "end" (no tools called by planner)
# This should actually be handled by a router *after* planner
# Let's simplify: Planner output (AIMessage) is checked by `tools_condition`.
# If tools -> tool_executor. If no tools -> means planner LLM wants to stop or is done.
# The problem is `tools_condition`'s "end" path implies the graph should end if no tools.
# We need to route based on our own logic.

# New routing logic:
graph_builder.add_conditional_edges(
    "planner_agent_node",
    should_continue_planning, # Our custom router after planner
    {
        "tool_executor": "tool_executor",
        "planner_agent_node": "planner_agent_node", # Should not happen directly here.
        "answer_generator_node": "answer_generator_node"
    }
)


graph_builder.add_edge("tool_executor", "planner_agent_node") # Always go back to planner after tools
graph_builder.add_edge("answer_generator_node", END)

compiled_graph = graph_builder.compile().with_config(patch_config(None, recursion_limit=150))


# === 8. FastAPI WS stub ===
app = FastAPI()

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.headers.get("sec-websocket-key", "default_session")
    print(f"\n[WebSocket ({session_id})] New connection.")

    # Initial state for a new session
    # Choose initial_spec: "valid_spec" or "cycle_spec"
    initial_spec = "valid_spec" # or "cycle_spec"
    # initial_spec = "cycle_spec"

    initial_state_template: BotState = {
        "messages": [], "openapi_spec": initial_spec,
        "api_details": None, "execution_graph": None, "generated_payload_data": None,
        "is_planner_done": False, "feedback": {}, "final_response": "",
        "loop_counter": 0, "max_loops": 5, # Reduced for quicker demo
        "initialised": False
    }
    
    print(f"[WebSocket ({session_id})] Kicking off with spec: {initial_spec}")
    # The first i
