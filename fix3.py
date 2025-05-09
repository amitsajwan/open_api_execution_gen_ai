import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.messages import (
    BaseMessage, AIMessage, ToolMessage, HumanMessage, SystemMessage
)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END # Ensure END is imported
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_community.chat_models.fake import FakeListChatModel
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
    api_details: Optional[Dict[str, Any]]
    execution_graph: Optional[Dict[str, Any]]
    cycle_check_feedback: Optional[Dict[str, Any]]
    generated_payload: Optional[Dict[str, Any]]
    is_planner_done: bool
    final_response: str
    loop_counter: int
    max_loops: int
    initialised: bool
    current_feedback: Optional[Dict[str, Any]]

# === 2. Dummy Tools defined with @tool ===
@tool("identify_apis_tool", description="Extracts API details from an OpenAPI spec.")
def identify_apis_tool(openapi_spec: str) -> Dict[str, Any]:
    print(f"[Tool Executed] identify_apis_tool (spec: '{openapi_spec[:20]}...')")
    if "valid_spec" in openapi_spec:
        return {"identified_api_details": {"data": "Valid APIs identified", "endpoints_count": 2, "spec_used": openapi_spec}}
    if "cycle_spec" in openapi_spec:
        return {"identified_api_details": {"data": "Cycle-prone APIs identified", "spec_used": openapi_spec}}
    return {"error": "Unknown spec for identify_apis_tool", "identified_api_details": None}

@tool("build_execution_graph_tool", description="Builds an API execution graph from API details.")
def build_execution_graph_tool(current_api_details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    print(f"[Tool Executed] build_execution_graph_tool (API details: {'Present' if current_api_details else 'Missing'})")
    if not current_api_details or current_api_details.get("data") is None: # Check for None or empty
        return {"built_execution_graph": None, "error": "Missing or invalid api_details for graph building"}
    if "Cycle-prone" in current_api_details.get("data", ""):
        return {"built_execution_graph": {"nodes": ["A", "B"], "edges": ["A->B", "B->A"], "is_cyclic_structure": True}}
    return {"built_execution_graph": {"nodes": ["Users", "Orders"], "edges": ["Users->Orders"], "is_cyclic_structure": False}}

@tool("check_graph_cycles_tool", description="Checks the execution graph for cycles.")
def check_graph_cycles_tool(current_execution_graph: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    print(f"[Tool Executed] check_graph_cycles_tool (Graph: {'Present' if current_execution_graph else 'Missing'})")
    if not current_execution_graph:
        return {"graph_cycle_feedback": {"has_cycle": False, "message": "Graph not available for cycle check."}}
    if current_execution_graph.get("is_cyclic_structure"):
        return {"graph_cycle_feedback": {"has_cycle": True, "message": "Cycle detected in graph by tool!", "suggestion": "Review dependencies."}}
    return {"graph_cycle_feedback": {"has_cycle": False, "message": "No cycles found."}}

@tool("generate_api_payload_tool", description="Generates a sample payload for an API endpoint.")
def generate_api_payload_tool(endpoint_name: str) -> Dict[str, Any]:
    print(f"[Tool Executed] generate_api_payload_tool for endpoint: '{endpoint_name}'")
    return {"newly_generated_payload": {"for_endpoint": endpoint_name, "payload": {"id": 1, "data": "sample"}}}

available_tools = [identify_apis_tool, build_execution_graph_tool, check_graph_cycles_tool, generate_api_payload_tool]
tool_executor = ToolNode(available_tools)

# === 3. LLM Instances (FakeLLM) ===
def create_fake_planner_responses(spec_type: str) -> List[AIMessage]:
    if spec_type == "cycle_spec":
        return [
            AIMessage(content="", tool_calls=[{"name": "identify_apis_tool", "args": {"openapi_spec": "cycle_spec"}, "id": "tc1_cycle"}]),
            AIMessage(content="", tool_calls=[{"name": "build_execution_graph_tool", "args": {"current_api_details": {"data": "Cycle-prone APIs identified", "spec_used": "cycle_spec"}}, "id": "tc2_cycle"}]),
            AIMessage(content="", tool_calls=[{"name": "check_graph_cycles_tool", "args": {"current_execution_graph": {"is_cyclic_structure": True}}, "id": "tc3_cycle"}]),
            AIMessage(content="PLANNING_COMPLETE_WITH_ERROR") # Cycle detected, planner stops
        ]
    # Default to valid_spec flow
    return [
        AIMessage(content="", tool_calls=[{"name": "identify_apis_tool", "args": {"openapi_spec": "valid_spec"}, "id": "tc1_valid"}]),
        AIMessage(content="", tool_calls=[{"name": "build_execution_graph_tool", "args": {"current_api_details": {"data": "Valid APIs identified", "spec_used": "valid_spec"}}, "id": "tc2_valid"}]),
        AIMessage(content="", tool_calls=[{"name": "check_graph_cycles_tool", "args": {"current_execution_graph": {"is_cyclic_structure": False}}, "id": "tc3_valid"}]),
        AIMessage(content="User wants payload for /users.", tool_calls=[{"name": "generate_api_payload_tool", "args": {"endpoint_name": "/users"}, "id": "tc4_valid"}]),
        AIMessage(content="PLANNING_COMPLETE")
    ]

planner_llm = FakeListChatModel(responses=[])
responder_llm = FakeListChatModel(responses=[AIMessage(content="Based on the actions, here is your summary.")])

# === 4. LLM-Powered Nodes ===
def create_llm_messages_for_planner(state: BotState) -> List[BaseMessage]:
    # (Content remains largely the same as before, ensures system prompt and relevant history)
    system_prompt = (
        "You are an AI planner. Your goal is to use tools to prepare for the user's request. "
        "Available tools: identify_apis_tool, build_execution_graph_tool, check_graph_cycles_tool, generate_api_payload_tool.\n"
        "Sequence: identify_apis -> build_execution_graph -> check_graph_cycles. Then handle user requests.\n"
        "If setup complete & request addressed, or critical error (like cycle), respond 'PLANNING_COMPLETE' or 'PLANNING_COMPLETE_WITH_ERROR'. Else, call next tool.\n"
        f"Spec: {state.get('openapi_spec')}, API Details: {bool(state.get('api_details'))}, Graph: {bool(state.get('execution_graph'))}, Cycle FB: {state.get('cycle_check_feedback')}, Gen FB: {state.get('current_feedback')}"
    )
    history = [msg for msg in state["messages"] if not (isinstance(msg, AIMessage) and msg.content == "")]
    return [SystemMessage(content=system_prompt)] + history

def planner_agent_node(state: BotState) -> Dict[str, Any]:
    print(f"\n[Node] PlannerAgentNode | Loop: {state['loop_counter']+1}/{state['max_loops']}")
    print(f"  State In: api_details: {bool(state.get('api_details'))}, exec_graph: {bool(state.get('execution_graph'))}, cycle_fb: {bool(state.get('cycle_check_feedback'))}")

    if state["loop_counter"] >= state["max_loops"]:
        print("  Max loops reached. Forcing planner to complete with error feedback.")
        return {
            "messages": [AIMessage(content="Max loops reached during planning.")], # Add a message for context
            "is_planner_done": True,
            "current_feedback": {"type": "error", "message": "Max loops reached by planner."}
        }

    messages_for_llm = create_llm_messages_for_planner(state)
    ai_response = planner_llm.invoke(messages_for_llm) # FakeListChatModel pops
    print(f"  LLM Raw Response: content='{ai_response.content}', tool_calls={ai_response.tool_calls}")

    # Determine if planner is done based on LLM response content or lack of tool calls
    planner_finished_signal = "PLANNING_COMPLETE" in ai_response.content # Includes _WITH_ERROR
    is_done_by_llm = planner_finished_signal or not (hasattr(ai_response, 'tool_calls') and ai_response.tool_calls)


    updates_from_last_tool_run: Dict[str, Any] = {}
    last_message = state["messages"][-1] if state["messages"] else None
    if isinstance(last_message, ToolMessage):
        print(f"  Processing ToolMessage: id={last_message.tool_call_id}, name={last_message.name}, content='{last_message.content[:100]}...'")
        try:
            tool_output_data = json.loads(last_message.content)
            if last_message.name == "identify_apis_tool":
                updates_from_last_tool_run["api_details"] = tool_output_data.get("identified_api_details")
                if tool_output_data.get("error"): updates_from_last_tool_run["current_feedback"] = {"type": "error", "message": tool_output_data.get("error")}
            elif last_message.name == "build_execution_graph_tool":
                updates_from_last_tool_run["execution_graph"] = tool_output_data.get("built_execution_graph")
                if tool_output_data.get("error"): updates_from_last_tool_run["current_feedback"] = {"type": "error", "message": tool_output_data.get("error")}
            elif last_message.name == "check_graph_cycles_tool":
                cycle_fb = tool_output_data.get("graph_cycle_feedback")
                updates_from_last_tool_run["cycle_check_feedback"] = cycle_fb
                if cycle_fb and cycle_fb.get("has_cycle"):
                    updates_from_last_tool_run["current_feedback"] = {"type": "error", "message": cycle_fb.get("message", "Cycle detected!"), "from_tool": "check_graph_cycles_tool"}
                    # If cycle detected, LLM should have said PLANNING_COMPLETE_WITH_ERROR, making is_done_by_llm true.
            elif last_message.name == "generate_api_payload_tool":
                updates_from_last_tool_run["generated_payload"] = tool_output_data.get("newly_generated_payload")
        except json.JSONDecodeError: print(f"  Warning: Could not parse JSON from ToolMessage content: {last_message.content}")
        except Exception as e: print(f"  Error processing ToolMessage: {e}")
    
    # If feedback indicates a critical error, ensure planner stops
    final_is_done = is_done_by_llm or (updates_from_last_tool_run.get("current_feedback", {}).get("type") == "error")

    print(f"  Planner Decision: is_done_by_llm={is_done_by_llm}, final_is_done={final_is_done}. Updates from tools: {list(updates_from_last_tool_run.keys())}")
    return {
        "messages": [ai_response], "is_planner_done": final_is_done,
        "loop_counter": state["loop_counter"] + 1,
        **updates_from_last_tool_run
    }

def answer_generator_node(state: BotState) -> Dict[str, Any]:
    # (Content largely same as before, creates a summary response)
    print("\n[Node] AnswerGeneratorNode")
    print(f"  Final State: api_details: {bool(state.get('api_details'))}, exec_graph: {bool(state.get('execution_graph'))}, cycle_fb: {state.get('cycle_check_feedback')}, gen_payload: {state.get('generated_payload')}")
    print(f"  Final Current Feedback: {state.get('current_feedback')}")

    summary_parts = ["Summary of operations:"]
    if state.get("current_feedback"): summary_parts.append(f"Feedback: {state['current_feedback']['message']}")
    if state.get("api_details"): summary_parts.append(f"API Details: {state['api_details'].get('data', 'Not available')}")
    if state.get("execution_graph"): summary_parts.append(f"Graph: {state['execution_graph'].get('nodes', 'Not available')}")
    if state.get("cycle_check_feedback"): summary_parts.append(f"Cycle Check: {state['cycle_check_feedback'].get('message', 'Not available')}")
    if state.get("generated_payload"): summary_parts.append(f"Generated Payload: {state['generated_payload'].get('for_endpoint', 'N/A')}")
    
    final_response_str = "\n".join(summary_parts) + "\nEnd of operation."
    # Simulate LLM creating this from messages or state
    # final_response_str = responder_llm.invoke(state['messages']).content

    print(f"  Generated Final Response: {final_response_str}")
    return {"final_response": final_response_str, "messages": [AIMessage(content=final_response_str)]}

# === 5. System Initialisation Node ===
def initialise_system_node(state: BotState) -> Dict[str, Any]:
    print("\n[Node] initialise_system_node")
    global planner_llm # pylint: disable=global-statement
    spec_to_use = state.get("openapi_spec", "valid_spec")
    planner_llm = FakeListChatModel(responses=create_fake_planner_responses(spec_to_use))
    print(f"  Initialised planner_llm with {len(planner_llm.responses)} responses for spec: {spec_to_use}")
    return {
        "initialised": True, "api_details": None, "execution_graph": None,
        "cycle_check_feedback": None, "generated_payload": None, "current_feedback": None,
        "is_planner_done": False, "loop_counter": 0,
        "max_loops": state.get("max_loops", 6),
        "messages": state.get("messages", [])
    }

# === 6. Router Logic ===
def route_after_planner(state: BotState) -> str:
    """Determines the next step after the planner_agent_node."""
    print(f"\n[Router] route_after_planner | PlannerDone: {state.get('is_planner_done')}, Loop: {state.get('loop_counter')}/{state.get('max_loops')}")
    
    # Priority 1: If planner explicitly signals it's done or an error has occurred.
    if state.get("is_planner_done"):
        print("  Route decision: Planner is done -> answer_generator_node")
        return "answer_generator_node"
        
    # Priority 2: If max loops reached.
    if state.get("loop_counter", 0) >= state.get("max_loops", 6):
        print("  Route decision: Max loops reached -> answer_generator_node")
        # Ensure current_feedback reflects this if not already set by planner
        # This path should ideally be preempted by planner setting is_planner_done.
        # If we reach here, it's a safety net.
        # It's better if planner_agent_node sets is_planner_done and feedback for max_loops.
        return "answer_generator_node"

    # Priority 3: Check if the last message from planner contains tool calls.
    last_ai_message = state["messages"][-1] if state["messages"] and isinstance(state["messages"][-1], AIMessage) else None
    if last_ai_message and hasattr(last_ai_message, "tool_calls") and last_ai_message.tool_calls:
        print("  Route decision: Planner called tools -> tool_executor")
        return "tool_executor"
    
    # Fallback/Unexpected: If planner is not done, not max_loops, and called no tools.
    # This state implies the planner LLM might want to "think" more or there's an issue.
    # For this skeleton, we'll consider this as planner needing to finish.
    # A more robust agent might loop back to planner or have an explicit "think" state.
    print("  Route decision: Planner called no tools and not explicitly done (should ideally not happen often if LLM follows prompt) -> answer_generator_node (to conclude)")
    # To be safe, this should probably go back to planner if not for the risk of tight loop.
    # If this path is hit, it means the FakeLLM prompt or responses for planner might need adjustment
    # to always either call tools or explicitly signal "PLANNING_COMPLETE...".
    # Forcing to answer_generator_node here to prevent potential deadlocks in the skeleton.
    return "answer_generator_node"


# === 7. Assemble Graph ===
graph_builder = StateGraph(BotState)
graph_builder.add_node("initialise_system", initialise_system_node)
graph_builder.add_node("planner_agent_node", planner_agent_node)
graph_builder.add_node("tool_executor", tool_executor)
graph_builder.add_node("answer_generator_node", answer_generator_node)

graph_builder.set_entry_point("initialise_system")
graph_builder.add_edge("initialise_system", "planner_agent_node")
graph_builder.add_conditional_edges(
    "planner_agent_node",
    route_after_planner,
    { # path_map: keys are strings returned by route_after_planner
        "tool_executor": "tool_executor",
        "answer_generator_node": "answer_generator_node"
        # IMPORTANT: Ensure route_after_planner ONLY returns keys defined here.
        # If it could return END or "__end__", that must be mapped too:
        # "__end__": END (or whatever string END is)
    }
)
graph_builder.add_edge("tool_executor", "planner_agent_node")
graph_builder.add_edge("answer_generator_node", END) # Correctly terminate

compiled_graph = graph_builder.compile().with_config(patch_config(None, recursion_limit=25))


app = FastAPI()
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.headers.get("sec-websocket-key", "skeleton_session")
    print(f"\n[WebSocket ({session_id})] New connection.")

    # Choose initial_spec: "valid_spec" or "cycle_spec"
    initial_spec_to_use = "valid_spec"
    # initial_spec_to_use = "cycle_spec"

    # This is the input to the graph for the first run for this session.
    # It should align with BotState. `initialise_system_node` will set defaults too.
    initial_graph_input: BotState = {
        "messages": [HumanMessage(content=f"Hello, please process the API spec: {initial_spec_to_use}")],
        "openapi_spec": initial_spec_to_use,
        "api_details": None, "execution_graph": None, "cycle_check_feedback": None,
        "generated_payload": None, "current_feedback": None,
        "is_planner_done": False, "final_response": "",
        "loop_counter": 0, "max_loops": 6, # max_loops set here will be used by initialise_system_node
        "initialised": False # Will be set to True by initialise_system_node
    }
    
    print(f"[WebSocket ({session_id})] Kicking off graph with spec: {initial_spec_to_use}")
    
    accumulated_state_for_logging = initial_graph_input.copy()

    async for event_chunk in compiled_graph.astream(
        initial_graph_input, # Provide the full initial state for the session
        config={"configurable": {"thread_id": session_id}}
    ):
        node_name = list(event_chunk.keys())[0]
        node_output = event_chunk[node_name]
        print(f"[WebSocket ({session_id}) Stream] Node: {node_name} | Output keys: {list(node_output.keys()) if isinstance(node_output, dict) else 'N/A'}")
        # For logging/display, manually update our local copy of state
        if isinstance(node_output, dict):
            for key, value in node_output.items():
                if key == "messages": # Use add_messages for proper handling
                    accumulated_state_for_logging["messages"] = add_messages(accumulated_state_for_logging.get("messages", []), value)
                else:
                    accumulated_state_for_logging[key] = value
    
    final_state_after_stream = accumulated_state_for_logging # This is our best guess from stream
    # Or, to be absolutely sure of the final persisted state for the thread_id:
    # final_state_after_stream = await compiled_graph.get_state(config={"configurable": {"thread_id": session_id}})


    print(f"\n[WebSocket ({session_id})] Graph processing complete.")
    if final_state_after_stream:
        print(f"  Final State: api_details: {bool(final_state_after_stream.get('api_details'))}, exec_graph: {bool(final_state_after_stream.get('execution_graph'))}")
        print(f"  Final Cycle FB: {final_state_after_stream.get('cycle_check_feedback')}, Final Gen Payload: {bool(final_state_after_stream.get('generated_payload'))}")
        print(f"  Final Current FB: {final_state_after_stream.get('current_feedback')}")
        final_response_from_state = final_state_after_stream.get("final_response", "No final response explicitly in state. Check logs for summary.")
        await websocket.send_text(final_response_from_state)
        print(f"  Sent final response: {final_response_from_state[:100]}...")
    else:
        await websocket.send_text("Graph processing finished, but final state could not be determined from stream.")

    try:
        while True:
            text = await websocket.receive_text()
            await websocket.send_text(f"Received: '{text}'. Conversational follow-up is not part of this auto-run skeleton.")
    except WebSocketDisconnect: print(f"[WebSocket ({session_id})] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket ({session_id})] Error: {e}")
        await websocket.close(code=1011, reason=f"Internal server error: {str(e)[:100]}")
    finally: print(f"[WebSocket ({session_id})] Connection closed.")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://0.0.0.0:8000/ws")
    print("Edit 'initial_spec_to_use' in ws_endpoint to 'valid_spec' or 'cycle_spec'.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

Continue with Gemini Advanced
You've reached your limit on 2.5 Pro (preview) until May 10, 8:08 am. Try Gemini Advanced for higher limits.

Try now








