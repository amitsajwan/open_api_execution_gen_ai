import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.messages import (
    BaseMessage, AIMessage, ToolMessage, HumanMessage, SystemMessage
)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
# Using FakeListChatModel for runnable example without API keys
from langchain_community.chat_models.fake import FakeListChatModel


# Helper for patch_config if not available in user's environment (stub)
def patch_config(config: Optional[RunnableConfig], **kwargs):
    if config is None:
        config = {}
    # Ensure 'recursion_limit' is an integer if present in kwargs
    if 'recursion_limit' in kwargs:
        kwargs['recursion_limit'] = int(kwargs['recursion_limit'])
    config.update(kwargs)
    return config

# === 1. State schema ===
class BotState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    openapi_spec: str
    api_details: Dict[str, Any]
    execution_graph: Dict[str, Any]
    # plan: List[Dict[str, Any]] # LLM will generate tool_calls, this might store a high-level textual plan
    plan_description: Optional[str] # For LLM's textual plan summary
    final_response: str
    # done: bool # Signifies PlannerAgentLLM is done with tool calls
    is_planner_done: bool
    feedback: Dict[str, Any]
    loop_counter: int
    max_loops: int # Added max loops to prevent infinite cycles
    initialised: bool

# === 2. Define Tools (Abilities for the LLM) ===
# These tools remain largely the same as they represent concrete actions.

@tool("identify_apis", description="Extracts API endpoint details and dependencies from an OpenAPI specification string. Use this first to understand available APIs.")
def identify_apis(spec: str) -> Dict[str, Any]:
    print(f"[Tool] identify_apis called with spec: {spec[:70]}...")
    if spec == "<YOUR_SPEC_WITH_CYCLE>":
        return {
            "paths": {
                "/a": {"get": {"summary": "Endpoint A", "x-dependencies": ["GET /b"]}},
                "/b": {"get": {"summary": "Endpoint B", "x-dependencies": ["GET /a"]}}
            }
        }
    if spec == "<VALID_SPEC>":
        return {"paths": {"/users": {"get": {"summary":"Get Users"}}, "/posts": {"get": {"summary":"Get Posts"}}}}
    return {"error": "Invalid or unknown OpenAPI spec provided to identify_apis."}

@tool("build_exec_graph", description="Computes an execution graph from API details (output of 'identify_apis'). This graph shows API call dependencies.")
def build_exec_graph(api_details: Dict[str, Any]) -> Dict[str, Any]:
    print(f"[Tool] build_exec_graph called with api_details: {api_details}")
    if "error" in api_details:
        return {"error": f"Cannot build graph due to error in api_details: {api_details['error']}"}
    graph = {}
    if api_details and "paths" in api_details:
        for path, operations in api_details["paths"].items():
            for op, details in operations.items():
                node_id = f"{op.upper()} {path}"
                dependencies = details.get("x-dependencies", [])
                if not isinstance(dependencies, list): dependencies = [dependencies]
                graph[node_id] = {"dependencies": dependencies, "summary": details.get("summary", "")}
    return graph

@tool("check_graph_for_cycles", description="Checks the API execution graph for circular dependencies. Returns details if a cycle is found.")
def check_graph_for_cycles(graph: Dict[str, Any]) -> Dict[str, Any]: # Renamed for clarity
    print(f"[Tool] check_graph_for_cycles called with graph: {graph}")
    if "error" in graph: return {"has_cycle": False, "message": "Graph not available due to prior error."}
    path, visited = set(), set()
    for node in graph:
        if node not in visited:
            if _has_cycle_util(node, graph, visited, path):
                return {"has_cycle": True, "message": f"Cycle detected involving: {', '.join(list(path))}.", "nodes_involved": list(path)}
    return {"has_cycle": False, "message": "No cycles detected."}

def _has_cycle_util(node_id, graph, visited, path): # Helper for check_graph_for_cycles
    visited.add(node_id)
    path.add(node_id)
    dependencies = graph.get(node_id, {}).get("dependencies", [])
    for dep_id in dependencies:
        if dep_id not in graph: continue
        if dep_id in path: return True
        if dep_id not in visited and _has_cycle_util(dep_id, graph, visited, path): return True
    path.remove(node_id)
    return False

@tool("generate_api_payload", description="Generates a sample payload for a given API endpoint schema or filter criteria.")
def generate_api_payload(endpoint_id: str, schema_or_filter: Dict[str, Any]) -> Dict[str, Any]: # Renamed
    print(f"[Tool] generate_api_payload for '{endpoint_id}' with schema/filter: {schema_or_filter}")
    # In a real scenario, this would use the schema to generate a payload
    return {"payload": {"example_param": "value", "for_endpoint": endpoint_id}}

@tool("add_dependency_edge", description="Adds a new dependency edge (e.g., API A must run before API B) to the execution graph. Useful for resolving issues or customization.")
def add_dependency_edge(graph: Dict[str, Any], from_node: str, to_node: str) -> Dict[str, Any]: # Renamed
    print(f"[Tool] add_dependency_edge: {from_node} -> {to_node}")
    if "error" in graph: return graph # Propagate error
    if from_node not in graph: graph[from_node] = {"dependencies": [], "summary": "New node"}
    if to_node not in graph: graph[to_node] = {"dependencies": [], "summary": "New node"}
    if "dependencies" not in graph[from_node] or not isinstance(graph[from_node]["dependencies"], list):
        graph[from_node]["dependencies"] = []
    if to_node not in graph[from_node]["dependencies"]: graph[from_node]["dependencies"].append(to_node)
    # Important: After adding an edge, re-check for cycles!
    # This tool should ideally trigger a cycle check or the LLM should call it.
    return graph

# This tool node will be used by the LLM planner
system_tool_executor = ToolNode([
    identify_apis, build_exec_graph, check_graph_for_cycles,
    generate_api_payload, add_dependency_edge
])

# === 3. LLM Instances (using FakeListChatModel for example) ===
# In a real app, replace with ChatOpenAI, ChatGoogleGenerativeAI, etc.
# and configure API keys.

# LLM for the PlannerAgentNode
# We will define responses contextually for the fake LLM.
# A single FakeListChatModel instance can be tricky for complex interactions.
# For this example, the "LLM call" will be simulated within the node logic.

# === 4. LLM-Powered Nodes ===

def create_llm_call_messages(state: BotState, system_prompt: str) -> List[BaseMessage]:
    """Helper to create message list for LLM call, including history and feedback."""
    messages = [SystemMessage(content=system_prompt)]
    # Add relevant message history, minus last AI message if it was just a placeholder
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content == "Thinking...": # Avoid duplicate thinking messages
            continue
        messages.append(msg)

    # Append current feedback if any
    if state.get("feedback") and state["feedback"].get("message"):
        messages.append(SystemMessage(content=f"Current system feedback to consider: {json.dumps(state['feedback'])}"))
    return messages

def planner_agent_node(state: BotState, llm: FakeListChatModel) -> Dict[str, Any]:
    print(f"\n[Node] PlannerAgentNode. Loop: {state['loop_counter']}. Feedback: {state['feedback'].get('message','None')}")
    state["loop_counter"] += 1

    if state["loop_counter"] > state["max_loops"]:
        print("[PlannerAgentNode] Max loops reached. Signaling to respond with error.")
        return {
            "feedback": {"type": "error", "code": "MAX_LOOPS_REACHED", "message": "Maximum planning loops reached. Unable to complete request."},
            "is_planner_done": True
        }

    system_prompt = (
        "You are an expert AI orchestrator. Your goal is to fulfill the user's request by planning and executing a sequence of tool calls. "
        "You have access to the following tools: identify_apis, build_exec_graph, check_graph_for_cycles, generate_api_payload, add_dependency_edge.\n"
        "Current OpenAPI Spec: " + state.get("openapi_spec", "Not specified") + "\n"
        "Current API Execution Graph: " + json.dumps(state.get("execution_graph", {}), indent=2) + "\n"
        "Consider the entire message history and any system feedback.\n"
        "If the graph has a cycle or other critical errors indicated in feedback, try to use tools like 'add_dependency_edge' to fix it if appropriate, or inform the user if unfixable.\n"
        "If the spec is not initialized, call 'identify_apis', then 'build_exec_graph', then 'check_graph_for_cycles'.\n"
        "Based on the user's latest query and the current state, decide the next tool to call. "
        "If you have all necessary information and all tools have run successfully, you can respond 'OK_READY_TO_ANSWER'. "
        "If you need to call a tool, provide the tool name and arguments in the required format."
    )
    messages_for_llm = create_llm_call_messages(state, system_prompt)

    # --- FAKE LLM RESPONSE LOGIC ---
    # This section simulates how an LLM might respond based on the state.
    # In a real app, this is `llm.invoke(messages_for_llm)`.
    last_human_message = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    simulated_llm_response = AIMessage(content="Thinking...") # Default

    if not state.get("initialised"): # Needs initialization
        simulated_llm_response = AIMessage(content="Initializing graph...", tool_calls=[{
            "name": "identify_apis", "args": {"spec": state["openapi_spec"]}, "id": "tool_init_identify"
        }])
    elif "tool_init_identify" in str(state["messages"]): # identify_apis was called
         # Check if identify_apis produced an error
        last_tool_msg = next((m for m in reversed(state["messages"]) if isinstance(m, ToolMessage) and m.tool_call_id == "tool_init_identify"), None)
        if last_tool_msg and isinstance(last_tool_msg.content, str) and "error" in json.loads(last_tool_msg.content).get("error",""): # Assuming content is JSON string
             simulated_llm_response = AIMessage(content="Error during identify_apis. Cannot proceed with graph initialization.")
             state["is_planner_done"] = True # Stop planning
        else:
            simulated_llm_response = AIMessage(content="Building execution graph...", tool_calls=[{
                "name": "build_exec_graph", "args": {"api_details": state["api_details"]}, "id": "tool_init_buildgraph"
            }])
    elif "tool_init_buildgraph" in str(state["messages"]): # build_exec_graph was called
        simulated_llm_response = AIMessage(content="Checking graph for cycles...", tool_calls=[{
            "name": "check_graph_for_cycles", "args": {"graph": state["execution_graph"]}, "id": "tool_init_checkcycle"
        }])
    elif "tool_init_checkcycle" in str(state["messages"]): # check_graph_for_cycles was called
        # After this, initialisation is considered done for this fake logic.
        # The actual `initialised` flag is set by the `initialise_system_node`.
        # The LLM should see the result of check_graph_for_cycles and decide.
        # If cycle detected, it might try to fix or say ready to answer (with error).
        last_tool_msg = next((m for m in reversed(state["messages"]) if isinstance(m, ToolMessage) and m.tool_call_id == "tool_init_checkcycle"), None)
        if last_tool_msg:
            cycle_check_data = json.loads(last_tool_msg.content) # Assuming content is JSON string
            if cycle_check_data.get("has_cycle"):
                simulated_llm_response = AIMessage(content=f"Cycle detected: {cycle_check_data.get('message')}. Suggestion: use add_dependency_edge or modify spec. For now, I will report this.")
                state["is_planner_done"] = True # Ready to answer (about the cycle)
            else: # No cycle, init seems fine
                simulated_llm_response = AIMessage(content="OK_READY_TO_ANSWER") # Assuming init is done and no user query yet
    elif "generate payload for /users" in last_human_message.lower():
        simulated_llm_response = AIMessage(content="Generating payload for /users...", tool_calls=[{
            "name": "generate_api_payload", "args": {"endpoint_id": "GET /users", "schema_or_filter": {"type": "users_schema"}}, "id": "tool_genpayload_users"
        }])
    elif "tool_genpayload_users" in str(state["messages"]): # Payload was generated
        simulated_llm_response = AIMessage(content="OK_READY_TO_ANSWER")
    else: # Default if no specific condition met by fake LLM
        if state.get("initialised") and not (state.get("feedback") and state["feedback"].get("type") == "error"):
             simulated_llm_response = AIMessage(content="OK_READY_TO_ANSWER") # If initialized and no errors, ready for query.
        elif state.get("feedback") and state["feedback"].get("type") == "error":
            simulated_llm_response = AIMessage(content=f"Acknowledging error: {state['feedback']['message']}. I will report this.")
            state["is_planner_done"] = True


    # In a real LLM call:
    # llm_response = llm.invoke(messages_for_llm)
    # state_updates = {"messages": [llm_response]}

    state_updates = {"messages": [simulated_llm_response]}

    if "OK_READY_TO_ANSWER" in simulated_llm_response.content or state.get("is_planner_done"):
        state_updates["is_planner_done"] = True
        state_updates["plan_description"] = "LLM indicates it's ready to generate the final answer."
    else:
        state_updates["is_planner_done"] = False
        state_updates["plan_description"] = f"LLM plans to execute tools: {[tc['name'] for tc in simulated_llm_response.tool_calls if hasattr(simulated_llm_response, 'tool_calls') and simulated_llm_response.tool_calls]}"


    print(f"[PlannerAgentNode] LLM decision: {simulated_llm_response.content}, Tool Calls: {simulated_llm_response.tool_calls if hasattr(simulated_llm_response, 'tool_calls') else 'None'}")
    return state_updates


def answer_generator_node(state: BotState, llm: FakeListChatModel) -> Dict[str, Any]:
    print(f"\n[Node] AnswerGeneratorNode. Feedback: {state['feedback'].get('message','None')}")
    system_prompt = (
        "You are a helpful AI assistant. Your task is to provide a final, comprehensive answer to the user based on the entire conversation history, "
        "including all tool calls and their results, the user's original query, and any system feedback.\n"
        "Synthesize this information into a clear, concise, and helpful natural language response. "
        "If there were unresolvable errors (e.g., cycles, tool failures) that prevented fulfilling the request, explain the situation clearly."
    )
    messages_for_llm = create_llm_call_messages(state, system_prompt)
    
    # --- FAKE LLM RESPONSE LOGIC ---
    last_human_message = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "the user's request")
    simulated_llm_response_content = f"Okay, I've processed {last_human_message}. "

    if state.get("feedback") and state["feedback"].get("type") == "error":
        simulated_llm_response_content += f"However, there was an error: {state['feedback']['message']}. "
        if state["feedback"].get("details", {}).get("suggestion"):
            simulated_llm_response_content += f"Suggestion: {state['feedback']['details']['suggestion']}. "
    elif "tool_genpayload_users" in str(state["messages"]): # Example: if payload was generated
        payload_tool_msg = next((m for m in reversed(state["messages"]) if isinstance(m, ToolMessage) and m.tool_call_id == "tool_genpayload_users"), None)
        if payload_tool_msg:
             payload_content = json.loads(payload_tool_msg.content) # Assuming content is JSON string
             simulated_llm_response_content += f"I have generated a payload: {json.dumps(payload_content)}. "
    elif not state.get("execution_graph") and state.get("initialised"):
        simulated_llm_response_content += "The API graph could not be built. "
    elif state.get("execution_graph"):
        simulated_llm_response_content += f"The API graph with {len(state['execution_graph'])} endpoints is ready. "
    
    simulated_llm_response_content += "What would you like to do next?"

    # In a real LLM call:
    # llm_response = llm.invoke(messages_for_llm)
    # final_answer = llm_response.content
    final_answer = simulated_llm_response_content
    
    print(f"[AnswerGeneratorNode] Generated final response: {final_answer}")
    return {"final_response": final_answer, "messages": [AIMessage(content=final_answer)]} # Add final answer to messages

# === 5. System Initialisation Node (Non-LLM, uses tools) ===
def initialise_system_node(state: BotState) -> Dict[str, Any]:
    print(f"\n[Node] initialise_system_node. Spec: {state['openapi_spec']}")
    
    # This node now primarily ensures the state for `initialised` is set.
    # The PlannerAgentLLM will drive the actual calls to identify_apis, build_exec_graph, check_graph_for_cycles.
    # This node could do a very basic check or setup.
    
    # For this new LLM-centric flow, the LLM planner should actually initiate these calls.
    # So, this node might become simpler or be removed if the planner handles full init.
    # Let's assume this node just sets the stage if spec is present.
    if not state.get("openapi_spec") or state["openapi_spec"] == "<YOUR_SPEC_WITH_CYCLE>" or state["openapi_spec"] == "<VALID_SPEC>":
        # If spec is one of the test specs, or a real one is provided later by user.
        # The LLM planner will decide to call identify_apis etc.
        # This node primarily ensures 'initialised' flag context is right.
        # And sets up max_loops.
        print("[InitialiseSystemNode] System ready for LLM planner to potentially initialize graph if needed.")
        return {
            "initialised": True, # Indicates system itself is up, not necessarily the graph.
            "feedback": {},
            "is_planner_done": False, # Planner should start its work.
            "max_loops": state.get("max_loops", 10) # Set default max_loops if not present
        }
    else: # No valid spec to start with
        return {
            "initialised": False, # Cannot proceed
            "feedback": {"type": "error", "code": "NO_SPEC", "message": "OpenAPI spec not provided or invalid at startup."},
            "is_planner_done": True, # Cannot plan without spec
            "max_loops": state.get("max_loops", 10)
        }


# === 6. Router ===
# === 6. Router ===
def router_node(state: BotState) -> str:
    print(f"\n[Node] router. Initialised: {state.get('initialised')}. Planner Done: {state.get('is_planner_done')}. Feedback: {state.get('feedback',{}).get('message','None')}. Loop: {state.get('loop_counter',0)}/{state.get('max_loops',0)}")

    if not state.get("initialised", False):
        print("[Router] -> initialise_system_node")
        return "initialise_system_node"

    # Check last message for tool calls
    last_message = state["messages"][-1] if state["messages"] else None
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("[Router] Last message has tool calls -> system_tool_executor")
        return "system_tool_executor"

    if state.get("is_planner_done"):
        print("[Router] Planner is done -> answer_generator_node")
        return "answer_generator_node"
    
    if state.get("loop_counter", 0) > state.get("max_loops", 10) : # Safety break
        print("[Router] Max loops reached, planner should have error feedback -> answer_generator_node")
        return "answer_generator_node"

    print("[Router] -> planner_agent_node")
    return "planner_agent_node"

# === 7. Assemble Graph ===
# For the fake LLM, we instantiate it here. In a real app, it would be a proper LLM client.
fake_llm = FakeListChatModel(responses=[]) # Responses will be dynamically set/ignored by fake logic

graph_builder = StateGraph(BotState)
graph_builder.add_node("initialise_system_node", initialise_system_node)
# Pass the (fake) LLM to the nodes that need it.
graph_builder.add_node("planner_agent_node", lambda s: planner_agent_node(s, fake_llm))
graph_builder.add_node("system_tool_executor", system_tool_executor)
graph_builder.add_node("answer_generator_node", lambda s: answer_generator_node(s, fake_llm))

graph_builder.add_edge(START, "initialise_system_node")

routing_map = {
    "initialise_system_node": "initialise_system_node",
    "planner_agent_node": "planner_agent_node",
    "system_tool_executor": "system_tool_executor",
    "answer_generator_node": "answer_generator_node",
    END: END # Ensure END is a valid destination if router logic points to it
}
graph_builder.add_conditional_edges("initialise_system_node", router_node, routing_map)
graph_builder.add_conditional_edges("planner_agent_node", router_node, routing_map)
graph_builder.add_conditional_edges("system_tool_executor", router_node, routing_map)
# Answer generator is typically a terminal node for a given query flow
graph_builder.add_edge("answer_generator_node", END)


compiled_graph = graph_builder.compile().with_config(patch_config(None, recursion_limit=150)) # Increased recursion limit

# === 8. FastAPI WS stub ===
app = FastAPI()

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.headers.get("sec-websocket-key", "default_session")
    print(f"\n[WebSocket ({session_id})] New connection.")

    # Initial state for a new session
    current_session_state_template: BotState = {
        "messages": [],
        "openapi_spec": "<VALID_SPEC>", # Start with a valid spec for the LLM to init
        "api_details": {}, "execution_graph": {},
        "plan_description": None, "final_response": "",
        "is_planner_done": False, "feedback": {},
        "loop_counter": 0, "max_loops": 7, # Max 7 agent loops for this example
        "initialised": False
    }
    
    print(f"[WebSocket ({session_id})] Kicking off initial graph processing...")
    # The first invocation uses the template state. The graph starts from START.
    # LangGraph manages state per thread_id (session_id here).
    async for event_part in compiled_graph.astream(
        current_session_state_template, # This sets the initial state for the session
        config={"configurable": {"thread_id": session_id}}
    ):
        print(f"[WebSocket ({session_id}) Initial Stream] Node: {list(event_part.keys())[0]}, Data: {list(event_part.values())[0]}")
        # We don't need to manually update current_session_state here if using thread_id for persistence.
        # The `ainvoke(None, ...)` call later will retrieve the latest state for that thread_id.
        pass

    # Get the state after initial processing (which should run init_system and then planner_agent)
    current_session_state = await compiled_graph.ainvoke(None, config={"configurable": {"thread_id": session_id}})
    print(f"[WebSocket ({session_id})] State after initial graph run: Feedback='{current_session_state.get('feedback',{}).get('message','None')}', Initialised={current_session_state.get('initialised')}, PlannerDone={current_session_state.get('is_planner_done')}")

    # Send initial message if any (e.g., if LLM planner already decided to respond)
    if current_session_state.get("final_response"):
        await websocket.send_text(f"Initial: {current_session_state['final_response']}")
    elif current_session_state.get("feedback") and current_session_state["feedback"].get("type") == "error":
        fb = current_session_state["feedback"]
        await websocket.send_text(f"Initialization Info: {fb.get('message')}. Suggestion: {fb.get('details',{}).get('suggestion','')}")
    elif current_session_state.get("initialised"):
         await websocket.send_text("System initialized. How can I help you with the API graph today?")


    try:
        while True:
            text = await websocket.receive_text()
            print(f"\n[WebSocket ({session_id})] Received text: {text}")
            
            turn_input = {"messages": [HumanMessage(content=text)]}
            
            async for event_part in compiled_graph.astream(
                turn_input, # Input for the current turn
                config={"configurable": {"thread_id": session_id}} # Persist state for this session
            ):
                print(f"[WebSocket ({session_id}) Stream] Node: {list(event_part.keys())[0]}, Data: {list(event_part.values())[0]}")
                # Optionally send intermediate updates/thoughts to client here
                pass 

            current_session_state = await compiled_graph.ainvoke(None, config={"configurable": {"thread_id": session_id}})
            print(f"[WebSocket ({session_id})] State after turn: Feedback='{current_session_state.get('feedback',{}).get('message','None')}', Loop={current_session_state.get('loop_counter')}, PlannerDone={current_session_state.get('is_planner_done')}")

            if current_session_state.get("final_response"):
                await websocket.send_text(current_session_state["final_response"])
                # Potentially reset parts of state for next query, or let LLM handle context.
                # current_session_state["final_response"] = "" # Clear after sending if graph doesn't do it.
            elif not current_session_state.get("is_planner_done"):
                 await websocket.send_text(f"Processing '{text}'... (LLM Planner is working, loop {current_session_state.get('loop_counter')})")
            elif current_session_state.get("feedback") and current_session_state["feedback"].get("type") == "error":
                fb = current_session_state["feedback"]
                await websocket.send_text(f"Error: {fb.get('message')}. Suggestion: {fb.get('details',{}).get('suggestion','')}")
            else: # Should ideally always have a final_response if planner is_done
                 await websocket.send_text(f"Processing of '{text}' complete. Waiting for next instruction or final summary if applicable.")


    except WebSocketDisconnect:
        print(f"[WebSocket ({session_id})] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket ({session_id})] Error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.close(code=1011, reason=f"Internal server error: {str(e)[:100]}")
    finally:
        print(f"[WebSocket ({session_id})] Connection closed.")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://0.0.0.0:8000/ws")
    print("This version uses a FakeListChatModel to simulate LLM responses.")
    print("The LLM planner will attempt to initialize the graph using tools based on the hardcoded 'openapi_spec'.")
    print("Try sending messages like 'generate payload for /users' after initialization.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
Continue with Gemini Advanced
You've reached your limit on 2.5 Pro (experimental) until May 9, 8:08 am. Try Gemini Advanced for higher limits.

