# graph.py

from typing import TypedDict, Annotated, List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# === 1. State schema with reducers ===
class BotState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    openapi_spec: str
    api_details: Dict[str, Any]
    execution_graph: Dict[str, Any]
    payloads_version: int
    plan: List[Dict[str, Any]]
    plan_payloads_version: int
    final_response: str
    done: bool
    feedback: str
    loop_counter: int

# === 2. Define your tools ===
@tool("identify_apis", description="Extract endpoints from spec")
def identify_apis(spec: str) -> Dict[str, Any]:
    ...

@tool("build_exec_graph", description="Compute execution graph from api_details")
def build_exec_graph(api_details: Dict[str, Any]) -> Dict[str, Any]:
    ...

@tool("check_cycle", description="Return True if graph has a cycle")
def check_cycle(graph: Dict[str, Any]) -> bool:
    ...

@tool("generate_payload", description="Generate payloads for endpoints")
def generate_payload(filter: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    ...

@tool("add_edge", description="Add a dependency edge")
def add_edge(graph: Dict[str, Any], frm: str, to: str) -> Dict[str, Any]:
    ...

@tool("answer_query", description="Answer user using current state")
def answer_query(query: str, execution_graph: Dict[str, Any], payloads_version: int) -> str:
    ...

# Bundle into ToolNode
system_tools = ToolNode([
    identify_apis, build_exec_graph, check_cycle,
    generate_payload, add_edge, answer_query
])

# === 3. Planner node ===
def planner(state: BotState) -> Dict[str, Any]:
    # bump loop counter
    state["loop_counter"] += 1

    # replan if payloads changed
    if state["payloads_version"] != state.get("plan_payloads_version"):
        state["plan"] = []
        state["done"] = False
        state["feedback"] = "Payload changed"
        return {"feedback": state["feedback"], "plan": state["plan"], "done": False}

    # otherwise generate plan if not done
    if not state["done"]:
        plan_msg = AIMessage(
            content="plan",
            tool_calls=[{
                "name": "llm_plan_tool",
                "args": {"graph": state["execution_graph"], "query": state["messages"][-1].content},
                "id": "plan_1",
                "type": "tool_call"
            }]
        )
        out = system_tools.invoke({"messages": [plan_msg]})
        state["plan"] = out["messages"][0].content
        state["plan_payloads_version"] = state["payloads_version"]
        state["done"] = True
        state["feedback"] = ""
        return {"plan": state["plan"], "done": True, "plan_payloads_version": state["plan_payloads_version"]}
    return {}

# === 4. Responder node ===
def responder(state: BotState) -> Dict[str, Any]:
    ans_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "answer_query",
            "args": {
                "query": state["messages"][-1].content,
                "execution_graph": state["execution_graph"],
                "payloads_version": state["payloads_version"]
            },
            "id": "ans_1",
            "type": "tool_call"
        }]
    )
    out = system_tools.invoke({"messages": [ans_msg]})
    state["final_response"] = out["messages"][0].content
    return {"final_response": state["final_response"]}

# === 5. Initialise node (runs once) ===
def initialise(state: BotState) -> Dict[str, Any]:
    # parse spec → identify_apis → build_exec_graph → check_cycle
    init_calls = [
        {"name":"identify_apis","args":{"spec":state["openapi_spec"]},"id":"i1","type":"tool_call"},
        {"name":"build_exec_graph","args":{"api_details":{} },"id":"i2","type":"tool_call"},
        {"name":"check_cycle","args":{"graph":state["execution_graph"]},"id":"i3","type":"tool_call"}
    ]
    msg = AIMessage(content="", tool_calls=init_calls)
    out = system_tools.invoke({"messages":[msg]})
    # unpack results
    for m in out["messages"]:
        if isinstance(m, ToolMessage):
            if m.tool.name=="identify_apis": state["api_details"]=m.content
            if m.tool.name=="build_exec_graph": state["execution_graph"]=m.content
            if m.tool.name=="check_cycle" and m.content:
                state["feedback"]="Cycle detected on init"
    state["payloads_version"]=0
    return {
        "api_details": state["api_details"],
        "execution_graph": state["execution_graph"],
        "payloads_version": state["payloads_version"],
        "feedback": state["feedback"]
    }

# === 6. Router ===
def router(state: BotState) -> str:
    # run initialise once
    if not state.get("initialised", False):
        return "initialise"
    # if user called any tool (payload, add_edge), run system_tools
    if state["messages"][-1].tool_calls:
        return "system_tools"
    # if planner needs to replan
    if not state["done"] or state.get("feedback"):
        return "planner"
    # otherwise answer
    return "responder"

# === 7. Assemble graph ===
graph = StateGraph(BotState)
graph.add_node("initialise", initialise)                     # START → initialise
graph.add_node("system_tools", system_tools.invoke)          # ToolNode step
graph.add_node("planner", planner)                           # plan generation
graph.add_node("responder", responder)                       # final answer
graph.add_edge(START, "initialise")                          # entrypoint :contentReference[oaicite:3]{index=3}
graph.add_edge("initialise", "planner")                      # then plan :contentReference[oaicite:4]{index=4}
graph.add_edge("planner", "responder")                       # plan → respond :contentReference[oaicite:5]{index=5}
graph.add_edge("system_tools", "planner")                    # after any tools, replan :contentReference[oaicite:6]{index=6}
graph.add_conditional_edges("planner", router)               # dynamic routing :contentReference[oaicite:7]{index=7}
graph.add_conditional_edges("responder", router)             # allow follow‑ups :contentReference[oaicite:8]{index=8}
compiled = graph.compile().with_config(patch_config(None, recursion_limit=50))

# === 8. FastAPI WS stub ===
app = FastAPI()
@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    state: BotState = {
        "messages": [], "openapi_spec": "<YOUR_SPEC>", "api_details": {}, "execution_graph": {},
        "payloads_version":0, "plan":[], "plan_payloads_version":-1,
        "final_response":"", "done":False, "feedback":"", "loop_counter":0, "initialised":False
    }
    try:
        while True:
            text = await ws.receive_text()
            state["messages"].append(AIMessage(content=text))
            out = compiled.invoke(state)
            if state["final_response"]:
                await ws.send_text(state["final_response"])
    except WebSocketDisconnect:
        pass



**Explaining payload versioning**

The user asks why we maintain payload versions instead of triggering the next tool automatically on payload changes. Payload versioning is useful because it allows the planner to detect changes and avoid using stale versions. Without versioning, handling changes might be tricky, especially if conditional routing is involved. Versioning provides structure and ensures data integrity. I’ll explain conceptually why this is important, but I don’t think a web search is necessary here. This is more about development practices.


Thought for 6 seconds


## Summary

You don’t actually need an explicit **payloads\_version** counter if you instead detect “payload change” by observing new `tool_calls` or by comparing the actual payload data structures in state. By relying on LangGraph’s built‑in **ToolNode** and **conditional edges**, any time the user invokes a payload‑generation tool (or any other tool), the graph will automatically re‑run the dependent steps—no manual versioning required.

---

## 1. Detecting Payload Changes via Tool Calls

### 1.1 Tool Calling Mechanism

When the LLM issues a `tool_call` for `generate_payload`, that call is recorded on the last `AIMessage.tool_calls`. You can simply route on the presence of those calls rather than tracking a numeric version ([Introduction | 🦜️🔗 LangChain][1]).

### 1.2 Conditional Edges on Tool Calls

Use `add_conditional_edges` with a router that checks:

```python
def router(state):
    # If the latest message has any tool_calls, run the ToolNode
    if state["messages"][-1].tool_calls:
        return "system_tools"
    # Otherwise proceed to planner or responder
    …
```

This guarantees that **any** payload‑generation call immediately re‑enters the `system_tools` node, which invokes `generate_payload` and then its downstream tools (`build_exec_graph`, `check_cycle`, `add_edge`) automatically ([GitHub][2]).

---

## 2. Removing Payload Version from State

### 2.1 Simplified State Schema

Instead of:

```python
payloads_version: int
plan_payloads_version: int
```

You can omit those entirely. Your state only needs:

```python
class BotState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    execution_graph: Dict[str, Any]
    plan: List[Dict[str,Any]]
    done: bool
    feedback: str
    loop_counter: int
    # no payloads_version needed
}
```

State fields are only updated by tool outputs, so change detection comes from new `ToolMessage`s ([GitHub][3]).

---

## 3. Revised Planner Logic

```python
def planner(state: BotState) -> Dict[str,Any]:
    # If we flagged feedback (missing/cycle), clear plan
    if state.get("feedback"):
        state["plan"] = []
        state["done"] = False
        return {"plan":state["plan"], "done":False}

    # If no plan yet, ask LLM to plan
    if not state["done"]:
        plan_msg = AIMessage(
            content="plan",
            tool_calls=[{
                "name":"llm_plan_tool",
                "args":{"graph":state["execution_graph"], "query":state["messages"][-1].content},
                "id":"plan1","type":"tool_call"
            }]
        )
        out = system_tools.invoke({"messages":[plan_msg]})
        state["plan"] = out["messages"][0].content
        state["done"] = True
        state["feedback"] = ""
        return {"plan":state["plan"], "done":True}
    return {}
```

* **No version checks**: The planner simply re‑plans when `feedback` is set.
* **Payload changes** set `feedback` via the router/ToolNode step, ensuring replanning.

---

## 4. Ensuring Dependent Tools Run

Bundle all related tools in one `ToolNode` so that after `generate_payload` executes, **ToolNode** will immediately run any other queued tools (e.g. `build_exec_graph`, `check_cycle`, `add_edge`) before returning control ([Medium][4]).

```python
system_tools = ToolNode([
    identify_apis, build_exec_graph, check_cycle,
    generate_payload, add_edge, answer_query
])
```

* If the user’s tool call chain is `generate_payload` → `add_edge`, ToolNode runs them sequentially (or in parallel) and updates state accordingly ([GitHub][2]).

---

## 5. Router Drives the Flow

```python
def router(state: BotState) -> str:
    # 1. Handle any new tool_calls (payload, add_edge, etc.)
    if state["messages"][-1].tool_calls:
        return "system_tools"
    # 2. If plan needs refresh (feedback set), replan
    if not state["done"] or state.get("feedback"):
        return "planner"
    # 3. Otherwise answer
    return "responder"
```

* **Step 1** ensures payload or edge‑addition calls immediately re‑enter `system_tools`.
* **Step 2** picks up any `feedback` from cycle/missing‑API detection or from your payload tool logic to re‑plan.
* **Step 3** goes to the responder once state is stable ([Nordic Semiconductor Documentation][5]).

---

## 6. Why This Is Better Than Versioning

| Versioning                               | Tool‑Call Detection                                                                 |
| ---------------------------------------- | ----------------------------------------------------------------------------------- |
| Adds extra counters to state             | Relies on natural tool\_call metadata in `messages` ([Home Assistant Community][6]) |
| Requires planner to compare versions     | Router simply checks `tool_calls`, no manual compare                                |
| Risk of drift if version not incremented | Always accurate: presence of a tool\_call implies change                            |
| More code to maintain                    | Simpler, leverages LangGraph’s built‑in mechanics                                   |

By removing **payloads\_version**, you reduce complexity and let LangGraph’s message‑based routing handle all state changes—including payload updates, edge additions, and graph rebuilds—seamlessly.

[1]: https://python.langchain.com/docs/concepts/tool_calling/?utm_source=chatgpt.com "Tool calling | 🦜️ LangChain"
[2]: https://github.com/langchain-ai/langgraph/discussions/2806?utm_source=chatgpt.com "langgraph/how-tos/update-state-from-tools/ #2806 - GitHub"
[3]: https://github.com/langchain-ai/langgraph/discussions/1306?utm_source=chatgpt.com "langgraph/how-tos/state-model/ #1306 - GitHub"
[4]: https://medium.com/%40vivekvjnk/introduction-to-tool-use-with-langgraphs-toolnode-0121f3c8c323?utm_source=chatgpt.com "Introduction to Tool Use with LangGraph's ToolNode - Medium"
[5]: https://docs.nordicsemi.com/bundle/ncs-latest/page/matter/chip_tool_guide.html?utm_source=chatgpt.com "Working with the CHIP Tool - Technical Documentation"
[6]: https://community.home-assistant.io/t/detecting-state-change-with-node-red/44556?utm_source=chatgpt.com "Detecting state change with Node-Red? - Home Assistant Community"



