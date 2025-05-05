import logging
import uuid
import json
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langgraph.checkpoint.memory import MemorySaver # Assuming you are still using MemorySaver
# If using a different checkpointer (e.g., SQL), import it here

# Assume build_graph, BotState, and initialize_llms are available
# We'll import from your existing files
from graph import build_graph
from models import BotState
# Import initialize_llms from main or define it here if main is only CLI
# For simplicity, let's replicate initialize_llms or ensure it's importable
# from main import initialize_llms # Assuming initialize_llms is in main.py and importable

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LLM Initialization (REPLACE PLACEHOLDER) ---
# Replicate or import initialize_llms.
# If you need to use environment variables, ensure python-dotenv is installed
# and you call `from dotenv import load_dotenv; load_dotenv()` early.
# from dotenv import load_dotenv
# load_dotenv() # Load environment variables from .env file

# Placeholder LLM class - replace with your actual LLM setup
class PlaceholderLLM:
    def invoke(self, prompt: str):
        logger.info(f"PlaceholderLLM invoked with prompt (first 100 chars): {prompt[:100]}...")
        # Simple logic to simulate responses based on prompt keywords
        prompt_lower = prompt.lower()
        if "classify the user's intent" in prompt_lower:
            if "openapi" in prompt_lower or "spec" in prompt_lower or "api documentation" in prompt_lower:
                 # Simulate router detecting spec-related intent
                 return '{"intent": "parse_openapi_spec"}'
            elif "what apis" in prompt_lower or "endpoints" in prompt_lower:
                 return '{"intent": "identify_apis"}'
            elif "generate payload" in prompt_lower:
                 return '{"intent": "generate_payloads"}'
            elif "create graph" in prompt_lower or "workflow" in prompt_lower:
                 return '{"intent": "generate_execution_graph"}'
            elif "describe graph" in prompt_lower or "show graph" in prompt_lower:
                 return '{"intent": "describe_graph"}'
            elif "answer question" in prompt_lower or "tell me about" in prompt_lower:
                 return '{"intent": "answer_openapi_query"}'
            else:
                 return '{"intent": "handle_unknown"}'
        elif "extract the essential information" in prompt_lower or "summarize" in prompt_lower:
            # Simulate parsing an OpenAPI spec - return a simple summary structure
            # This needs to be a valid JSON string if parse_llm_json_output_with_model is used
            # For parse_openapi_spec, the core_logic expects a string summary currently.
            # Let's simulate a string summary
            return "This is a simulated summary of the OpenAPI spec provided. It contains information about /users and /items endpoints."
        elif "identify relevant apis" in prompt_lower:
             return '{"api_list": ["get_users", "create_user"]}' # Example JSON response
        elif "generate example payloads" in prompt_lower:
             return '{"payloads": {"create_user": {"name": "test", "email": "test@example.com"}}}' # Example JSON
        elif "generate a directed acyclic graph" in prompt_lower:
            # Simulate a simple graph JSON structure
            return '{"graph": {"nodes": [{"operationId": "start_node"}], "edges": []}}' # Example JSON
        elif "describe the execution graph" in prompt_lower:
             return "This is a simulated description of the execution graph."
        else:
            # Default fallback for other prompts
            return "This is a simulated LLM response."


def initialize_llms():
    """
    Initializes and returns the router and worker LLM instances.
    Replace with your actual LLM setup (e.g., OpenAI, Google GenAI).
    """
    # --- REPLACE WITH YOUR ACTUAL LLM INSTANTIATION ---
    # Example using OpenAI:
    # from langchain_openai import ChatOpenAI
    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     raise ValueError("OPENAI_API_KEY environment variable not set.")
    # router_llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
    # worker_llm = ChatOpenAI(api_key=api_key, model="gpt-4o", temperature=0.5)

    # Using PlaceholderLLM for demonstration if no actual LLM is configured
    logger.warning("Using PlaceholderLLM. Replace with actual LLM implementation for production.")
    router_llm = PlaceholderLLM()
    worker_llm = PlaceholderLLM()
    # ---------------------------------------------------

    logger.info("LLMs initialized.")
    return {"router_llm": router_llm, "worker_llm": worker_llm}


# --- FastAPI App Setup ---
app = FastAPI()

# Serve a simple HTML page for testing the WebSocket connection
# Ensure you have a 'templates' directory with 'index.html'
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the HTML page for WebSocket testing."""
    return templates.TemplateResponse("index.html", {"request": request})

# Define the WebSocket endpoint
@app.websocket("/ws/submit_openapi")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Handles WebSocket connections for a specific session."""
    await websocket.accept()
    logger.info(f"WebSocket accepted connection from session: {session_id}")
    await handle_websocket_session(websocket, session_id)

async def handle_websocket_session(websocket: WebSocket, session_id: str):
    logger.info(f"WebSocket session started: {session_id}")

    # Initialize or load state for this session
    checkpointer = MemorySaver() # Or your persistent checkpointer
    checkpoint = checkpointer.get_state(session_id)
    if checkpoint:
        logger.info(f"Loaded existing state for session {session_id}")
        initial_state = BotState(**checkpoint.state)
    else:
        logger.info(f"Initializing new state for session {session_id}")
        initial_state = BotState()

    # Build the graph (ensure LLMs are initialized)
    # Assuming initialize_llms is defined or imported
    llms = initialize_llms()
    router_llm = llms["router_llm"]
    worker_llm = llms["worker_llm"]
    app = build_graph(router_llm, worker_llm, checkpointer)

    try:
        while True:
            try:
                # --- ADDED LOG ---
                logger.info(f"Waiting for message on session {session_id}...")
                input_data = await websocket.receive_json()
                # --- ADDED LOG ---
                logger.info(f"Received message on session {session_id}: {input_data}")

                # Prepare the state for the graph invocation
                state_config: Dict[str, Any] = {"current_step_input": input_data.get("query", "")}

                # Check if input is an OpenAPI spec
                input_is_spec = False
                if "spec" in input_data and isinstance(input_data["spec"], str):
                    state_config["openapi_spec_string"] = input_data["spec"]
                    input_is_spec = True
                    logger.info(f"Detected OpenAPI spec in input for session {session_id}.")
                elif "spec" in input_data and input_data["spec"] is not None:
                     # Handle cases where 'spec' might be sent but not a string (e.g., null)
                     logger.warning(f"Received 'spec' in input for session {session_id} but it's not a string. Ignoring spec.")


                # Add other relevant input to state_config if needed
                # For example, you might want to add 'input_is_spec' itself to state
                state_config["input_is_spec"] = input_is_spec # <-- Add this to state_config

                # --- ADDED LOG ---
                logger.info(f"Invoking graph for session {session_id} with state_config: {state_config}")
                # Invoke the graph with the current state and input
                async for event in app.astream(state_config, config={"configurable": {"thread_id": session_id}}):
                    for key, value in event.items():
                        if key != "__end__":
                            # --- ADDED LOG INSIDE THE LOOP ---
                            logger.info(f"Received event from graph for session {session_id} - Node: {key}")
                            # Process graph output (e.g., send to user)
                            if key == "responder" and value and value.get('final_response'):
                                final_response = value.get('final_response')
                                logger.info(f"Sending final response for session {session_id}: {final_response[:100]}...")
                                await websocket.send_json({"type": "final_response", "content": final_response})
                            # You might want to send other intermediate updates too
                            # else:
                            #     logger.debug(f"Ignoring intermediate event from node {key}")

                logger.info(f"Graph execution completed for session {session_id}")

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received on session {session_id}")
                await websocket.send_json({"type": "error", "content": "Invalid JSON format received."})
            except Exception as e:
                 # --- ADDED LOG ---
                logger.critical(f"Error during graph processing for session {session_id}: {e}", exc_info=True)
                await websocket.send_json({"type": "error", "content": f"An error occurred during processing: {e}"})
                # Decide if you want to close the connection on error or continue
                # await websocket.close(code=1011) # Example: Close on internal error
                # break # Example: Break loop on error

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        # Catch any other exceptions in the WebSocket loop itself
         # --- ADDED LOG ---
        logger.critical(f"Unexpected error in WebSocket loop for session {session_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "content": f"An unexpected error occurred: {e}"})
        except:
            pass # Ignore errors sending error message
    finally:
        # Ensure WebSocket is closed if not already
        try:
            await websocket.close()
        except:
            pass # Ignore errors closing


# --- How to Run ---
# Save this code as main.py (or api.py)
# Make sure you have FastAPI, uvicorn, and your other dependencies installed:
# pip install fastapi uvicorn websockets jinja2 # jinja2 only if you use the HTML template
#
# Run from your terminal:
# uvicorn main:app --reload # Use main:app if filename is main.py
#
# Then connect to ws://localhost:8000/ws/submit_openapi from a WebSocket client
# (like a simple HTML page or a WebSocket testing tool).
