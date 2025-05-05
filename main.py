# filename: main.py
import logging
import uuid
import json
import os
from typing import Any, Dict, Optional # Import Optional

from graph import build_graph
from models import BotState # Ensure BotState is imported

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- LLM Initialization (NEEDS ACTUAL IMPLEMENTATION) ---
# ... (keep initialize_llms as is or replace with your actual LLM setup)
def initialize_llms():
    """
    Initializes and returns the router and worker LLM instances.
    Replace this with your actual LLM setup.
    Ensure API keys are handled securely (e.g., environment variables).
    """
    logger.warning("Initializing LLMs - REPLACE PLACEHOLDER LOGIC.")

    # Example using environment variables for API keys
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # google_api_key = os.getenv("GOOGLE_API_KEY")
    # if not openai_api_key:
    #     logger.error("OPENAI_API_KEY environment variable not set.")
    #     # Handle error appropriately - exit or raise exception
    # if not google_api_key:
    #     logger.error("GOOGLE_API_KEY environment variable not set.")
    #     # Handle error appropriately

    try:
        # Replace with your actual LLM instantiation
        # Example:
        # router_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
        # worker_llm = ChatOpenAI(model="gpt-4", temperature=0.1, api_key=openai_api_key)
        # or
        # worker_llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1, google_api_key=google_api_key)

        # Using PlaceholderLLM for now to allow code execution without real keys
        # REMOVE THIS IN YOUR ACTUAL IMPLEMENTATION
        class PlaceholderLLM:
            def __init__(self, name="PlaceholderLLM"): self.name = name
            def invoke(self, prompt: Any, **kwargs) -> Any:
                logger.warning(f"Using {self.name}. Needs replacement.")
                # Simplified simulation logic from original main.py
                # This placeholder won't perfectly simulate the responses needed
                # for the new intermediate messages, but allows the code structure to run.
                prompt_str = str(prompt)
                if "Determine the most appropriate next action" in prompt_str: return "handle_unknown"
                if "Parse the following OpenAPI specification" in prompt_str:
                    # Simulate setting a response during parsing
                    return '{"response": "Simulating: Parsed OpenAPI spec.", "openapi": "3.0.0", "info": {"title": "Simulated", "version": "1.0"}, "paths": {}}'
                if "identify the key API endpoints" in prompt_str:
                     return '{"response": "Simulating: Identified APIs." , "identified_apis": [{"operationId": "simulatedOp", "method": "get", "path": "/simulated"}]}'
                if "describe example request payloads" in prompt_str:
                     return '{"response": "Simulating: Described payloads.", "payload_descriptions": {"simulatedOp": "Simulated payload description"}}'
                if "Generate a description of an API execution workflow graph" in prompt_str:
                     return '{"response": "Simulating: Generated graph description." , "nodes": [{"operationId": "simulatedOp", "payload_description": "...", "input_mappings": []}], "edges": [], "description": "Simulated graph description."}'
                if "provide a concise, natural language description of the workflow" in prompt_str:
                     return '{"response": "Simulating: Described workflow.", "description": "Simulated workflow description."}'
                if "question" in prompt_str:
                     return '{"response": "Simulating: Answered query."}'


                return f"Placeholder response from {self.name}"

            # Add necessary methods for LangChain integration if your actual LLM requires them
            # e.g., _call, _acall, ainvoke, batch, abatch
            # For basic invoke, we only need invoke.
        router_llm = PlaceholderLLM("RouterLLM")
        worker_llm = PlaceholderLLM("WorkerLLM")
        # --- END OF PLACEHOLDER ---

        # Validate that the LLMs have the required 'invoke' method
        if not hasattr(router_llm, 'invoke') or not hasattr(worker_llm, 'invoke'):
            raise TypeError("Initialized LLMs must have an 'invoke' method.")

        logger.info("LLM clients initialized (using placeholders - replace!).")
        return router_llm, worker_llm


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting OpenAPI LLM Assistant...")

    # Initialize LLMs
    try:
        router_llm, worker_llm = initialize_llms()
    except Exception:
        exit(1) # Exit if LLMs can't be initialized

    # Build the LangGraph application
    try:
        app = build_graph(router_llm, worker_llm)
        logger.info("Graph built successfully.")
    except Exception as e:
        logger.critical(f"Failed to build graph: {e}", exc_info=True)
        exit(1)

    # --- Interactive Loop ---
    session_id = str(uuid.uuid4()) # Generate a unique session ID
    print(f"\nStarting new session: {session_id}")
    print("Enter your OpenAPI spec, questions, or commands. Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        if not user_input.strip():
            continue

        # Prepare the input state for the graph
        # The checkpointer handles loading/merging previous state for the session
        config = {"configurable": {"thread_id": session_id}}
        current_input = {"user_input": user_input, "session_id": session_id}

        try:
            # Stream events from the graph execution
            final_state_snapshot = None
            print("\nAssistant:") # Start assistant output line

            # Use stream to get intermediate steps and final result
            # stream_mode="values" yields the full state object after each node completes
            # We will now process these intermediate states
            events = app.stream(current_input, config=config, stream_mode="values")

            # Process events to print intermediate responses
            for intermediate_state in events:
                 # Check if the node set a response message
                 if isinstance(intermediate_state, dict):
                     response_message = intermediate_state.get("response")
                     # Print intermediate messages as they appear
                     if response_message:
                         print(f"  - {response_message}")
                         # Clear the response field in the state copy to avoid re-printing
                         # Note: This doesn't modify the actual state passed to the next node
                         intermediate_state["response"] = None # Clear message after displaying

                 # Keep track of the latest state snapshot
                 final_state_snapshot = intermediate_state

            # After the stream completes, the final response is in the last state snapshot
            if final_state_snapshot and isinstance(final_state_snapshot, dict):
                 final_response = final_state_snapshot.get("final_response")
                 if final_response:
                     print(f"\nFinal Result: {final_response}") # Indicate final result
                 else:
                     # Fallback if 'final_response' isn't set (e.g., error before responder)
                     # We already printed intermediate responses, so maybe just a generic message
                     print("\nProcessing finished.")
                     logger.warning(f"Graph execution finished, but 'final_response' was empty in the final state.")


                 # Log final state details for debugging
                 logger.debug(f"Final state for session/thread {session_id}: {json.dumps(final_state_snapshot, indent=2, default=str)}")
            else:
                 print("\nSorry, something went wrong, and I don't have a valid final state.")
                 logger.error(f"Graph execution finished without a valid final state dictionary. Last event: {final_state_snapshot}")

        except Exception as e:
            print(f"\nAn error occurred during graph execution: {e}")
            logger.critical(f"Error during graph stream/execution: {e}", exc_info=True)
            # Optionally break the loop or try to recover

    print("\nSession ended. Goodbye!")


async def handle_websocket_message(websocket, path):
    session_id = # Get or create session ID for the user

    async for message in websocket:
        user_input = message
        config = {"configurable": {"thread_id": session_id}}
        current_input = {"user_input": user_input, "session_id": session_id}

        try:
            async for intermediate_state in app.astream(current_input, config=config, stream_mode="values"): # Use astream for async
                 # Process each state update
                 intermediate_response = intermediate_state.get("response")
                 if intermediate_response:
                     # Send intermediate message to user via WebSocket
                     await websocket.send(json.dumps({"type": "intermediate_message", "content": intermediate_response}))

                 # Keep track of the last state
                 final_state_snapshot = intermediate_state

            # After the stream finishes, process the final state
            final_response = final_state_snapshot.get("final_response")
            if final_response:
                 # Send the final response message
                 await websocket.send(json.dumps({"type": "final_response", "content": final_response}))
            else:
                 # Handle cases where final_response wasn't set (e.g., error)
                 await websocket.send(json.dumps({"type": "error", "content": "Processing finished without a final response."}))

            # For specific queries like "list apis", you might need additional logic here
            # Example: Check user_input and look at final_state_snapshot.get("identified_apis")
            # This requires knowing the user's original intent *after* the graph completes.
            # A simpler approach is to let answer_openapi_query format the list into final_response.


        except Exception as e:
            error_message = f"An error occurred: {e}"
            print(f"Error: {e}")
            await websocket.send(json.dumps({"type": "error", "content": error_message}))
