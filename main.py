import logging
import uuid
import json # Added for debug logging
from typing import Any, Dict

# Assume graph.py contains build_graph
from graph import build_graph
# Assume models.py contains BotState
from models import BotState
# Assume utils.py has save_state, load_state (though not explicitly used in loop if checkpointer handles it)
# from utils import save_state, load_state

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Placeholder LLM Initialization ---
# Replace with your actual LLM setup (e.g., using LangChain, OpenAI API, etc.)
# Ensure the LLMs have an 'invoke' method compatible with llm_call_helper in utils.py

class PlaceholderLLM:
    """A dummy LLM for demonstration purposes."""
    def __init__(self, name="PlaceholderLLM"):
        self.name = name
        logger.warning(f"Using {self.name}. Replace with actual LLM implementation.")

    def invoke(self, prompt: Any, **kwargs) -> Any:
        """Simulates LLM invocation. Returns predefined responses based on prompt hints."""
        prompt_str = str(prompt) # Handle both string and message list inputs
        logger.info(f"{self.name} received prompt hint: {prompt_str[:150]}...") # Log hint

        # --- Router LLM Simulation ---
        if "Determine the most appropriate next action" in prompt_str:
            if "parse" in prompt_str.lower() or "openapi spec" in prompt_str.lower() or "swagger" in prompt_str.lower():
                return "parse_openapi_spec"
            elif "generate graph" in prompt_str.lower() or "execution plan" in prompt_str.lower() or "workflow" in prompt_str.lower():
                 return "generate_execution_graph"
            elif "payload" in prompt_str.lower():
                 return "generate_payloads"
            elif "add edge" in prompt_str.lower():
                 return "add_graph_edge"
            elif "validate" in prompt_str.lower():
                 return "validate_graph"
            elif "describe" in prompt_str.lower():
                 return "describe_graph"
            elif "show graph" in prompt_str.lower() or "get json" in prompt_str.lower():
                 return "get_graph_json"
            elif "what all apis" in prompt_str.lower() or "identify apis" in prompt_str.lower(): # Added hint for identify_apis
                 return "identify_apis"
            else:
                return "handle_unknown"
        elif "Extract parameters" in prompt_str:
             # Simulate parameter extraction (very basic)
             params = {}
             if "add_graph_edge" in prompt_str:
                  # Look for simple patterns like "add edge A to B"
                  import re
                  match = re.search(r'add edge (\w+) to (\w+)', prompt_str, re.IGNORECASE)
                  if match:
                       params = {"from_node": match.group(1), "to_node": match.group(2)}
                  else:
                       # Fallback if pattern not found
                       params = {"from_node": "example_source", "to_node": "example_target", "description": "Extracted from fallback"}
             elif "generate_payloads" in prompt_str:
                  params = {"instructions": "Generate default example payloads."}
             elif "generate_execution_graph" in prompt_str:
                  params = {"goal": "Generate a standard workflow."}

             logger.info(f"Simulating parameter extraction: {params}")
             return json.dumps(params) # Return as JSON string

        # --- Worker LLM Simulation ---
        elif "Parse the following OpenAPI specification" in prompt_str:
            # Simulate parsing - return a minimal valid OpenAPI structure
            logger.info("Simulating OpenAPI parsing...")
            return """
            {
              "openapi": "3.0.0",
              "info": { "title": "Simulated API", "version": "1.0.0" },
              "paths": {
                "/items": {
                  "get": { "operationId": "listItems", "summary": "List all items" },
                  "post": { "operationId": "createItem", "summary": "Create a new item" }
                },
                "/items/{itemId}": {
                  "get": { "operationId": "getItem", "summary": "Get item by ID" },
                  "put": { "operationId": "updateItem", "summary": "Update item by ID" },
                   "parameters": [{"name": "itemId", "in": "path", "required": true, "schema": {"type": "string"}}]
                }
              }
            }
            """
        elif "identify the key API endpoints" in prompt_str:
             logger.info("Simulating API identification...")
             return """
             [
               { "operationId": "listItems", "summary": "List all items", "method": "get", "path": "/items" },
               { "operationId": "createItem", "summary": "Create a new item", "method": "post", "path": "/items" },
               { "operationId": "getItem", "summary": "Get item by ID", "method": "get", "path": "/items/{itemId}" },
               { "operationId": "updateItem", "summary": "Update item by ID", "method": "put", "path": "/items/{itemId}" }
             ]
             """
        elif "generate example request payloads" in prompt_str:
             logger.info("Simulating payload generation...")
             # Return payloads for the identified APIs from the simulation above
             return """
             {
               "listItems": null,
               "createItem": { "name": "New Item Name", "value": 100 },
               "getItem": null,
               "updateItem": { "name": "Updated Item Name", "value": 150 }
             }
             """
        elif "Generate an API execution workflow graph" in prompt_str:
             logger.info("Simulating graph generation...")
             return """
             {
               "nodes": [
                 { "operationId": "createItem", "summary": "Create a new item", "example_payload": { "name": "New Item Name", "value": 100 } },
                 { "operationId": "getItem", "summary": "Get item by ID", "example_payload": null },
                 { "operationId": "updateItem", "summary": "Update item by ID", "example_payload": { "name": "Updated Item Name", "value": 150 } }
               ],
               "edges": [
                 { "from_node": "createItem", "to_node": "getItem", "description": "Use ID from create response" },
                 { "from_node": "getItem", "to_node": "updateItem", "description": "Use ID from getItem path" }
               ],
               "description": "Workflow: Create an item, then get its details, then update it."
             }
             """
        elif "provide a concise, natural language description" in prompt_str:
             logger.info("Simulating graph description...")
             return "This workflow involves creating an item, retrieving it using its ID, and then updating it."
        elif "formulate a polite response acknowledging the input" in prompt_str:
             logger.info("Simulating unknown intent response...")
             return "I'm sorry, I couldn't determine a specific action from your request. Could you clarify? I can help parse specs, generate/modify graphs, and generate payloads."

        # Default fallback for unknown prompts
        logger.warning(f"{self.name} received unhandled prompt type.")
        return f"Placeholder response from {self.name} for prompt: {prompt_str[:100]}..."


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting OpenAPI LLM Assistant...")

    # Initialize placeholder LLMs (or your actual LLMs)
    router_llm = PlaceholderLLM("RouterLLM")
    worker_llm = PlaceholderLLM("WorkerLLM")

    # Build the LangGraph application
    try:
        app = build_graph(router_llm, worker_llm)
        logger.info("Graph built successfully.")
    except Exception as e:
        logger.critical(f"Failed to build graph: {e}", exc_info=True)
        exit(1)

    # --- Interactive Loop ---
    session_id = str(uuid.uuid4()) # Generate a unique session ID for file state (if used)
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
        # *** Use 'thread_id' as the key for MemorySaver ***
        config = {"configurable": {"thread_id": session_id}}

        current_input = {"user_input": user_input,"session_id": session_id }

        try:
            # Stream events from the graph execution
            final_state = None
            print("\nAssistant:", end=" ", flush=True)

            # Use stream to get intermediate steps and final result
            events = app.stream(current_input, config=config, stream_mode="values")
            for event in events:
                 # The event itself is the full state dictionary at that point
                 # We are interested in the 'response' field of the *last* state update
                 final_state = event # Keep track of the latest state
                 # Optional: Print intermediate node names for debugging
                 # last_node = list(event.keys())[-1] # Get the last node that ran
                 # logger.debug(f"[Debug] Node '{last_node}' finished.")


            # Process the final state
            if final_state and isinstance(final_state, dict) and 'response' in final_state:
                 print(f"{final_state['response']}")
                 # Log final state details for debugging
                 logger.debug(f"Final state for session/thread {session_id}: {json.dumps(final_state, indent=2, default=str)}")
            else:
                 print("Sorry, something went wrong, and I don't have a response.")
                 logger.error(f"Graph execution finished without a valid final state or response. Last state: {final_state}")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            logger.critical(f"Error during graph execution: {e}", exc_info=True)
            # Optionally break the loop or try to recover

    print("\nSession ended. Goodbye!")

 