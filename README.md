# open_api_execution_gen_ai

Prerequisites:

Python: Ensure you have Python installed (version 3.8 or later is recommended).
Install Libraries: You'll need to install the required libraries. Open your terminal or command prompt and run:
Bash

pip install pydantic tenacity langgraph langchain # Or langchain-openai, langchain-core depending on your LLM setup
# Add any specific library for your chosen LLM (e.g., openai, azure-identity, google-cloud-aiplatform)
# Example for OpenAI:
# pip install openai
# Example for Azure OpenAI:
# pip install openai azure-identity
Note: jsonschema is used internally by Pydantic, so it should be installed as a dependency.
Save the Code:

Create a new directory for your project (e.g., openapi_llm_assistant).
Inside this directory, save each code block I provided into its corresponding file name:
models.py (from the first code block)
utils.py (from the second code block)
core_logic.py (from the third code block)
router.py (from the fourth code block)
graph.py (from the fifth code block)
main.py (from the sixth code block)
Replace Placeholder LLMs:

This is the most crucial step. Open main.py.

Find the PlaceholderLLM class and the lines where router_llm and worker_llm are initialized:

Python

# --- Placeholder LLM Initialization ---
# Replace with your actual LLM setup...
# ... (PlaceholderLLM class definition) ...

# Initialize placeholder LLMs
router_llm = PlaceholderLLM("RouterLLM")
worker_llm = PlaceholderLLM("WorkerLLM")
Delete the PlaceholderLLM class definition.

Replace the initialization lines with your actual LLM client setup. Make sure the LLM objects you create have an .invoke() method that's compatible with the llm_call_helper function in utils.py.

Example using LangChain with OpenAI:

Python

# main.py (replace placeholder section)
import os
from langchain_openai import ChatOpenAI

# Load your API keys (best practice is environment variables)
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# Initialize actual LLMs
try:
    # You might use different models or configurations for router vs worker
    router_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    worker_llm = ChatOpenAI(model="gpt-4", temperature=0.2) # Maybe a more powerful model for complex tasks
    logger.info("Initialized OpenAI LLMs.")
except Exception as e:
    logger.critical(f"Failed to initialize LLMs: {e}. Check API keys and library setup.", exc_info=True)
    exit(1)

# --- Main Execution ---
# ... (rest of the main.py code) ...
Example using LangChain with Azure OpenAI:

Python

# main.py (replace placeholder section)
import os
from langchain_openai import AzureChatOpenAI

# Load your Azure credentials (best practice is environment variables)
# os.environ["AZURE_OPENAI_API_KEY"] = "your_azure_api_key"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "your_azure_endpoint"
# os.environ["OPENAI_API_VERSION"] = "your_api_version" # e.g., "2023-07-01-preview"

# Initialize actual LLMs
try:
    # Specify your deployment names
    router_llm = AzureChatOpenAI(
        deployment_name="your_router_deployment_name", # e.g., gpt-35-turbo
        temperature=0.0
    )
    worker_llm = AzureChatOpenAI(
        deployment_name="your_worker_deployment_name", # e.g., gpt-4
        temperature=0.2
    )
    logger.info("Initialized Azure OpenAI LLMs.")
except Exception as e:
    logger.critical(f"Failed to initialize LLMs: {e}. Check API keys, endpoint, version, deployment names, and library setup.", exc_info=True)
    exit(1)

# --- Main Execution ---
# ... (rest of the main.py code) ...
Run the Script:

Navigate to your project directory (openapi_llm_assistant) in your terminal.
Execute the main.py script:
Bash

python main.py
Interact:

The script will start, initialize the LLMs (the real ones now!), build the graph, and print:
Starting new session: <some-uuid>
Enter your OpenAPI spec, questions, or commands. Type 'quit' to exit.

You:
Now you can interact with it:
Paste an OpenAPI Spec: Paste your YAML or JSON spec directly. The router should detect it and trigger parse_openapi_spec.
Ask for Graph: "generate the execution graph" or "create a workflow for creating and then updating a user".
Ask for Payloads: "generate example payloads"
Modify Graph: "add edge createUser to getUser"
Ask Questions: "describe the graph", "validate the graph", "show the graph json"
Type quit to exit the interactive loop.
The application will use the LLMs to understand your input, route it to the correct tool function (core_logic.py), execute the logic (often involving another LLM call), update the state, and provide a response. State for your session is saved between turns in the .state_cache directory. Resolved schemas are cached in .openapi_cache.





You: what all apis are there



Assistant: