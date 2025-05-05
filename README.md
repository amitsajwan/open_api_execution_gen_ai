OpenAPI Specification Analyzer AgentThis project implements an AI agent using LangGraph that can analyze and describe OpenAPI (v3) specifications. It leverages Large Language Models (LLMs) to understand the structure of an API, identify relevant endpoints, describe potential workflows, and answer questions about the specification.Important: This agent focuses solely on analyzing and describing the OpenAPI specification. It does not execute any actual API calls.FeaturesParse OpenAPI Specs: Accepts OpenAPI v3 specifications in YAML or JSON format.Summarize Specs: Generates human-readable summaries of the API's capabilities.Identify APIs: Identifies relevant API operations based on user goals or general analysis.Describe Payloads: Generates natural language descriptions of example request payloads for API operations (does not create executable data).Describe Workflows: Generates execution graph descriptions (as a Directed Acyclic Graph - DAG) outlining potential sequences of API calls and data dependencies based on the spec and user goals.Answer Questions: Responds to user queries about the loaded spec, identified APIs, or the described workflow graph.Stateful Conversation: Remembers the loaded spec and generated artifacts across multiple turns within a session using LangGraph's checkpointing.Caching: Caches parsed schemas to speed up processing of previously seen specifications.PrerequisitesPython: Version 3.8 or later recommended.LLM Access: API keys and necessary libraries for your chosen Large Language Model provider (e.g., OpenAI, Google Gemini, Azure OpenAI).Install Libraries:pip install pydantic langgraph langchain diskcache
# Add your specific LLM client library, e.g.:
# pip install langchain-openai
# pip install langchain-google-genai
# pip install python-dotenv # If using .env for API keys
SetupSave the Code:Create a new directory for your project (e.g., openapi_analyzer).Save the provided Python files (models.py, utils.py, core_logic.py, router.py, graph.py, main.py) into this directory.Save the provided requirements.txt and this README.md.Replace Placeholder LLMs:Crucial Step: Open main.py.Locate the initialize_llms function and the PlaceholderLLM class definition within it.Delete the PlaceholderLLM class definition.Replace the placeholder initializations (router_llm = PlaceholderLLM(...), worker_llm = PlaceholderLLM(...)) with your actual LLM client setup (e.g., using ChatOpenAI, ChatGoogleGenerativeAI, AzureChatOpenAI from LangChain).Ensure the LLM objects you create have an .invoke() method compatible with the llm_call_helper function in utils.py.Configure your API keys securely (environment variables are recommended, potentially using python-dotenv and a .env file).Example using LangChain with OpenAI (replace placeholders):# main.py -> initialize_llms() function
import os
from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# load_dotenv() # Uncomment if using .env file

# Load your API keys (best practice is environment variables)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
     logger.critical("OPENAI_API_KEY environment variable not set.")
     raise ValueError("Missing OpenAI API Key")

try:
    # Use appropriate models for routing and core tasks
    router_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, api_key=openai_api_key)
    worker_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1, api_key=openai_api_key) # More capable model for analysis
    logger.info("Initialized OpenAI LLMs.")
    return router_llm, worker_llm
except Exception as e:
    logger.critical(f"Failed to initialize LLMs: {e}. Check API keys and library setup.", exc_info=True)
    raise
Running the AgentNavigate to your project directory (openapi_analyzer) in your terminal.Execute the main.py script:python main.py
The script will start, initialize the (real) LLMs, build the graph, and prompt you for input:Starting new session: <some-uuid>
Enter your OpenAPI spec, questions, or commands. Type 'quit' to exit.

You:
Interacting with the AgentProvide an OpenAPI Spec: Paste your YAML or JSON spec directly into the prompt. The agent should detect and parse it.Ask for Workflow Description: "Generate the execution graph description" or "Describe a workflow for creating and then getting a user".Ask for Payload Descriptions: "Generate example payload descriptions for the user APIs".Ask Questions: "Describe the graph", "Show the graph description json", "What endpoints are available for managing products?", "Summarize the API".Type quit to exit the interactive loop.The agent uses LLMs to understand your input, routes it through the defined graph nodes (core_logic.py), updates its internal state, and provides a natural language response. Remember, all outputs are descriptions based on the spec, not results of actual API calls. Session state is maintained using checkpointing (persisted in memory by default). Resolved schemas are cached in the .openapi_cache directory.
