````markdown
## OpenAPI Specification Analyzer Agent

**OpenAPI Specification Analyzer Agent** is a stateful AI assistant built with LangGraph and LLMs for deep analysis of OpenAPI v3 specs. It parses your API schema, summarizes capabilities, identifies endpoints, outlines workflows as DAGs, and answers your questions—all without making any real API calls.

---

## Features

- **Parse OpenAPI Specs**  
  Accepts YAML or JSON v3 specs and validates structure.  
- **Summarize Specs**  
  Generates human‑readable overviews of available endpoints and operations.  
- **Identify APIs**  
  Discovers relevant operations based on user intent.  
- **Describe Payloads**  
  Produces natural‑language examples of request bodies (non‑executable).  
- **Describe Workflows**  
  Outputs a Directed Acyclic Graph (DAG) of API call sequences and data dependencies.  
- **Answer Questions**  
  Responds to queries about the loaded spec, endpoints, or generated graph.  
- **Stateful Conversation**  
  Maintains spec, API list, and graph across turns via LangGraph checkpointing.  
- **Caching**  
  Uses DiskCache to persist resolved schemas for faster repeat analysis.

---

## Prerequisites

- **Python 3.8+**  
- **LLM Access** (e.g. OpenAI API key)  
- **Environment management** (recommended): `python‑dotenv` for secure key loading  

---

## Installation

```bash
pip install pydantic langgraph langchain diskcache langchain-openai python-dotenv
````

---

## Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/amitsajwan/open_api_execution_gen_ai.git
   cd open_api_execution_gen_ai
   ```
2. **Configure environment**
   Create a `.env` file:

   ```ini
   OPENAI_API_KEY=<your_api_key_here>
   ```

---

## Usage

```bash
python main.py
```

* Paste your OpenAPI YAML/JSON spec at the prompt.
* Ask questions like “What endpoints are available?” or “Describe the graph.”
* Type `quit` to exit.

---

## Project Structure

```
.
├── main.py
├── core_logic.py
├── graph.py
├── router.py
├── models.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Contributing

1. Fork the repo and create a feature branch.
2. Write tests for new behavior.
3. Submit a pull request with a clear description.

---

## License

Released under the **AMIT License**. See [LICENSE](LICENSE) for details.

*Last updated: May 5, 2025*

```
```
