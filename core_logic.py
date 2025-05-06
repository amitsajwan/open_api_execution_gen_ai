# filename: core_logic.py
# ... (imports and other methods remain the same) ...

class OpenAPICoreLogic:
    # ... (__init__ and other methods remain the same) ...

    def answer_openapi_query(self, state: BotState) -> Dict[str, Any]:
        """
        Answers general informational queries about the loaded OpenAPI specification,
        identified APIs, generated payload descriptions, or the execution graph description
        using the worker LLM and available state data.
        Emphasizes that these are descriptions/plans, not execution results.
        Returns a dictionary of state updates including {'__next__': next_node_name}.
        """
        tool_name = "answer_openapi_query"
        state.update_scratchpad_reason(tool_name, f"Starting to answer general query about spec/plan: {state.user_input}")
        logger.debug("Executing answer_openapi_query node (no execution).")

        updates: Dict[str, Any] = {}
        updates['response'] = "Answering your question based on the loaded spec and generated artifacts..."


        query = state.user_input
        schema_loaded = state.openapi_schema is not None
        graph_exists = state.execution_graph is not None
        apis_identified = state.identified_apis is not None
        payloads_described = state.payload_descriptions is not None
        plan_exists = state.execution_plan is not None and len(state.execution_plan) > 0

        if not schema_loaded and not graph_exists and not plan_exists and not apis_identified:
             updates['response'] = "I don't have an OpenAPI specification loaded or any described artifacts (like identified APIs, plans, or graphs) yet. Please provide a spec or ask me to generate something first."
             state.update_scratchpad_reason(tool_name, "Cannot answer query: No schema or artifacts available.")
             logger.info("Cannot answer OpenAPI query: No schema or generated artifacts.")
             updates['__next__'] = "responder"
             return updates

        # Construct a prompt for the LLM using available state information
        prompt_parts = [f"The user is asking a question: \"{query}\""]
        prompt_parts.append("\nAnswer the question based *only* on the following context about the currently loaded OpenAPI specification and described artifacts.")
        prompt_parts.append("--- START CONTEXT ---")

        if schema_loaded and state.schema_summary:
             prompt_parts.append("\nAvailable OpenAPI Schema Summary:")
             prompt_parts.append("```")
             prompt_parts.append(state.schema_summary[:3000] + ("..." if len(state.schema_summary) > 3000 else ""))
             prompt_parts.append("```")
        elif schema_loaded:
             prompt_parts.append("\nContext: An OpenAPI Schema is loaded, but its summary is not available or too long to include fully.")
        else: # Should not happen due to check above, but good practice
             prompt_parts.append("\nContext: No OpenAPI Schema is loaded.")


        if apis_identified:
             prompt_parts.append(f"\nIdentified APIs ({len(state.identified_apis)} found):")
             api_summaries = [f"- {api.get('operationId', 'N/A')}: {api.get('summary', 'No summary')} ({api.get('method','').upper()} {api.get('path','')})" for api in state.identified_apis[:15]]
             prompt_parts.append("\n".join(api_summaries))
             if len(state.identified_apis) > 15:
                 prompt_parts.append(f"... and {len(state.identified_apis) - 15} more.")

        if graph_exists:
             prompt_parts.append("\nExecution Graph Description Exists:")
             prompt_parts.append(f"- Description: {state.execution_graph.description or 'No overall description provided.'}")
             prompt_parts.append(f"- Nodes (Steps): {len(state.execution_graph.nodes)}")
             node_ids_list = [node.effective_id for node in state.execution_graph.nodes]
             prompt_parts.append(f"- Edges (Dependencies): {len(state.execution_graph.edges)}")
             if len(node_ids_list) < 15:
                 node_list = ", ".join(node_ids_list)
                 prompt_parts.append(f"  Node Sequence (approx): {node_list}")
             else:
                 prompt_parts.append(f"  Node IDs: {', '.join(node_ids_list[:15])}...")


        if plan_exists:
             prompt_parts.append(f"\nExecution Plan Description Exists:")
             prompt_parts.append(f"- Steps: {len(state.execution_plan)}")
             prompt_parts.append(f"  Plan Sequence: {state.execution_plan}")

        if payloads_described:
             prompt_parts.append(f"\nExample Payload Descriptions Generated for {len(state.payload_descriptions)} operations.")
             payload_desc_previews = [f"- {op_id}: {desc[:60]}..." for op_id, desc in list(state.payload_descriptions.items())[:10]]
             prompt_parts.append("  Example Descriptions:")
             prompt_parts.append("\n".join(payload_desc_previews))
             if len(state.payload_descriptions) > 10:
                 prompt_parts.append(f"... and descriptions for {len(state.payload_descriptions) - 10} more operations.")

        prompt_parts.append("\n--- END CONTEXT ---")
        prompt_parts.append("\nInstructions for Answering:")
        prompt_parts.append("1. Carefully read the user's question.")
        prompt_parts.append("2. Find the answer *only* within the provided context above.")
        prompt_parts.append("3. CRITICAL: Base your entire answer ONLY on the context provided. Do not use external knowledge or assume information not present in the context.")
        prompt_parts.append("4. If the context contains the answer, provide it clearly and concisely.")
        prompt_parts.append("5. If the answer is not found in the context, state explicitly that the information is not available in the current loaded specification or artifacts.")
        prompt_parts.append("6. Remember and subtly remind the user if relevant that I can only analyze the spec and describe workflows/plans; I cannot actually execute API calls.")
        prompt_parts.append("\nAnswer to user:")

        full_prompt = "\n".join(prompt_parts)

        try:
            llm_response = llm_call_helper(self.worker_llm, full_prompt)
            updates['response'] = llm_response.strip()
            state.update_scratchpad_reason(tool_name, "LLM generated response to general query based on context (no execution).")
            logger.info("LLM generated response for general OpenAPI query (no execution).")
            updates['__next__'] = "responder"
        except Exception as e:
            updates['response'] = f"I encountered an error while trying to answer your question about the OpenAPI specification: {e}. Please try rephrasing. Remember, I can only describe specs and plans, not execute them."
            state.update_scratchpad_reason(tool_name, f"LLM call failed: {e}")
            logger.error(f"Error calling LLM for answer_openapi_query: {e}", exc_info=True)
            updates['__next__'] = "responder"

        return updates

    # ... (rest of the class and file remain the same) ...
