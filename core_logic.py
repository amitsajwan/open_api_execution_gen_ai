def process_schema_completely(self, state: BotState) -> Dict[str, Any]:
    """
    Process the loaded OpenAPI schema completely in one go.
    This includes:
    1. Creating a detailed schema summary
    2. Identifying all APIs in the spec
    3. Generating example payload descriptions
    4. Creating an execution graph description
    
    This creates a comprehensive set of artifacts that can be used for 
    subsequent queries without additional processing.
    
    Returns a dictionary of state updates including {'__next__': next_node_name}.
    """
    tool_name = "process_schema_completely"
    state.update_scratchpad_reason(tool_name, "Starting comprehensive schema processing")
    logger.info("Executing process_schema_completely node for full schema processing")
    
    updates: Dict[str, Any] = {}
    
    # Step 1: Generate a detailed schema summary
    try:
        # Similar to what would be done in parse_openapi_spec but focusing on the summary generation
        if not state.openapi_schema:
            updates['response'] = "No OpenAPI schema is loaded. Please provide a spec first."
            updates['__next__'] = "responder"
            state.update_scratchpad_reason(tool_name, "No schema loaded, cannot process.")
            return updates
            
        spec = state.openapi_schema
        info = spec.get('info', {})
        title = info.get('title', 'Untitled API')
        version = info.get('version', 'Unknown')
        description = info.get('description', 'No description provided')
        
        summary_prompt = f"""
        You are summarizing an OpenAPI specification for later reference. Create a concise but comprehensive summary of the API.
        
        API Title: {title}
        API Version: {version}
        
        Description: {description[:1000] + '...' if len(description) > 1000 else description}
        
        Include in your summary:
        1. The overall purpose of this API
        2. Major resource categories/endpoints
        3. Any authentication requirements mentioned
        4. Notable features or patterns in the API design
        
        Keep your summary informative but concise (under 1000 words).
        """
        
        schema_summary = llm_call_helper(self.worker_llm, summary_prompt)
        updates['schema_summary'] = schema_summary
        state.update_scratchpad_reason(tool_name, "Generated schema summary successfully")
        logger.info("Schema summary generated")
        
        # Step 2: Identify all APIs in the spec
        paths = spec.get('paths', {})
        components = spec.get('components', {})
        
        # Extract all API operations
        all_apis = []
        for path, path_item in paths.items():
            # Skip parameters at path level
            for method, operation in path_item.items():
                if method not in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']:
                    continue
                    
                operation_id = operation.get('operationId', f"{method}_{path}")
                summary = operation.get('summary', 'No summary')
                description = operation.get('description', 'No description')
                
                api_item = {
                    'operationId': operation_id,
                    'path': path,
                    'method': method,
                    'summary': summary,
                    'description': description
                }
                all_apis.append(api_item)
        
        updates['identified_apis'] = all_apis
        state.update_scratchpad_reason(tool_name, f"Identified {len(all_apis)} APIs in spec")
        logger.info(f"Identified {len(all_apis)} APIs in spec")
        
        # Step 3: Generate example payload descriptions for all APIs
        payload_descriptions = {}
        
        # Only process a reasonable number of APIs to avoid excessive LLM calls
        apis_to_process = all_apis[:30]  # Process up to 30 APIs
        
        for api in apis_to_process:
            operation_id = api['operationId']
            path = api['path']
            method = api['method']
            
            # Find the operation in the spec
            operation = paths.get(path, {}).get(method, {})
            if not operation:
                continue
                
            # Check if it has a request body
            request_body = operation.get('requestBody', {})
            parameters = operation.get('parameters', [])
            
            # Skip if no request body or parameters
            if not request_body and not parameters:
                payload_descriptions[operation_id] = "No request body or parameters required."
                continue
                
            # Generate a description for this API
            payload_prompt = f"""
            Generate a clear, natural language description of an example payload for the following API operation.
            This is NOT for execution but to help users understand how to use the API.
            
            Operation ID: {operation_id}
            Path: {path}
            Method: {method}
            Summary: {api['summary']}
            
            Parameters: {json.dumps(parameters, indent=2) if parameters else "None"}
            
            Request Body Schema: {json.dumps(request_body, indent=2) if request_body else "None"}
            
            Provide a description that explains:
            1. What parameters or body fields are required
            2. The expected format/type of each field
            3. A realistic example of values that would work
            4. Any constraints or validation rules
            
            Format your response as a clear description, not actual JSON or code.
            """
            
            try:
                payload_description = llm_call_helper(self.worker_llm, payload_prompt)
                payload_descriptions[operation_id] = payload_description
                logger.debug(f"Generated payload description for {operation_id}")
            except Exception as e:
                logger.error(f"Error generating payload description for {operation_id}: {e}")
                payload_descriptions[operation_id] = f"Error generating description: {str(e)}"
        
        updates['payload_descriptions'] = payload_descriptions
        state.update_scratchpad_reason(tool_name, f"Generated payload descriptions for {len(payload_descriptions)} APIs")
        logger.info(f"Generated payload descriptions for {len(payload_descriptions)} APIs")
        
        # Step 4: Create a basic execution graph description
        # This creates a simple sample workflow using some of the identified APIs
        
        # First, categorize APIs by resource and action type
        resources = {}
        for api in all_apis:
            # Extract resource from path or operationId
            path_parts = api['path'].strip('/').split('/')
            resource = path_parts[0] if path_parts else "unknown"
            
            if resource not in resources:
                resources[resource] = []
            resources[resource].append(api)
        
        # Select top resources for the graph (up to 3)
        top_resources = list(resources.keys())[:3]
        selected_apis = []
        for resource in top_resources:
            selected_apis.extend(resources[resource][:3])  # Up to 3 APIs per resource
        
        if not selected_apis:
            selected_apis = all_apis[:5]  # Fallback: take first 5 APIs
            
        # Create nodes from selected APIs
        nodes = []
        for i, api in enumerate(selected_apis[:10]):  # Limit to 10 nodes
            node = Node(
                operationId=api['operationId'],
                display_name=f"{api['method'].upper()}_{api['path'].replace('/', '_')}",
                summary=api['summary'],
                description=api['description'],
                payload_description=payload_descriptions.get(api['operationId'], "No payload description generated.")
            )
            nodes.append(node)
        
        # Create some logical edges based on resource relationships
        edges = []
        for i in range(len(nodes) - 1):
            if i < len(nodes) - 1:
                # Create some edges based on potential data flow
                edge = Edge(
                    from_node=nodes[i].effective_id,
                    to_node=nodes[i+1].effective_id,
                    description=f"Potential data flow from {nodes[i].operationId} to {nodes[i+1].operationId}"
                )
                edges.append(edge)
        
        # Create graph output
        graph_output = GraphOutput(
            nodes=nodes,
            edges=edges,
            description=f"Sample execution graph showing potential workflow using {len(nodes)} operations from the {title} API."
        )
        
        updates['execution_graph'] = graph_output
        state.update_scratchpad_reason(tool_name, f"Generated execution graph with {len(nodes)} nodes and {len(edges)} edges")
        logger.info(f"Generated execution graph with {len(nodes)} nodes and {len(edges)} edges")
        
        # Set response to user
        updates['response'] = f"""
        I've analyzed your OpenAPI specification for {title} v{version} and prepared the following:
        
        1. A comprehensive summary of the API and its capabilities
        2. Identified {len(all_apis)} API operations across various endpoints
        3. Generated detailed payload descriptions for {len(payload_descriptions)} operations
        4. Created a sample execution graph showing how {len(nodes)} operations could be connected in a workflow
        
        You can now ask me questions about the API, like "What endpoints are available?", "Explain how to use the user creation API", or "Show me the execution graph".
        """
        
        updates['__next__'] = "responder"
        state.update_scratchpad_reason(tool_name, "Completed comprehensive schema processing successfully")
        logger.info("Completed comprehensive schema processing")
        
    except Exception as e:
        error_msg = f"Error processing schema completely: {str(e)}"
        logger.error(error_msg, exc_info=True)
        updates['response'] = f"I encountered an error while processing your OpenAPI specification: {str(e)}. Please try again or provide more specific instructions."
        updates['__next__'] = "responder"
        state.update_scratchpad_reason(tool_name, error_msg)
    
    return updates
