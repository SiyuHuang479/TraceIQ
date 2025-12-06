# llm_client.py

import os 
import json
import requests
from google import genai
from google.genai import types


# ======= configurations ========
GEMINI_API_KEY = "AIzaSyCiUm5i5YcDBe2d_iHYhPhoRYUbYR2I4BM"
FASTAPI_BASE_URL = "http://localhost:8000"
OPENAPI_URL = f"{FASTAPI_BASE_URL}/openapi.json"



# ======== Initialize Gemini client ========
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"


# get LLM tool declarations 
def load_tool_declaration():
    print(f"getting tool declaration from {OPENAPI_URL}...")
    try:
        response = requests.get(OPENAPI_URL)
        response.raise_for_status()
        
        openapi_spec = response.json()
        
        tools = []
        for path, methods in openapi_spec.get('paths', {}).items():
            for method, spec in methods.items():
                if method.lower() == 'post' and path == '/tools/spark_sql_query':
                    function_name = spec['operationId']
                    description = spec['description']
                    
                    # 1. get request schema
                    request_schema = spec['requestBody']['content']['application/json']['schema']

                    # 2. get actual schema
                    ref_path = request_schema.get('$ref')
                    if not ref_path:
                        parameters_schema = request_schema 
                    else:
                        schema_name = ref_path.split('/')[-1]
                        parameters_schema = openapi_spec['components']['schemas'][schema_name]
                    
                    # 3. adapt Gemini SDK format
                    if 'properties' in parameters_schema:
                        parameters_definition = {
                            'type': 'object',
                            'properties': parameters_schema['properties'],
                            'required': parameters_schema.get('required', [])
                        }
                    else:
                        parameters_definition = parameters_schema
                    
                    # adapt Gemini SDK FunctionDeclaration format
                    tools.append(types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=function_name,
                                description=description,
                                parameters=parameters_definition
                            )
                        ]  
                    ))

                    print(f"Loaded tool declaration for function: {function_name}")
                    return tools 
        
        if not tools:
            raise ValueError("didn't find execute_spark_sql_query in openapi spec")
        
    except Exception as e:
        print(f"Error loading tool declaration: {e}")
        return None 



# execute RAG workflow
def run_nl_to_sql_rag(user_prompt:str, tools):
    # LLM -> SQL -> FastAPI -> Result -> LlM -> Final Response
    print(f"user question: {user_prompt}")

    # step 1, LLM to generate whether it needs to use tool
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(tools= tools)
        ) 
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "fail to get LLM response"
    
    # step 2, check whether to use tool
    if not response.function_calls:
        print("LLM decide to not use tool. Here is the direct reply:")
        print(response.text)
        return response.text

    # step 3, get SQL query and call FastAPI tool endpoint
    function_call = response.function_calls[0]
    func_name = function_call.name

    if func_name != 'execute_spark_sql_query':
        print(f"Unexpected function call: {func_name}")
        return "fail due to unexpected function call"
    
    sql_query = function_call.args['sql_query']
    print(f"LLM generated SQL query: {sql_query}")

    # call FastAPI tool endpoint
    tool_endpoint = f"{FASTAPI_BASE_URL}/tools/spark_sql_query"
    try:
        server_response = requests.post(
            tool_endpoint,
            json={"sql_query": sql_query}
        )
        server_response.raise_for_status()

        query_results = server_response.json() 
        print(f"Received query results from FastAPI tool endpoint.")
    except Exception as e:
        print(f"Error calling FastAPI tool endpoint: {e}")
        query_results = {"error": f"server error: {e}"}

    # step 4, feed results back to LLM for final response
    function_call_dict = {
        'name': function_call.name,
        'args': function_call.args
    }

    tool_response_dict = {
        "name": func_name,
        "response": query_results
    }
    response_to_llm = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            # 1. user prompt
            types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]), 
            
            # 2. LLM function call
            types.Content(
                role="model", 
                parts=[types.Part.from_function_call(**function_call_dict)] # <-- 传入解包后的字典！
            ),
            
            # 3. execution results
            types.Content(
                role="tool", 
                parts=[types.Part.from_function_response(**tool_response_dict)] # <-- 传入解包后的字典！
            )
        ])
    
    return response_to_llm.text



def main():
    # load tool declaration from FastAPI openapi spec
    tool_declarations = load_tool_declaration()
    if not tool_declarations:
        print("Failed to load tool declarations. Exiting.")
        return

    print("RAG system is ready to accept user queries.")

    while True:
        try:
            user_prompt = input("\nEnter your question (or 'exit' to quit): ")
            if user_prompt.lower() == "exit":
                break 
            if not user_prompt.strip():
                continue

            final_answers = run_nl_to_sql_rag(user_prompt, tool_declarations)
            print(final_answers)
            print("-------------------------\n")
        
        except KeyboardInterrupt:
            print("program interrupt")
            break 
        except Exception as e:
            print(f"unknown error: {e}")
            break 


if __name__ == "__main__":
    main()
                
                