# mcp_server.py

# ======== imports ========
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from pyspark.sql import SparkSession
import requests 
from google import genai 
from google.genai import types 
import threading 




# ======== configurations ========
OUTPUT_PARQUET_PATH = "cluster_trace.parquet"
TABLE_NAME = "instances"
GEMINI_API_KEY = "AIzaSyCiUm5i5YcDBe2d_iHYhPhoRYUbYR2I4BM"
FASTAPI_BASE_URL = "http://localhost:8000"
OPENAPI_URL = f"{FASTAPI_BASE_URL}/openapi.json"


# ======== Initialize Gemini client ========
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"
TOOL_DECLARATIONS = None 



# ======== Pydantic models ========
class SQLQuery(BaseModel):
    sql_query: str = Field(..., description="The Spark SQL query to execute against the '{TABLE_NAME}' table.")

class ChatRequest(BaseModel):
    prompt:str 



# ======== Initialize Spark Session ========
spark = SparkSession.builder \
    .appName("MCPSparkRunner") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "localhost") \
    .getOrCreate()

try:
    df_raw = spark.read.parquet(OUTPUT_PARQUET_PATH)
    df_raw.createOrReplaceTempView(TABLE_NAME)
    print(f"Data loaded into Spark table '{TABLE_NAME}'.")
except Exception as e:
    print(f"Error loading data: {e}. Please run load_data.py first!")
    exit(1)


# ======== Initialize FastAPI app and CORS ========
app = FastAPI(
    title="DLRM Cluster MCP Server (Spark SQL)",
    description="Tool for running dynamic Spark SQL queries against the 'instances' table.",
    version="1.0.0",
)
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def load_tool_declaration_internal():
    global TOOL_DECLARATIONS
    
    # delay import to make sure that server already starts
    from time import sleep
    sleep(2) 

    try:
        response = requests.get(OPENAPI_URL)
        response.raise_for_status()
        
        openapi_spec = response.json()
        
        
        tool_declaration_data = {
            'name': 'execute_spark_sql_query',
            'description': openapi_spec['paths']['/tools/spark_sql_query']['post']['description'],
            'parameters': {
                'type': 'object',
                'properties': {'sql_query': {'type': 'string', 'description': 'The Spark SQL query to execute.'}},
                'required': ['sql_query']
            }
        }
        
        TOOL_DECLARATIONS = [
            types.Tool(function_declarations=[types.FunctionDeclaration(**tool_declaration_data)])
        ]
        
        print("✅ Tool declarations successfully prepared for Gemini.")
        
    except Exception as e:
        print(f"❌ Error during internal tool loading: {e}")


@app.on_event("startup")
async def startup_event():
    threading.Thread(target=load_tool_declaration_internal).start()








# ======== MCP endpoint ========
@app.post(
    "/tools/spark_sql_query",
    summary=f"Execute Spark SQL query on the {TABLE_NAME} table",
    description=f"Allows the LLM to write and execute any Spark SQL query (SELECT, GROUP BY, AVG, COUNT, etc.) against the '{TABLE_NAME}' table to answer complex questions. Columns available: instance_sn, role, app_name, cpu_request, gpu_request, rdma_request, memory_request, disk_request, creation_time, scheduled_time, deletion_time. NOTE: creation_time, scheduled_time, and deletion_time are already numerical timestamps in seconds (no need for unix_timestamp() or conversion). Calculate delay using: (scheduled_time - creation_time).",
    tags=["SparkSQL"],
    operation_id="execute_spark_sql_query"
)

def execute_spark_sql_query(query: SQLQuery) -> Dict[str, Any]:
    # safety check: only allow SELECT statements
    if not query.sql_query.lower().strip().startswith("select"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed.")
    
    try:
        # execute Spark SQL
        result_df = spark.sql(query.sql_query)

        # transform result to list of dicts
        result_list = result_df.limit(100).toPandas().to_dict('records')

        return {
            "query_executed": query.sql_query,
            "result_count": len(result_list),
            "results": result_list
        }
    except Exception as e:
        # catch SQL errors
        raise HTTPException(status_code=500, detail=f"Spark SQL Execution Error: {str(e)}")




@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest) -> Dict[str, Any]:
    """
    get user prompt, execute RAG flow, return final answer
    """
    if not TOOL_DECLARATIONS:
        return {"response": "initializing system... pls try later"}

    user_prompt = request.prompt
    
    # RAG workflow
    # step 1: use LLM to identify whether tool is needed
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(tools=TOOL_DECLARATIONS)
    )
    
    if not response.function_calls:
        # LLM reply directly
        return {"response": response.text, "sql_query": None}
    
    # step 2 & 3: attain SQL command and execute 
    function_call = response.function_calls[0]
    func_name = function_call.name
    sql_query = function_call.args['sql_query']
    query_results = execute_spark_sql_query(SQLQuery(sql_query=sql_query))

    # step 4: return tool results to LLM 
    function_call_dict = {
        'name': function_call.name,
        'args': function_call.args
    }
    total_response_dict = {
        'name': func_name,
        'response': query_results
    }
    response_to_llm = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]),
            types.Content(
                role="model", 
                parts=[types.Part.from_function_call(**function_call_dict)] # <-- 传入解包后的字典！
            ),
            types.Content(role="tool", parts=[types.Part.from_function_response(name=func_name, response=query_results)])
        ]
    )
    
    return {
        "response": response_to_llm.text,
        "sql_query": sql_query 
    }


# ================= start Uvicorn ===================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)