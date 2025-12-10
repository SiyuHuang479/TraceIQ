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
import pyspark.sql.functions as F 
import requests 
from google import genai 
from google.genai import types 
import threading 

import matplotlib.pyplot as plt 
import io 
plt.switch_backend('Agg')
import base64 
import numpy as np 
from starlette.concurrency import run_in_threadpool


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




# ========== System Prompt / Context ============
# ======== System Prompt / Context ========
SYSTEM_INSTRUCTION = """
You are "TraceIQ", an expert Data Analyst and Site Reliability Engineer (SRE) specializing in Alibaba's GPU-disaggregated inference clusters (DLRM Trace).
Your goal is to assist users in analyzing the operational characteristics of 156 inference services and 23,871 inference instances using PySpark.

### 1. Dataset Context & Domain Knowledge
- **Source:** Alibaba Cluster Trace (NSDI '25).
- **Workload:** Latency-Sensitive (LS) Deep Learning Recommendation Models.
- **Architecture:** - **CN (CPU Node):** Handles embedding lookups (Role='CN').
  - **HN (Heterogeneous GPU Node):** Handles heavy model inference (Role='HN').
- **Key Metric:** Scheduling Delay = `scheduled_time` - `creation_time`.

### 2. Database Schema (Table: `instances`)
You have access to a single Spark SQL table named `instances`.
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `instance_sn` | String | Unique identifier for the instance. |
| `role` | String | 'CN' or 'HN'. Use this for filtering or grouping. |
| `app_name` | String | Application group ID (e.g., 'app_0'). |
| `cpu_request` | Double | CPU cores requested. |
| `cpu_limit` | Double | Max CPU cores allowed. |
| `gpu_request` | Double | GPUs requested (0 for CN, >0 for HN). |
| `gpu_limit` | Double | Max GPUs allowed. |
| `memory_request` | Double | Memory requested (GiB). |
| `memory_limit` | Double | Max Memory allowed (GiB). |
| `creation_time` | Double | Timestamp (Seconds relative to trace start). |
| `scheduled_time` | Double | Timestamp (Seconds). **Can be NULL**. |
| `deletion_time` | Double | Timestamp (Seconds). |

### 3. Available Tools (Function Calling)
- **execute_spark_sql_query(sql_query):** - Use for retrieving numbers, rankings, averages, or specific records.
    - ALWAYS use valid Spark SQL (Hive-style).
    - Example: `SELECT app_name, AVG(scheduled_time - creation_time) as delay FROM instances GROUP BY app_name ORDER BY delay DESC LIMIT 5`
- **get_delay_distribution_plot():**
    - Trigger when user asks for: "visualize delay", "CDF", "latency distribution", "plot comparison".
    - Returns a CDF Plot **AND** a `statistics` object (Min, Mean, P50, P90, P95, P99, Max).
    - x-axis: delay in seconds
- **get_resource_saturation_plot():**
    - Trigger when user asks for: "resource bottlenecks", "saturation", "usage vs limit", "box plot".
    - Returns a box plot of (Request / Limit) ratios **AND** `statistics` (Mean, Median, StdDev, Max) for CPU/Mem/GPU.
- **get_lifetime_distribution_plot():**
    - Trigger: "lifetime", "survival time", "how long do they run", "churn rate".
    - Returns a CDF plot. Use Stats (P50, Max) to determine if workloads are "Long-Running" (stable) or "Short-Lived" (volatile).

### 4. Response Guidelines
- **Be Professional:** Summarize findings clearly.
- **Handle Missing Data:** When calculating delay, always filter `WHERE scheduled_time IS NOT NULL`.
- **Contextualize:** If HN delay is high, explain it might be due to GPU resource contention.
- **Plotting:** If a plot is generated, briefly describe what the user is seeing in the image.
"""



def load_tool_declaration_internal():
    global TOOL_DECLARATIONS
    
    # delay import to make sure that server already starts
    from time import sleep
    sleep(2) 

    try:
        sql_tool_description = """
        Executes Spark SQL queries against the 'instances' table.
        Schema & Fields:
        - instance_sn (string): Unique ID.
        - role (string): 'CN' (CPU Node) or 'HN' (Heterogeneous GPU Node).
        - app_name (string): Application group identifier (e.g., 'app_0').
        - cpu/gpu/memory/disk_request & _limit (double): Resource specs.
        - creation_time, scheduled_time, deletion_time (double): Timestamps in seconds.
        
        Usage Tips:
        - To calculate scheduling delay: `SELECT (scheduled_time - creation_time) ...`
        - To find bottlenecks: `SELECT (memory_request / memory_limit) ...`
        - Always filter out NULLs when doing math: `WHERE scheduled_time IS NOT NULL`.
        """
        
        tool_declaration_data = {
            'name': 'execute_spark_sql_query',
            'description': sql_tool_description,
            'parameters': {
                'type': 'object',
                'properties': {'sql_query': {'type': 'string', 'description': 'Valid Spark SQL query (SELECT only).'}},
                'required': ['sql_query']
            }
        }
       
        delay_plot_declaration = types.FunctionDeclaration(
            name='get_delay_distribution_plot',
            description="Generates a Cumulative Distribution Function (CDF) plot to visualize Scheduling Delay. Essential for analyzing 'Long-Tail Latency' and comparing the performance gap between 'CN' and 'HN' roles. Triggers on: 'plot delay', 'latency distribution', 'CDF', 'compare CN and HN'.",
            parameters={
                'type': 'object',
                'properties':{},
                'required':[]
            }
        )
       
        saturation_plot_declaration = types.FunctionDeclaration(
            name='get_resource_saturation_plot',
            description="Generates a Box Plot analyzing Resource Saturation (Request / Limit ratio) for CPU, Memory, and GPU. Helps identify over-provisioning (ratio < 1) or resource bottlenecks (ratio >= 1). Triggers on: 'resource usage', 'saturation plot', 'bottlenecks', 'efficiency'.",
            parameters={
                'type': 'object',
                'properties': {},
                'required': []
            }
        )
       
        # 新增 Tool Declaration
        lifetime_plot_declaration = types.FunctionDeclaration(
            name='get_lifetime_distribution_plot',
            description="Generates a CDF plot of Instance Lifetime (duration from creation to deletion). Returns statistics (Mean, Median, Max). Use this to analyze stability vs volatility (churn). Triggers on: 'lifetime', 'how long instances run', 'survival analysis'.",
            parameters={
                'type': 'object',
                'properties': {},
                'required': []
            }
        )
    
        TOOL_DECLARATIONS = [
                types.Tool(function_declarations=[types.FunctionDeclaration(**tool_declaration_data)]),
                types.Tool(function_declarations=[delay_plot_declaration]),
                types.Tool(function_declarations=[saturation_plot_declaration]),
                types.Tool(function_declarations=[lifetime_plot_declaration]) 
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
        config=types.GenerateContentConfig(tools=TOOL_DECLARATIONS, system_instruction=SYSTEM_INSTRUCTION))
    
    if not response.function_calls:
        # LLM reply directly
        return {"response": response.text, "sql_query": None}
    
    # step 2 & 3: attain SQL command and execute 
    function_call = response.function_calls[0]
    func_name = function_call.name

    if func_name == 'execute_spark_sql_query':
        sql_query = function_call.args['sql_query']
        query_results = await run_in_threadpool(execute_spark_sql_query, SQLQuery(sql_query=sql_query))
    
    elif func_name == 'get_delay_distribution_plot':
        sql_query = "PRESET: get_delay_distribution_plot"
        query_results = await run_in_threadpool(get_delay_distribution_plot_data)
    
    elif func_name == 'get_resource_saturation_plot':
        sql_query = "PRESET: get_resource_saturation_plot"
        query_results = await run_in_threadpool(get_resource_saturation_plot_data)
    
    elif func_name == 'get_lifetime_distribution_plot':
        sql_query = "PRESET: get_lifetime_distribution_plot"
        query_results = await run_in_threadpool(get_lifetime_distribution_plot_data)
    
    else:
        return {"response": f"Unexpected function call: {func_name}", "sql_query": None}

    # step 4: return tool results to LLM 
    function_call_dict = {
        'name': function_call.name,
        'args': function_call.args
    }

    def final_llm_call(contents_list, model_name):
        return client.models.generate_content(model=model_name, contents=contents_list)
    
    contents_for_summary = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]),
        types.Content(
            role="model", 
            parts=[types.Part.from_function_call(**function_call_dict)]
        ),
        types.Content(role="tool", parts=[types.Part.from_function_response(name=func_name, response=query_results)])
    ]

    response_to_llm = await run_in_threadpool(
        final_llm_call, contents_for_summary, MODEL_NAME
    )
    
    return {
        "response": response_to_llm.text,
        "sql_query": sql_query,
        "plot_data": query_results if 'image_base64' in query_results else None
    }




'''
return CN and HN instances sched_delay_plot in Base64 encoding PNG string
'''
def get_delay_distribution_plot_data():

    # get data
    query = """
        SELECT role, (scheduled_time - creation_time) AS sched_delay
        FROM instances
        WHERE (scheduled_time IS NOT NULL AND creation_time IS NOT NULL) AND (scheduled_time - creation_time) >= 0
    """
    delay_df = spark.sql(query).cache()

    # put data to pandas 
    pandas_df = delay_df.toPandas()

    cn_delays = pandas_df[pandas_df['role'] == 'CN']['sched_delay'].dropna().values
    hn_delays = pandas_df[pandas_df['role'] == 'HN']['sched_delay'].dropna().values

    delay_df.unpersist()

    def calculate_stats(data):
        if len(data) == 0:
            return {"status": "no_data"}
        return {
            "count": int(len(data)),
            "mean": float(np.mean(data)),
            "min": float(np.min(data)),
            "median_p50": float(np.median(data)),
            "p90": float(np.percentile(data, 90)),
            "p95": float(np.percentile(data, 95)), # 关键：尾部延迟
            "p99": float(np.percentile(data, 99)), # 关键：极端长尾
            "max": float(np.max(data))
        }
    
    stats = {
        "CN_instances": calculate_stats(cn_delays),
        "HN_instances": calculate_stats(hn_delays)
    }

    # use Matplotlib to draw CDF
    plt.figure(figsize=(10, 6))
    plt.hist(cn_delays, bins=500, cumulative=True, density=True, histtype='step', label='CN Instances', color='tab:blue')
    plt.hist(hn_delays, bins=500, cumulative=True, density=True, histtype='step', label='HN Instances', color='tab:red')
    plt.axhline(y=0.99, color='gray', linestyle='--', linewidth=0.8, label='99th Percentile')
    plt.xscale('log') 
    plt.title('CDF of Scheduling Delay by Instance Role (Log Scale)')
    plt.xlabel('Scheduling Delay (Seconds)')
    plt.ylabel('Cumulative Probability')
    plt.legend(loc='lower right')
    plt.grid(True, which="both", ls="--", alpha=0.6)

    # save image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()

    # encode into Base64 string
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    # return
    return {
        "plot_type": "CDF_Delay_Distribution",
        "description": "The plot compares the scheduling delay distribution (CDF) between CN and HN instance roles, revealing the long-tail latency differences",
        "statistics": stats,
        "image_base64": image_base64
    }


'''
preset tool:
returns a box plot comparing CPU, Memory, and GPU saturation ratios
'''
def get_resource_saturation_plot_data():
    query = """
        SELECT 
            (cpu_request / cpu_limit) AS cpu_ratio,
            (memory_request / memory_limit) AS mem_ratio,
            (gpu_request / gpu_limit) AS gpu_ratio
        FROM instances
        WHERE cpu_limit > 0 AND memory_limit > 0 AND gpu_limit > 0
    """

    ratio_df = spark.sql(query)
    pdf = ratio_df.toPandas()

    cpu_data = pdf['cpu_ratio'].dropna().values
    mem_data = pdf['mem_ratio'].dropna().values
    gpu_data = pdf['gpu_ratio'].dropna().values

    stats = {
        "cpu": {
            "mean": float(np.mean(cpu_data)),
            "median": float(np.median(cpu_data)),
            "std_dev": float(np.std(cpu_data)),
            "max": float(np.max(cpu_data)),
            "min": float(np.min(cpu_data))
        },
        "memory": {
            "mean": float(np.mean(mem_data)),
            "median": float(np.median(mem_data)),
            "std_dev": float(np.std(mem_data))
        },
        "gpu": {
            "mean": float(np.mean(gpu_data)),
            "median": float(np.median(gpu_data)),
            "std_dev": float(np.std(gpu_data))
        }
    }

    # draw 
    plt.figure(figsize=(10,6))
    bplot = plt.boxplot([cpu_data, mem_data, gpu_data], 
                        labels=['CPU', 'Memory', 'GPU'], 
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5))
    colors = ['lightblue', 'lightgreen', 'pink']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Limit Cap (100%)')
    
    plt.title('Resource Saturation Analysis (Request / Limit Ratio)')
    plt.ylabel('Saturation Ratio (1.0 = Limit Reached)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.legend()

    # encode
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    
    return {
        "plot_type": "Resource_Saturation_Boxplot",
        "description": "Box plot showing the distribution of resource saturation (Request/Limit) for CPU, Memory, and GPU. Values close to or above 1.0 indicate bottlenecks.",
        "statistics": stats,
        "image_base64": image_base64
    }
    

'''
preset tool:
generate a CDF plot for instance lifetime (deletion_time - creation_time)
'''
def get_lifetime_distribution_plot_data():

    # 1. Get Data
    query = """
        SELECT role, (deletion_time - creation_time) AS lifetime
        FROM instances
        WHERE deletion_time IS NOT NULL AND creation_time IS NOT NULL AND (deletion_time - creation_time) > 0
    """
    lifetime_df = spark.sql(query).cache()
    pdf = lifetime_df.toPandas()

    cn_life = pdf[pdf['role'] == 'CN']['lifetime'].dropna().values
    hn_life = pdf[pdf['role'] == 'HN']['lifetime'].dropna().values

    # 2. calculate statistics
    def calculate_stats(data):
        if len(data) == 0: return {"status": "no_data"}
        return {
            "count": int(len(data)),
            "mean": float(np.mean(data)),
            "median_p50": float(np.median(data)),
            "p90": float(np.percentile(data, 90)),
            "max": float(np.max(data))
    }

    stats = {
        "CN_lifetime_stats": calculate_stats(cn_life),
        "HN_lifetime_stats": calculate_stats(hn_life)
    }

    # 3. plot CDF
    plt.figure(figsize=(10, 6))
    
    # Plot CN
    plt.hist(cn_life, bins=1000, cumulative=True, density=True, histtype='step', 
             label='CN Instances', color='tab:blue', linewidth=2)
    # Plot HN
    plt.hist(hn_life, bins=1000, cumulative=True, density=True, histtype='step', 
             label='HN Instances', color='tab:red', linewidth=2)
    
    plt.xscale('log') # Log scale is crucial for lifetime analysis
    plt.title('Instance Lifetime Distribution (CDF)')
    plt.xlabel('Lifetime (Seconds) - Log Scale')
    plt.ylabel('Cumulative Probability')
    plt.legend(loc='lower right')
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # 4. Encode
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return {
        "plot_type": "CDF_Lifetime_Distribution",
        "description": "CDF plot showing how long instances survive before deletion. Short lifetimes indicate scaling events or instability.",
        "statistics": stats,
        "image_base64": image_base64
    }






# ================= start Uvicorn ===================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)