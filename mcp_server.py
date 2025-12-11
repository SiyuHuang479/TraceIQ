# mcp_server.py

# ======== imports ========
import json
import math
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



# helper function
def safe_float(value):
    try:
        if value is None:
            return 0.0
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    except:
        return 0.0



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



# load functions
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
       
        lifetime_plot_declaration = types.FunctionDeclaration(
            name='get_lifetime_distribution_plot',
            description="Generates a **CDF VISUALIZATION (Plot)** of Instance Lifetime (duration from creation to deletion), which is essential for comparing long-tail survival characteristics between CN and HN. Triggers on: 'lifetime distribution', 'survival analysis', 'churn rate'. **DO NOT USE SQL FOR DISTRIBUTIONS.**",
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




# ======= MCP tools =============

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

    print(f"\n--- DEBUG: Received Prompt ---")
    print(f"User: {user_prompt}")
    print(f"-----------------------------\n")
    
    # RAG workflow
    # step 1: use LLM to identify whether tool is needed
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(tools=TOOL_DECLARATIONS, system_instruction=SYSTEM_INSTRUCTION))
    
    print(f"\n--- DEBUG: LLM Response ---")
    if response.function_calls:
        func_call = response.function_calls[0]
        print(f"Tool Called: {func_call.name}")
        print(f"Arguments: {func_call.args}")
    else:
        print(f"Direct Text: {response.text[:50]}...")
    print(f"----------------------------\n")
    
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

    def final_llm_call(contents_list, model_name, config):
        # --- FIX: Adding try-except for external API failure ---
        try:
            return client.models.generate_content(model=model_name, contents=contents_list, config=config)
        except google.genai.errors.ServerError as e:
            print(f"\n🚨 GEMINI SUMMARY FAILED: {e}\n")
            return types.GenerateContentResponse(
                text=f"Error: The AI model service failed to summarize the results (503 UNAVAILABLE). The Spark query executed successfully, but the final natural language summary could not be generated. Please try again in a moment."
            )
        
    
    contents_for_summary = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]),
        types.Content(
            role="model", 
            parts=[types.Part.from_function_call(**function_call_dict)]
        ),
        types.Content(role="tool", parts=[types.Part.from_function_response(name=func_name, response=query_results)])
    ]

    final_config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
    response_to_llm = await run_in_threadpool(
        final_llm_call, contents_for_summary, MODEL_NAME, final_config
    )
    
    return {
        "response": response_to_llm.text,
        "sql_query": sql_query,
        "plot_data": query_results if 'image_base64' in query_results else None
    }



# ============ PRESET TOOL ===========
'''
return CN and HN instances sched_delay_plot in Base64 encoding PNG string
'''
def get_delay_distribution_plot_data():
    # 1. Filter Valid Data using Spark DataFrame API (Lazy Evaluation)
    df = spark.table(TABLE_NAME)
    df = df.filter("scheduled_time IS NOT NULL AND creation_time IS NOT NULL") \
           .withColumn("delay", F.col("scheduled_time") - F.col("creation_time")) \
           .filter("delay >= 0")

    # 2. Define probabilities for CDF curve (0% to 100%, step 1%)
    # This gives us the shape of the curve without fetching raw data
    probabilities = [i / 100.0 for i in range(101)]
    
    # 3. Calculate Stats & CDF Points in ONE Spark Job
    # Group by Role (CN/HN) and agg
    results = df.groupBy("role").agg(
        F.count("*").alias("count"),
        F.mean("delay").alias("mean"),
        F.min("delay").alias("min"),
        F.max("delay").alias("max"),
        # percentile_approx uses Greenwald-Khanna algorithm (Efficient for Big Data)
        F.percentile_approx("delay", 0.5, 10000).alias("p50"),
        F.percentile_approx("delay", 0.9, 10000).alias("p90"),
        F.percentile_approx("delay", 0.95, 10000).alias("p95"),
        F.percentile_approx("delay", 0.99, 10000).alias("p99"),
        # Get the full CDF curve points
        F.percentile_approx("delay", probabilities, 10000).alias("cdf_points")
    ).collect() # Trigger Action: Return only 2 rows (CN, HN) to Driver

    # 4. Process Data for Plotting (Driver Side)
    stats_dict = {}
    
    plt.figure(figsize=(10, 6))
    colors = {'CN': 'tab:blue', 'HN': 'tab:red'}

    for row in results:
        role = row['role']
        cdf_x = row['cdf_points'] # The delay values at each percentile
        
        # Prepare stats output
        stats_dict[f"{role}_instances"] = {
            "count": row['count'], "mean": row['mean'], "min": row['min'],
            "median_p50": row['p50'], "p90": row['p90'], "p95": row['p95'],
            "p99": row['p99'], "max": row['max']
        }

        # Plot CDF using pre-calculated points
        # x = delay values, y = probabilities (0.0 to 1.0)
        plt.plot(cdf_x, probabilities, label=f'{role} Instances', color=colors.get(role, 'gray'))

    plt.axhline(y=0.99, color='gray', linestyle='--', linewidth=0.8, label='99th Percentile')
    plt.xscale('log') 
    plt.title('CDF of Scheduling Delay (Calculated via Spark Aggregation)')
    plt.xlabel('Scheduling Delay (Seconds) - Log Scale')
    plt.ylabel('Cumulative Probability')
    plt.legend(loc='lower right')
    plt.grid(True, which="both", ls="--", alpha=0.6)

    # Encode
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return {
        "plot_type": "CDF_Delay_Distribution",
        "description": "CDF generated using distributed percentile approximation in Spark.",
        "statistics": stats_dict,
        "image_base64": image_base64
    }


'''
preset tool:
returns a box plot comparing CPU, Memory, and GPU saturation ratios
'''
def get_resource_saturation_plot_data():
    df = spark.table(TABLE_NAME).filter("cpu_limit > 0 AND memory_limit > 0 AND gpu_limit > 0")
    
    # Calculate Ratios
    df = df.select(
        (F.col("cpu_request") / F.col("cpu_limit")).alias("cpu"),
        (F.col("memory_request") / F.col("memory_limit")).alias("mem"),
        (F.col("gpu_request") / F.col("gpu_limit")).alias("gpu")
    )

    # Calculate Boxplot Stats in Spark
    # We need: Min, Q1 (25%), Median (50%), Q3 (75%), Max, Mean
    stats_query = []
    resources = ['cpu', 'mem', 'gpu']
    
    for res in resources:
        stats_query.extend([
            F.mean(res).alias(f"{res}_mean"),
            F.min(res).alias(f"{res}_min"),
            F.max(res).alias(f"{res}_max"),
            F.stddev(res).alias(f"{res}_std"),
            # Get Q1, Median, Q3 in one go
            F.percentile_approx(res, [0.25, 0.5, 0.75], 10000).alias(f"{res}_quantiles")
        ])
    
    row = df.agg(*stats_query).collect()[0] # Only 1 row returned

    # Prepare Data for Matplotlib `bxp`
    plot_data = []
    stats_output = {}

    for res in resources:
        quantiles = row[f"{res}_quantiles"] # [q1, med, q3]
        
        # Structure for ax.bxp()
        item = {
            'label': res.upper(),
            'mean': row[f"{res}_mean"],
            'med': quantiles[1],
            'q1': quantiles[0],
            'q3': quantiles[2],
            'whislo': row[f"{res}_min"], # Simplification: use min/max for whiskers
            'whishi': row[f"{res}_max"],
            'fliers': [] # We don't fetch outliers to save bandwidth
        }
        plot_data.append(item)
        
        stats_output[res] = {
            "mean": row[f"{res}_mean"], "median": quantiles[1],
            "max": row[f"{res}_max"], "std_dev": row[f"{res}_std"]
        }

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw using pre-calculated stats
    ax.bxp(plot_data, showfliers=False, patch_artist=True)
    
    # Color customization (harder with bxp but possible)
    # Adding horizontal line
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Limit Cap (100%)')
    
    ax.set_title('Resource Saturation Analysis (Request / Limit Ratio)')
    ax.set_ylabel('Saturation Ratio (1.0 = Limit Reached)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    
    return {
        "plot_type": "Resource_Saturation_Boxplot",
        "description": "Box plot generated from distributed Spark statistics.",
        "statistics": stats_output,
        "image_base64": image_base64
    }
    

'''
preset tool:
generate a CDF plot for instance lifetime (deletion_time - creation_time)
'''
def get_lifetime_distribution_plot_data():
    # 1. Get Data via Spark SQL
    query = """
        SELECT role, (deletion_time - creation_time) AS lifetime
        FROM instances
        WHERE deletion_time IS NOT NULL AND creation_time IS NOT NULL
    """
    
    try:
        pdf = spark.sql(query).toPandas()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return {
            "plot_type": "CDF_Lifetime_Distribution",
            "description": "Error fetching data.",
            "statistics": {},
            "image_base64": ""
        }

    # 2. Pandas cleaning
    pdf['lifetime'] = list(map(lambda x: x if x > 0 else np.nan, pdf['lifetime']))
    pdf = pdf.dropna(subset=['lifetime'])

    cn_life = pdf[pdf['role'] == 'CN']['lifetime'].values
    hn_life = pdf[pdf['role'] == 'HN']['lifetime'].values

    # 3. calculate
    def calculate_stats(data):
        if len(data) == 0: 
            return {
                "count": 0, "mean": 0.0, "median_p50": 0.0, 
                "p90": 0.0, "max": 0.0
            }
        
       
        return {
            "count": int(len(data)),
            "mean": safe_float(float(np.mean(data))),
            "median_p50": safe_float(float(np.median(data))),
            "p90": safe_float(float(np.percentile(data, 90))),
            "max": safe_float(float(np.max(data)))
        }

    stats = {
        "CN_lifetime_stats": calculate_stats(cn_life),
        "HN_lifetime_stats": calculate_stats(hn_life)
    }

    print("\n--- DEBUG: Lifetime Stats ---")
    print(json.dumps(stats, indent=2))
    print("-----------------------------\n")

    # 4. draw
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(cn_life) > 0:
        ax.hist(cn_life, bins=1000, cumulative=True, density=True, histtype='step', 
                 label='CN Instances', color='tab:blue', linewidth=2)
    
    if len(hn_life) > 0:
        ax.hist(hn_life, bins=1000, cumulative=True, density=True, histtype='step', 
                 label='HN Instances', color='tab:red', linewidth=2)
    
    ax.set_xscale('log') 
    ax.set_title('Instance Lifetime Distribution (CDF)')
    ax.set_xlabel('Lifetime (Seconds) - Log Scale')
    ax.set_ylabel('Cumulative Probability')
    ax.legend(loc='lower right')
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # 5. save and close
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig) 
    
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return {
        "plot_type": "CDF_Lifetime_Distribution",
        "description": "CDF plot showing how long instances survive before deletion.",
        "statistics": stats,
        "image_base64": image_base64
    }




# ================= start Uvicorn ===================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)