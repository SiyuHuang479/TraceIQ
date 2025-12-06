# mcp_server.py

# ======== imports ========
import json
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from pyspark.sql import SparkSession




# ======== configurations ========
OUTPUT_PARQUET_PATH = "cluster_trace.parquet"
TABLE_NAME = "instances"




# ======== Pydantic models ========
class SQLQuery(BaseModel):
    sql_query: str = Field(..., description="The Spark SQL query to execute against the '{TABLE_NAME}' table.")




# ======== Initialize Spark Session ========
spark = SparkSession.builder \
    .appName("MCPSparkRunner") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "localhost") \
    .getOrCreate()




# ======== load parquet data into Spark table ========
try:
    df_raw = spark.read.parquet(OUTPUT_PARQUET_PATH)
    df_raw.createOrReplaceTempView(TABLE_NAME)
    print(f"Loaded data from {OUTPUT_PARQUET_PATH} into Spark table '{TABLE_NAME}'")
except Exception as e:
    print(f"Error loading data from {OUTPUT_PARQUET_PATH}: {e}")
    exit(1)




# ======== Initialize FastAPI app ========
app = FastAPI(
    title="DLRM Cluster MCP Server (Spark SQL)",
    description="Tool for running dynamic Spark SQL queries against the 'instances' table.",
    version="1.0.0",
)




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



# ================= start Uvicorn ===================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)