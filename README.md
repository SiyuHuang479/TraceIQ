# TraceIQ: Intelligent Observability for GPU Clusters
> **CS451 Final Project - Distributed Systems Analysis**

## 📖 Project Overview

**TraceIQ** transforms raw, massive cluster trace data into actionable operational insights. It is designed to analyze the **Alibaba GPU-disaggregated Inference Cluster Trace (DLRM Trace)**, focusing on the principles of **Data-Intensive Distributed Computing**.

The core innovation is combining **PySpark**'s distributed processing power with **Google Gemini 2.5**'s **Natural Language Processing (NLP)** capabilities. Users simply ask questions, and the system executes the necessary distributed SQL or visualization routines.

### 🛠 Tech Stack
* **Compute Engine:** Apache Spark (PySpark) - *For large-scale, distributed data processing.*
* **AI/RAG:** Google Gemini 2.5 Flash - *Handles Natural Language to Tool/SQL translation (Function Calling).*
* **API Layer:** FastAPI (MCP Server) - *Serves as the Model-Controller-Proxy layer.*
* **Visualization:** Matplotlib/Numpy - *Renders analytical plots (CDF, Boxplot) on the fly.*

---

## 🚀 Quick Start (Recommended)

The entire setup—installing dependencies, cleaning the dataset, and launching the server/frontend—is automated via a single shell script.

1.  **Grant Execution Permissions:**
    ```bash
    chmod +x start_traceiq.sh
    ```
2.  **Run the launcher script:**
    ```bash
    bash start_traceiq.sh
    ```
3.  The script will automatically handle all setup steps and **open the `index.html` frontend in your default browser.**

> **Note:** The first run involves a one-time ETL process to convert raw CSV to optimized Parquet format. Subsequent launches are instant.

---


## ⚙️ Implementation Details

### 1. Data Ingestion & ETL
* **Format Conversion:** Raw CSV data is converted into **Parquet** format using PySpark. Parquet is essential for distributed computing as it is columnar, enabling efficient data skipping and **Predicate Pushdown** during querying.
* **Schema Enforcement:** Initial cleaning ensures correct data types and handling of missing values.

### 2. The Multi-Tool Backend (`mcp_server.py`)
The backend manages the persistent Spark Session and routes LLM calls to four distinct analytical tools. Each tool is designed to address a core operational question with a specific visualization output:

| Tool Name | Analysis Focus | Visualization Output (X-axis vs. Y-axis) |
| :--- | :--- | :--- |
| **SQL Query** | Universal aggregation, counting, and ranking. | *Returns raw numerical data (no plot).* |
| **Delay CDF** | **Latency** (Long-Tail Latency, Scheduling Delay distribution). | **CDF Plot** (Cumulative Probability vs. **Delay [Seconds, Log Scale]**). |
| **Saturation** | **Efficiency** (Resource Bottlenecks via Request/Limit ratios). | **Box Plot** (Resource Type [CPU/Mem/GPU] vs. **Saturation Ratio [0 to >1]**). |
| **Lifetime CDF** | **Stability** (Instance Volatility vs. Long-Running Workloads). | **CDF Plot** (Cumulative Probability vs. **Lifetime [Seconds, Log Scale]**). |


### 3. AI-Driven Distributed Execution

The platform is defined by the seamless, scalable execution pipeline integrating AI and Distributed Compute:

* **Intelligent Tool Orchestration:** The **Gemini LLM** acts as an intelligent router. Based on the user's intent ("Show me a plot" vs. "Count the instances"), the LLM dynamically decides whether to call the generic `execute_spark_sql_query` tool or one of the specialized visualization tools (e.g., `get_delay_distribution_plot`).
* **Scalable Aggregation:** All complex calculations (like calculating CDF quantiles and Box Plot quartiles) are pushed down to the Spark cluster's **Worker Nodes**. This is achieved using specialized PySpark functions (e.g., `F.percentile_approx`) to ensure that performance scales horizontally with the data size.
* **Semantic Data Summarization:** After the Spark cluster executes the task and returns the raw statistics (e.g., P99 value), the LLM processes this numerical output to provide a high-level, human-readable summary, translating raw numbers into **data-driven operational insights**.

---

## 📊 Dataset Context

* **Name:** Alibaba Cluster Trace (GPU-Disaggregated DLRM)
* **Source:** NSDI '25 Academic Research Trace
* **Scale:** 23,871 Inference Instances (CN & HN roles).
* **Characteristics:** High-priority, Latency-sensitive workloads, reflecting real-world AI serving infrastructure performance.

---

## 📝 Dependencies & Installation

All required Python packages can be installed using `pip`:

```bash
pip install -r requirements.txt
