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

### ▶️ Demo Video


https://github.com/user-attachments/assets/382bbd5d-0a4e-4cdc-a2c0-bc74c5d2fd08



---

## 🚀 How to Start? 
### A. Automated Launch

The `start_traceiq.sh` script automatically installs dependencies, runs the ETL process, and launches both the backend server and the frontend application.

1.  **Grant Execution Permissions:**
    ```bash
    chmod +x start_traceiq.sh
    ```
2.  **Run the Launcher Script:**
    ```bash
    bash start_traceiq.sh
    ```
3.  The system will automatically initialize the Spark cluster and open `index.html` in your default browser.

### B. Manual Setup & Deployment

If the script fails or if you require a specific environment setup (e.g., in a virtual environment), follow these steps:

1.  **Install Dependencies:**
    Install all required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the ETL Pipeline:**
    Process the raw cluster trace data into the optimized Parquet format. This is a crucial one-time step for PySpark performance.
    ```bash
    python load_data.py
    ```

3.  **Start the Backend MCP Server:**
    This command initializes the Spark Session and exposes the analytical tools via the FastAPI endpoint on port 8000.
    ```bash
    python mcp_server.py
    ```

4.  **Access the Frontend:**
    Once the server is running, manually open the frontend interface in your browser:
    ```
    # Locate index.html in your project directory and open it.
    # Example (Browser Address Bar): file:///path/to/your/project/index.html
    ```
---


## 🧠 What This Project Does (System Design & Analysis Pipeline)

This project consists of two tightly-coupled components:  
**(1) a distributed data processing & analytics pipeline**, and  
**(2) an MCP server that allows an LLM to interact with the dataset via natural language.**

Below is a breakdown of what each major component does.

---

### 📊 Data Processing & ETL (`load_data.py`)

The raw Alibaba cluster trace data is provided as CSV files, which are not ideal for large-scale analytical workloads.  
In `load_data.py`, I implement a one-time **ETL (Extract–Transform–Load)** pipeline using **PySpark**:

- Load raw CSV trace data into Spark DataFrames.
- Perform basic **data cleaning and validation**, including:
  - Enforcing resource constraints such as  
    `cpu_request ≤ cpu_limit`, `gpu_request ≤ gpu_limit`, etc.
  - Validating timestamps, e.g.  
    `creation_time ≤ scheduled_time ≤ deletion_time` (when available).
- Persist the cleaned dataset as **Parquet files**, enabling efficient Spark SQL queries and repeated analysis.


---

### 🧩 MCP Server & Analytical Tools (`mcp_server.py`)

The core of TraceIQ is an **MCP-compliant server** built with **FastAPI**, which exposes multiple analytical tools that can be invoked by an LLM.

When the server starts, it:
- Initializes a Spark session.
- Loads the preprocessed Parquet dataset.
- Registers it as a Spark SQL table for distributed querying.

The server exposes the following tools:

#### 🔧 MCP Tool 1: Distributed Spark SQL Query Execution
- The LLM generates a Spark SQL query from natural language.
- The MCP server executes the query using Spark and returns:
  - The executed SQL command
  - SQL execution results

---

#### 📈 MCP Tool 2: Scheduling Delay Distribution
- Computes scheduling delays for different instance types (e.g., CN vs HN).
- Uses Spark DataFrame operations to aggregate results.
- Generates distribution plots (CDF) using Matplotlib.
- Returns both numerical summaries and generated plots to the LLM.

---

#### 📉 MCP Tool 3: Resource Saturation Analysis
- Analyzes requested vs. allocated resources (CPU, memory, GPU, RDMA).
- Identifies skew, imbalance, and potential long-tail behaviors.
- Produces visualizations as boxplots.
- Returns both numerical summaries and generated plots to the LLM.
  
---

#### ⏱ MCP Tool 4: Instance Lifetime Analysis
- Computes and compares instance lifetimes.
- - Generates distribution plots (CDF) using Matplotlib.
- Outputs both summary statistics and lifetime distribution plots.

---

### 🤖 End-to-End Interaction Flow

1. A user asks a **natural language question** (e.g., via the web UI).
2. The LLM interprets the question and decides which MCP tool to invoke.
3. The MCP server executes distributed computation or visualization logic using Spark.
4. Results (tables, statistics, plots) are returned to the LLM.
5. The LLM synthesizes a final, human-readable answer.

This design allows users to **explore and analyze a large-scale GPU inference cluster without writing SQL or Spark code**, while still leveraging the full power of distributed computation.

---

### 🎯 Why This Matters

TraceIQ demonstrates how **Model Context Protocol (MCP)** can bridge:
- Large-scale distributed systems,
- Data-intensive analytics,
- And modern LLM-based interfaces.

Instead of static notebooks, users can **interactively query, analyze, and diagnose complex GPU cluster behavior using natural language**, making operational insights more accessible and exploratory.


---
## 📌 Key Insights

Using the dataset, this project explores several operational questions, including:

- Whether CPU-only (CN) and GPU-backed (HN) instances experience different scheduling delays.
- How resource requests (CPU, memory, GPU, RDMA) are distributed, and whether long-tail behaviors exist.
- How long different types of inference instances typically run, and how lifetimes vary across roles.

These insights are explored both through static analysis (in notebooks)
and dynamically via the MCP tools exposed by the server.

---

## ❓ Example Queries
- How many instances are in the dataset?
- Do GPU (HN) instances experience longer scheduling delays than CPU (CN) instances?
- How does instance lifetime differ between CN and HN？
- Is there evidence of resource over-provisioning?
- Compare scheduling delay (CDF).
- Identify CPU / GPU bottlenecks

--- 

## 📊 Dataset Context

* **Name:** Alibaba Cluster Trace (GPU-Disaggregated DLRM)
* **Source:** NSDI '25 Academic Research Trace
* **Scale:** 23,871 Inference Instances (CN & HN roles).
* **Characteristics:** High-priority, Latency-sensitive workloads, reflecting real-world AI serving infrastructure performance.

---

## 📂 Repository Structure (Quick Guide)

- `load_data.py` — ETL and data cleaning pipeline.
- `mcp_server.py` — MCP server and analytical tools.
- `index.html` — Simple web UI for interacting with the system.





