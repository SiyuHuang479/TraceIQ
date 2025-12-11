#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}       TraceIQ System Launcher v1.0      ${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# 1. check & install dependencies
echo -e "${YELLOW}[1/3] Checking dependencies...${NC}"
if pip install -r requirements.txt; then
    echo -e "${GREEN}✔ Dependencies installed.${NC}"
else
    echo -e "\033[0;31m✘ Failed to install dependencies. Please check your python environment.${NC}"
    exit 1
fi
echo ""

# 2. only run when there's no parquet file
echo -e "${YELLOW}[2/3] Checking dataset status...${NC}"
PARQUET_DIR="cluster_trace.parquet"

if [ -d "$PARQUET_DIR" ]; then
    echo -e "${GREEN}✔ Parquet data found. Skipping ETL process.${NC}"
else
    echo -e "${YELLOW}⚡ Parquet data missing. Running ETL (load_data.py)...${NC}"
    if python load_data.py; then
        echo -e "${GREEN}✔ Data successfully processed.${NC}"
    else
        echo -e "\033[0;31m✘ Data processing failed.${NC}"
        exit 1
    fi
fi
echo ""

# 3. run backend services
echo -e "${YELLOW}[3/3] Starting TraceIQ Backend Server...${NC}"
echo -e "${GREEN}✔ Server is launching at http://0.0.0.0:8000${NC}"
echo -e "${BLUE}ℹ  Keep this terminal open.${NC}"
echo ""

# open webpage

HTML_FILE=$(pwd)/index.html

if command -v open >/dev/null 2>&1; then
    # macOS
    open $HTML_FILE
elif command -v xdg-open >/dev/null 2>&1; then
    # Linux (e.g., Ubuntu/WSL)
    xdg-open $HTML_FILE
elif command -v start >/dev/null 2>&1; then
    # Windows (via Git Bash/Cygwin)
    start $HTML_FILE
else
    echo -e "${YELLOW}⚠️ Could not open browser automatically. Please open the following path manually:${NC}"
    echo -e "${YELLOW}   file://$HTML_FILE${NC}"
fi

# run server
python mcp_server.py