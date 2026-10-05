from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import time

from Vector_RAG.Vector_RAG_query import run_vector_rag
from Graph_RAG.Graph_RAG_query import run_ast_graph_rag 
from evaluate_hallucination import evaluate_rag_output

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

import zipfile
import shutil
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

class GithubRepoRequest(BaseModel):
    repo_url: str

def clear_data_dir():
    # Helper to wipe the data directory clean before a new codebase is uploaded
    for item in DATA_DIR.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    print("Cleared data directory.")

def trigger_ingestion():
    # Helper to trigger the ingestion scripts after new files are added
    print("Triggering Graph Ingestion (Vector disabled due to API limits)...")
    # p1 = subprocess.Popen(["python", "Vector_RAG/Vector_RAG_ingest.py"])
    p2 = subprocess.Popen(["python", "Graph_RAG/Graph_RAG_ingest.py"])
    # p1.wait()
    p2.wait()

@app.post("/upload")
async def upload_codebase(file: UploadFile = File(...)):
    clear_data_dir()
    # 1. Save the uploaded ZIP file
    file_location = DATA_DIR / file.filename
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # 2. Extract the ZIP file
    if file.filename.endswith(".zip"):
        extract_dir = DATA_DIR / file.filename.replace(".zip", "")
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(file_location, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        # Remove the zip file after extraction
        os.remove(file_location)
        
        trigger_ingestion()
        return {"message": f"Successfully extracted {file.filename} into {extract_dir}. Background ingestion started."}
    else:
        # If it's a single file
        trigger_ingestion()
        return {"message": f"Successfully saved {file.filename}. Background ingestion started."}

@app.post("/upload-github")
async def upload_github_repo(req: GithubRepoRequest):
    repo_url = req.repo_url.strip()
    if not repo_url.startswith("https://github.com/"):
        return {"error": "Invalid GitHub URL"}
    
    # Extract repo name
    parsed = urlparse(repo_url)
    repo_name = parsed.path.strip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
        
    clear_data_dir()
    target_dir = DATA_DIR / repo_name
    
    if target_dir.exists():
        return {"message": f"Repository {repo_name} already exists. Triggering re-ingestion."}
        
    # Clone the repository
    try:
        subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)
        trigger_ingestion()
        return {"message": f"Successfully cloned {repo_name}. Background ingestion started."}
    except subprocess.CalledProcessError as e:
        return {"error": f"Failed to clone repository: {str(e)}"}

from code_agent import generate_feature_code

class FeatureRequest(BaseModel):
    request: str

@app.post("/generate-code")
async def api_generate_code(req: FeatureRequest):
    code = generate_feature_code(req.request)
    return {"generated_code": code}

@app.post("/query")
async def query_codebase(req: QueryRequest):
    test_query = req.query
    
    # Run Vector RAG (Disabled to save tokens)
    try:
        # v_context, v_answer = run_vector_rag(test_query)
        v_answer = "Vector-RAG disabled to save API tokens. Enjoy Graph-RAG!"
        v_context = "[]"
    except Exception as e:
        v_answer = "API Error"
        v_context = "[]"
    
    vector_report = {"score": 0, "total_claims": 1, "unverified_claims": []}
    vector_report["answer"] = v_answer
    vector_report["accuracy_score"] = 0
    vector_report["hallucination_score"] = 0
    
    # Run AST Graph RAG
    try:
        g_context, g_answer = run_ast_graph_rag(test_query) 
    except Exception as e:
        g_context = "[]"
        g_answer = f"Graph-RAG Generation API Error: {str(e)}"
        
    try:
        graph_report = evaluate_rag_output("AST-Graph-RAG", test_query, g_context, g_answer)
    except Exception as e:
        print(f"Skipping RAGAS Evaluation due to API limit: {e}")
        graph_report = {"score": 0, "total_claims": 1, "unverified_claims": []}
        
    graph_report["answer"] = g_answer
    graph_report["accuracy_score"] = graph_report.get("score", 0)
    
    g_total = graph_report.get("total_claims", 1)
    g_total = g_total if g_total > 0 else 1
    graph_report["hallucination_score"] = round(len(graph_report.get("unverified_claims", [])) / g_total, 2)
    
    return {
        "query": test_query,
        "vector_rag": vector_report,
        "graph_rag": graph_report
    }
