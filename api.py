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

@app.post("/upload")
async def upload_codebase(file: UploadFile = File(...)):
    # Mock implementation of codebase upload to simulate the processing step
    return {"message": f"Successfully uploaded {file.filename}. Processing into AST Graph and Vector DB..."}

@app.post("/query")
async def query_codebase(req: QueryRequest):
    test_query = req.query
    
    # Run Vector RAG
    v_context, v_answer = run_vector_rag(test_query)
    vector_report = evaluate_rag_output("Vector-RAG", test_query, v_context, v_answer)
    vector_report["answer"] = v_answer
    vector_report["accuracy_score"] = vector_report.get("score", 0)
    
    # Calculate hallucination score as the ratio of unverified claims to total claims
    v_total = vector_report.get("total_claims", 1)
    v_total = v_total if v_total > 0 else 1
    vector_report["hallucination_score"] = round(len(vector_report.get("unverified_claims", [])) / v_total, 2)
    
    # Run AST Graph RAG
    g_context, g_answer = run_ast_graph_rag(test_query) 
    graph_report = evaluate_rag_output("AST-Graph-RAG", test_query, g_context, g_answer)
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
