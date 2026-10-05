# 🥊 RAG Showdown: Vector-RAG vs Graph-RAG

A full-stack knowledge retrieval showdown designed to benchmark and compare traditional **Vector-RAG** against structural, AST-powered **Graph-RAG**. 

This platform allows you to dynamically ingest entire codebases from GitHub or local uploads, query them, and instantly see how both architectures answer the same question. It then rigorously evaluates both responses using the **RAGAS** framework to generate a hallucination and accuracy score.

---

## 🏗️ Architecture Overview

This project is structured as a modern Monorepo:

* **`/RAG Engine` (Python FastAPI):** The heavy-lifting backend. It orchestrates GitHub cloning, chunking, embeddings, Neo4j Graph interactions, and LLM evaluations.
* **`/WebApp` (React + Vite + Tailwind):** The sleek frontend UI that handles conversational queries, codebase uploads, and side-by-side RAG evaluations.

### The Two Contenders:
1. **Vector-RAG:** Uses traditional text chunking, Gemini API Embeddings, and ChromaDB to find semantically similar code snippets.
2. **Graph-RAG:** Uses `tree-sitter` to parse the Abstract Syntax Tree (AST) of the uploaded codebase, supporting **10+ languages** simultaneously (Java, C++, Python, JavaScript, etc.). It maps files, classes, functions, and cross-file calls into a **Neo4j** graph database, querying it via LLM-generated Cypher.

---

## 🚀 Getting Started

### Prerequisites
* **Python 3.10+**
* **Node.js 18+**
* **Neo4j Desktop / Aura** (Running locally or in the cloud)
* API Keys for **Google Gemini** and **OpenRouter** (For Ragas evaluation)

### 1. Environment Setup
Inside the `/RAG Engine` directory, create a `.env` file with the following credentials:
```env
# Gemini API (For Embeddings and Graph-RAG generation)
GEMINI_API_KEY=your_gemini_api_key

# OpenRouter API (For Ragas Hallucination Evaluation)
OPENROUTER_API_KEY=your_openrouter_api_key

# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 2. Start the Backend (RAG Engine)
Open a terminal and navigate to the backend folder:
```bash
cd "RAG Engine"
pip install -r requirements_multilang_graph_rag.txt
python api.py
```
*The FastAPI server will start on `http://127.0.0.1:8000`.*

### 3. Start the Frontend (WebApp)
Open a second terminal and navigate to the frontend folder:
```bash
cd WebApp
npm install
npm run dev
```
*The Vite development server will start on `http://localhost:5173`.*

---

## 🧪 Testing the Pipelines

1. **Ingest a Codebase:** Open the WebApp and paste a GitHub URL (e.g., `https://github.com/expressjs/cors`). The backend will automatically wipe the old database, clone the new repo, and build both the Vector and Graph databases.
2. **Run a Query:** Ask a structural question like *"What are the main functions exported in this project?"*
3. **Compare Results:** Watch as both Vector-RAG and Graph-RAG generate answers, followed by the RAGAS evaluator assigning a hallucination score to each!

---

## 🛠️ Built With
* **Backend:** FastAPI, LangChain, Tree-sitter, Neo4j, ChromaDB, RAGAS
* **Frontend:** React, Vite, Tailwind CSS
* **LLMs:** Gemini 1.5/3.6 Flash (Generation & Embeddings), OpenRouter (Evaluation)
