import json
from vector_RAG_callable import run_vector_rag
# NEW: Import your newly created AST Graph RAG function
from ast_graph_rag_callable import run_ast_graph_rag 
from evaluate_hallucination import evaluate_rag_output

# Define the complex multi-hop query
test_query = "Which components are rendered inside App.jsx, and what props are passed to them?"

print(f"Running Showdown for Query: '{test_query}'\n")

# ==========================================
# 1. Vector-RAG Evaluation
# ==========================================
print("Fetching Vector-RAG response...")
v_context, v_answer = run_vector_rag(test_query)
vector_report = evaluate_rag_output("Vector-RAG", test_query, v_context, v_answer)

# ==========================================
# 2. AST Graph-RAG Evaluation
# ==========================================
print("Fetching AST Graph-RAG response...")
# NEW: Call the deterministic AST function
g_context, g_answer = run_ast_graph_rag(test_query) 
graph_report = evaluate_rag_output("AST-Graph-RAG", test_query, g_context, g_answer)

# ==========================================
# 3. Final Hallucination Matrix
# ==========================================
print("\n" + "="*50)
print("--- SHOWDOWN EVALUATION REPORT ---")
print("="*50)

final_results = [vector_report, graph_report]
print(json.dumps(final_results, indent=2))

# Save it to a file to easily copy-paste into your project report
with open("evaluation_results.json", "w") as f:
    json.dump(final_results, f, indent=2)