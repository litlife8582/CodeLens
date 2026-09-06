import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

print("Initializing AST Graph-RAG Retrieval Pipeline...")

# 1. Load Credentials & Connect to Neo4j
load_dotenv(override=True)
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD")
)

# 2. Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.0)

# 3. Create the Query Engine with Context Extraction
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    return_intermediate_steps=True  # Crucial for evaluation context
)

# ==========================================
# Callable Integration Function
# ==========================================
def run_ast_graph_rag(query: str):
    """
    Executes a query against the AST-mapped Knowledge Graph.
    Returns the raw Cypher paths (context) and the generated answer.
    """
    try:
        response = chain.invoke({"query": query})
        
        # Extract the generated answer string
        answer = response.get("result", "I don't know the answer.")
        
        # Extract the raw graph traversal paths to serve as verifiable context
        intermediate_steps = response.get("intermediate_steps", [])
        context = str(intermediate_steps) if intermediate_steps else "No graph context retrieved."
        
        return context, answer
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return str(e), "Error generating answer."

# Optional: Run a quick test if executing this script directly
if __name__ == "__main__":
    test_query = "Which components are rendered inside App.jsx?"
    print(f"\nTesting Query: '{test_query}'")
    
    retrieved_context, final_answer = run_ast_graph_rag(test_query)
    
    print("\n--- RETRIEVED AST CONTEXT ---")
    print(retrieved_context)
    print("\n--- GENERATED ANSWER ---")
    print(final_answer)