import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph, LLMGraphTransformer, GraphCypherQAChain

load_dotenv(override=True)

print("Initializing Graph-RAG Pipeline...")

# 1. Connect to Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE")
)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.0)

# 2. Load and Split Documents
loader = DirectoryLoader(
    "data", 
    glob="**/*.*", 
    loader_cls=TextLoader, 
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 3. Extract and Save Knowledge Graph
print(f"Extracting graph entities from {len(splits)} chunks...")
transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Component", "Hook", "File", "CSS_Class", "Variable"],
    allowed_relationships=["IMPORTS", "RENDERS", "USES_HOOK", "STYLED_BY", "DEPENDS_ON"]
)
graph_documents = transformer.convert_to_graph_documents(splits)
graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

# 4. Setup Query Engine with Intermediate Steps
print("Setting up Graph RAG Query Engine...")
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
def run_graph_rag(query: str):
    """
    Executes a query against the Graph-RAG pipeline.
    Returns the intermediate Cypher graph context and the generated answer.
    """
    try:
        response = chain.invoke({"query": query})
        
        # Extract the final answer string
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
    print("\n--- Graph-RAG System Ready ---")
    test_query = "Which components are rendered inside App.jsx, and what props are passed to them?"
    print(f"Testing Query: {test_query}\n")
    
    retrieved_context, final_answer = run_graph_rag(test_query)
    
    print("\n--- RETRIEVED GRAPH CONTEXT ---")
    print(retrieved_context)
    print("\n--- GENERATED ANSWER ---")
    print(final_answer)