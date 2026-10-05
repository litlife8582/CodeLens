import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Import both RAGs for the Hybrid Approach
from Vector_RAG.Vector_RAG_query import run_vector_rag
from Graph_RAG.Graph_RAG_query import run_ast_graph_rag

load_dotenv(override=True)

llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

code_generation_prompt = PromptTemplate(
    input_variables=["vector_context", "graph_context", "feature_request"],
    template="""
    You are an expert software engineer. 
    You have access to two types of context from the codebase to help you satisfy the user's feature request:

    1. Semantic/Fuzzy Context (Vector DB):
    {vector_context}
    
    2. Structural Code Graph Context (AST Neo4j):
    {graph_context}
    
    Feature Request:
    {feature_request}

    Using BOTH contexts, generate the exact code needed. 
    Return ONLY the updated or new code in Markdown format with the appropriate language tags. 
    If the context is insufficient, explain what additional context you need.
    """
)

code_chain = code_generation_prompt | llm

def generate_feature_code(feature_request: str) -> str:
    print(f"Executing Hybrid-RAG for: {feature_request}")
    
    # 1. Fetch semantic context from Vector DB
    print("- Fetching Semantic Context (Vector)...")
    vector_context, _ = run_vector_rag(feature_request)
    
    # 2. Fetch structural context from Graph DB
    print("- Fetching Structural Context (Graph)...")
    rag_query = f"Find all components, render sites, and files relevant to implementing: {feature_request}. Return their properties and source."
    graph_context, _ = run_ast_graph_rag(rag_query)
    
    # 3. Generate the code using both contexts
    print("- Generating Code via LLM...")
    try:
        response = code_chain.invoke({
            "vector_context": vector_context,
            "graph_context": graph_context,
            "feature_request": feature_request
        })
        return response.content
    except Exception as e:
        print(f"Error in code generation: {e}")
        return f"An error occurred while generating code: {e}"

if __name__ == "__main__":
    test_request = input("Enter a feature request: ")
    print("\nResult:\n")
    print(generate_feature_code(test_request))
