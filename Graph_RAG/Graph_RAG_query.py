import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

print("Initializing AST Graph-RAG Retrieval Pipeline...")

# 1. Load Credentials & Connect to Neo4j
load_dotenv(override=True)
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD")
)

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template="""
You are a Cypher query generator for an AST-based React code knowledge graph.

Generate a READ-ONLY Cypher query.

IMPORTANT GRAPH STRUCTURE:

A Component is connected to a RenderSite using this relationship:

(:Component)-[:RENDERS]->(:RenderSite)

The Component is the PARENT component that contains the JSX.

The RenderSite represents the ACTUAL JSX ELEMENT being rendered.

IMPORTANT PROPERTIES:

Component properties include:
- name
- path
- declaration_type
- line
- column
- id

RenderSite properties include:
- name
- file
- line
- column
- source
- id

INTERPRETATION:

The Component node is the component doing the rendering.

The RenderSite node represents the component being rendered.

For example, conceptually:

App --RENDERS--> RenderSite

where:
- App is the parent component
- RenderSite.name is Header
- RenderSite.source is the JSX source
- RenderSite.file is app/src/App.jsx

Therefore:

c.name = parent component

r.name = rendered component

r.source = actual JSX source and contains information about props

r.file = file containing the render

r.line = line where the render occurs


CRITICAL RULE FOR RENDER QUESTIONS:

If the question asks:

Which components are rendered inside App.jsx?

DO NOT return c.name as the rendered component.

Return r.name.

If the question asks:

Which components are rendered inside App.jsx, and what props are passed to them?

Return:
- r.name
- r.source
- r.line

Use this query pattern:

MATCH (c:Component)-[:RENDERS]->(r:RenderSite)
WHERE r.file ENDS WITH 'App.jsx'
RETURN
    c.name AS parent_component,
    r.name AS rendered_component,
    r.line AS render_line,
    r.source AS jsx
ORDER BY r.line


IMPORTANT:

Do NOT return c.declaration_type as props.

Do NOT treat Component.declaration_type as prop information.

Do NOT return only c.name when the question asks which component is rendered.

Do NOT invent labels, properties, or relationships.

Use only the labels, properties, and relationships in the provided schema.

Generate ONLY the Cypher query.
Do not include Markdown fences.
Do not include explanations.

Schema:

{schema}

Question:

{question}
"""
)

# 2. Initialize the LLM
#llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.0)
llm=ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# 3. Create the Query Engine with Context Extraction
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    verbose=True,
    allow_dangerous_requests=True,
    return_intermediate_steps=True
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
    test_query =( "Which components are rendered inside App.jsx, "
    "and what props are passed to them?"
    )
    print(f"\nTesting Query: '{test_query}'")
    
    retrieved_context, final_answer = run_ast_graph_rag(test_query)
    
    print("\n--- RETRIEVED AST CONTEXT ---")
    print(retrieved_context)
    print("\n--- GENERATED ANSWER ---")
    print(final_answer)