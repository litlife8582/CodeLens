import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph, LLMGraphTransformer, GraphCypherQAChain

load_dotenv(override=True)

load_dotenv(override=True)
print("URI:", repr(os.getenv("NEO4J_URI")))
print("USER:", repr(os.getenv("NEO4J_USERNAME")))
print("PASS LEN:", len(os.getenv("NEO4J_PASSWORD") or ""))
print("PASS REPR:", repr(os.getenv("NEO4J_PASSWORD")))

print("Connecting to Neo4j...")
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE")
)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.0)

print("Loading codebase from 'data/' directory...")
loader = DirectoryLoader(
    "data", 
    glob="**/*.*", 
    loader_cls=TextLoader, 
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

print(f"Extracting graph entities and relationships from {len(splits)} chunks...")

transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Component", "Hook", "File", "CSS_Class", "Variable"],
    allowed_relationships=["IMPORTS", "RENDERS", "USES_HOOK", "STYLED_BY", "DEPENDS_ON"]
)
graph_documents = transformer.convert_to_graph_documents(splits)

print("Saving Knowledge Graph to Neo4j...")
graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

print("Setting up Graph RAG Query Engine...")
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)

print("\n--- Graph-RAG System Ready ---")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    query = input("\nAsk a graph-based question: ")
    if query.lower() in ['exit', 'quit']:
        print("Shutting down...")
        break
    if not query.strip():
        continue
        
    print(f"\nTranslating to Cypher and traversing graph...\n")
    try:
        response = chain.invoke({"query": query})
        print("\n--- ANSWER ---")
        print(response["result"])
    except Exception as e:
        print(f"\nAn error occurred: {e}")

    print("\n" + "="*50)