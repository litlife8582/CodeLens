import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()   

print("Initializing Vector-RAG Pipeline...")

# 1. Initialize LLM and Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.0)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# 2. Loading data
loader = DirectoryLoader(
    "data", 
    glob="**/*.*", 
    loader_cls=TextLoader, 
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()

# 3. Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 4. Storing chunks in ChromaDB
persist_directory = "./chroma_db"
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings,
    persist_directory=persist_directory
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Setting up prompt template
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 6. Build Retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ==========================================
# Callable Integration Function
# ==========================================
def run_vector_rag(query: str):
    """
    Executes a query against the Vector-RAG pipeline.
    Returns the retrieved context as a single string and the generated answer.
    """
    response = rag_chain.invoke({"input": query})
    
    # Extract the final answer string
    answer = response["answer"]
    
    # Combine the retrieved document chunks into a single formatted string for the evaluator
    context_chunks = "\n---\n".join([doc.page_content for doc in response["context"]])
    
    return context_chunks, answer

# Optional: Run a quick test if executing this script directly
if __name__ == "__main__":
    print("\n--- Vector-RAG System Ready ---")
    test_query = "Which components are rendered inside App.jsx, and what props are passed to them?"
    print(f"Testing Query: {test_query}\n")
    
    retrieved_context, final_answer = run_vector_rag(test_query)
    print("--- GENERATED ANSWER ---")
    print(final_answer)