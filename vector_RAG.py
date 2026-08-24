import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()   

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

#Loading data
print("Loading documents from 'data/' directory...")
loader = DirectoryLoader("data", glob="**/*.*")
docs = loader.load()

#text splitting
print(f"Loaded {len(docs)} document(s). Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

#storing chunks in chromaDB
print("Setting up ChromaDB...")
persist_directory = "./chroma_db"

vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings,
    persist_directory=persist_directory
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#setting up prompt template
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

#Retrieval chain
print("Building the retrieval chain...")
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#query execution
query = "What is the primary function of the auth module?"
print(f"\nExecuting Query: '{query}'\n")

response = rag_chain.invoke({"input": query})


#output
print("--- ANSWER ---")
print(response["answer"])

print("\n--- RETRIEVED CONTEXT CHUNKS ---")
for i, doc in enumerate(response["context"]):
    print(f"\n[Chunk {i+1}]")
    print(doc.page_content)