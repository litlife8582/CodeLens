import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv() 

print("Initializing Vector Data Ingestion...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

loader = DirectoryLoader("data", glob="**/*.*", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# Store chunks in ChromaDB and persist to disk
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Ingestion complete. Database saved to ./chroma_db")