import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv() 

print("Initializing Vector Data Ingestion...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

docs = []
for root, _, files in os.walk("data"):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', '.bin', '.exe', '.dll')):
            continue
        path = os.path.join(root, file)
        try:
            docs.extend(TextLoader(path, encoding="utf-8").load())
        except Exception as e:
            print(f"Skipping {file} due to read error.")

# Extension to LangChain Language mapping
LANG_MAPPING = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".cpp": Language.CPP,
    ".c": Language.CPP,
    ".cs": Language.CSHARP,
}

splits = []
default_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

print(f"Loaded {len(docs)} documents. Applying AST/Language-aware chunking...")

for doc in docs:
    ext = Path(doc.metadata.get("source", "")).suffix.lower()
    lang = LANG_MAPPING.get(ext)
    
    if lang:
        # Use AST/Language-aware chunking for code files
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=800, chunk_overlap=100
        )
        splits.extend(splitter.split_documents([doc]))
    else:
        # Fallback to default chunking for markdown, text, unknown
        splits.extend(default_splitter.split_documents([doc]))

import time

print(f"Created {len(splits)} chunks. Storing in ChromaDB...")

# Store chunks in ChromaDB and persist to disk in batches to avoid Gemini Free Tier rate limits
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

batch_size = 25
for i in range(0, len(splits), batch_size):
    batch = splits[i:i + batch_size]
    print(f"Ingesting batch {i // batch_size + 1}/{(len(splits) // batch_size) + 1}...")
    try:
        vectorstore.add_documents(batch)
    except Exception as e:
        print(f"Hit rate limit! Pausing for 60 seconds before retrying... ({e})")
        time.sleep(60)
        vectorstore.add_documents(batch)
    time.sleep(5)  # Increased delay to respect the 100 requests/minute quota

print("Ingestion complete. Database saved to ./chroma_db")