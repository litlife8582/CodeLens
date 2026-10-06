import os
from langchain_chroma import Chroma
from tenacity import retry, wait_exponential, stop_after_attempt
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv() 

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.0)

# llm = ChatOpenAI(
#     model="openrouter/free",
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPENROUTER_API_KEY"),
# )
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# Connect directly to the existing database
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@retry(wait=wait_exponential(multiplier=2, min=10, max=65), stop=stop_after_attempt(5))
def run_vector_rag(query: str):
    try:
        response = rag_chain.invoke({"input": query})
        answer = response["answer"]
        context_chunks = "\n---\n".join([doc.page_content for doc in response["context"]])
        return context_chunks, answer

    except Exception as e:
        print(f"\nAPI Error encountered: {e}. Tenacity is initiating retry...")
        # Must re-raise so Tenacity catches it
        raise e