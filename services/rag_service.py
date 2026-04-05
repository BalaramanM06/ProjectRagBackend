import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from core.database import engine

# ==========================================
# 1. Setup Embeddings
# ==========================================
# We use a free huggingface model that is highly efficient for document similarity
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ==========================================
# 2. Setup VectorStore (PostgreSQL pgvector)
# ==========================================
collection_name = "pdf_rag_collection"

# Connection parameter uses the SQLAlchemy engine we built in core/database.py
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=engine,
    use_jsonb=True,
)


# ==========================================
# 3. Setup LLM (Groq)
# ==========================================
# Ensure we instantiate exactly when needed to catch env keys if they load late
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile", 
        api_key=os.getenv("GROQ_API_KEY")
    )


# ==========================================
# 4. Core Features
# ==========================================
def process_and_store_pdf(file_path: str, workspace_id: int) -> int:
    """
    Loads a PDF, splits it into chunks, embedded them, and stores them in PostgreSQL.
    Returns the number of chunks processed.
    """
    # Load the PDF Document
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Split the document into 1000-character chunks with a 200-character overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    # Add workspace_id metadata so we can isolate Vector searches 
    for split in splits:
        split.metadata["workspace_id"] = workspace_id
        
    # Store the chunks securely into Supabase Vector DB
    vectorstore.add_documents(splits)
    
    return len(splits)


def get_chat_response(message: str, workspace_id: int, history: list = None) -> str:
    """
    Takes a user question, retrieves the best documents from the Vector DB, 
    and uses Groq LLM to generate an answer.
    """
    llm = get_llm()
    
    # Format history string for simplicity
    history_str = ""
    if history:
        for role, content in history:
            history_str += f"{role.capitalize()}: {content}\n"

    # Create a retriever that pulls the Top 4 most relevant chunks
    # CRITICAL: Isolate context per workspace securely using metadata tag
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"workspace_id": workspace_id}
        }
    )
    
    system_prompt = (
        "You are an expert AI assistant that helps users understand their documents. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If you don't know the answer or the context doesn't contain the answer, "
        "just say that you don't know. Do not make up information. \n\n"
        "Here is the Chat History with the user so far:\n"
        "{history}\n\n"
        "Context from Documents: {context}"
    )
    
    prompt = PromptTemplate(
        template=system_prompt + "\n\nUser Question: {input}\nAnswer:", 
        input_variables=["context", "input", "history"]
    )
    
    # Create the RAG chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Invoke the chain with the user's question & history
    response = rag_chain.invoke({"input": message, "history": history_str})
    
    return response["answer"]
