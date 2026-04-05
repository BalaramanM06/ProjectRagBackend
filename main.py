from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import os
import shutil
from dotenv import load_dotenv

load_dotenv()

from core.database import engine, get_db, Base
import core.models as models
from core.auth import get_current_user

Base.metadata.create_all(bind=engine)

from services.rag_service import process_and_store_pdf, get_chat_response

app = FastAPI(
    title="RAG Chatbot API",
    description="Backend for PDF-based RAG Chatbot",
    version="1.0.0"
)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    workspace_id: int

class WorkspaceCreate(BaseModel):
    name: str

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running."}

@app.post("/workspaces")
def create_workspace(req: WorkspaceCreate, db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user)):
    ws = models.Workspace(user_id=current_user_id, name=req.name)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return {"id": ws.id, "name": ws.name, "created_at": ws.created_at}

@app.get("/workspaces")
def get_workspaces(db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user)):
    workspaces = db.query(models.Workspace).filter(models.Workspace.user_id == current_user_id).order_by(models.Workspace.created_at.desc()).all()
    return [{"id": w.id, "name": w.name, "created_at": w.created_at} for w in workspaces]

@app.delete("/workspaces/{workspace_id}")
def delete_workspace(workspace_id: int, db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user)):
    # 1. Verify Ownership
    ws = db.query(models.Workspace).filter(models.Workspace.id == workspace_id, models.Workspace.user_id == current_user_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    try:
        # 2. Delete related documents & UI chat messages
        db.query(models.Document).filter(models.Document.workspace_id == workspace_id).delete()
        db.query(models.ChatMessage).filter(models.ChatMessage.workspace_id == workspace_id).delete()
        
        # 3. Clean up the PGVector embeddings holding context chunks
        from sqlalchemy import text # Make sure we can run raw SQL
        db.execute(
            text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'workspace_id' = :wsid"),
            {"wsid": str(workspace_id)}
        )
        
        # 4. Finally delete the workspace
        db.delete(ws)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database cascade cleanup failed: {str(e)}")
        
    return {"message": "Workspace deleted successfully"}

@app.get("/documents")
def get_documents(workspace_id: int, db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user)):
    docs = db.query(models.Document).filter(
        models.Document.user_id == current_user_id,
        models.Document.workspace_id == workspace_id
    ).order_by(models.Document.upload_time.desc()).all()
    return [{"filename": doc.filename} for doc in docs]

@app.get("/chat/history")
def get_chat_history(workspace_id: int, db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user)):
    messages = db.query(models.ChatMessage).filter(
        models.ChatMessage.user_id == current_user_id,
        models.ChatMessage.workspace_id == workspace_id
    ).order_by(models.ChatMessage.timestamp.asc()).all()
    return [{"id": str(msg.id), "type": msg.role, "content": msg.content} for msg in messages]

@app.post("/upload")
async def upload_document(
    workspace_id: int, 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db), 
    current_user_id: str = Depends(get_current_user)
):
    """
    Endpoint strictly for uploading PDF documents to index into the RAG model.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # 1. Save the file temporarily to disk
    upload_dir = "data"
    os.makedirs(upload_dir, exist_ok=True)
    document_path = os.path.join(upload_dir, file.filename)
    
    with open(document_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. Verify Ownership of Workspace
        ws = db.query(models.Workspace).filter(models.Workspace.id == workspace_id, models.Workspace.user_id == current_user_id).first()
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found")

        # 2. Trigger the modular RAG Service to Process and Store in PostgreSQL
        chunks_processed = process_and_store_pdf(document_path, workspace_id)
        
        # 3. Save into DB
        new_doc = models.Document(
            filename=file.filename, 
            user_id=current_user_id,
            workspace_id=workspace_id
        )
        db.add(new_doc)
        db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    
    return {
        "message": "Document uploaded successfully", 
        "filename": file.filename,
        "chunks_indexed": chunks_processed
    }

@app.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user)):
    """
    Endpoint for querying the RAG chatbot with the user's message isolated to a specific workspace.
    """
    user_message = request.message
    workspace_id = request.workspace_id
    
    try:
        # Verify Ownership
        ws = db.query(models.Workspace).filter(models.Workspace.id == workspace_id, models.Workspace.user_id == current_user_id).first()
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found")

        # Save User Message to DB
        user_msg_db = models.ChatMessage(
            role="user", 
            content=user_message, 
            user_id=current_user_id,
            workspace_id=workspace_id
        )
        db.add(user_msg_db)
        db.commit()

        # Fetch chat history for LLM context, isolated by workspace
        history = db.query(models.ChatMessage).filter(
            models.ChatMessage.user_id == current_user_id,
            models.ChatMessage.workspace_id == workspace_id
        ).order_by(models.ChatMessage.timestamp.asc()).all()
        formatted_history = [(msg.role, msg.content) for msg in history]

        # Fetch AI response via Vector DB Retrieval + Groq LLM
        ai_reply = get_chat_response(user_message, workspace_id, formatted_history)

        # Save AI Message to DB
        ai_msg_db = models.ChatMessage(
            role="bot", 
            content=ai_reply, 
            user_id=current_user_id,
            workspace_id=workspace_id
        )
        db.add(ai_msg_db)
        db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")
    
    return {"reply": ai_reply}

