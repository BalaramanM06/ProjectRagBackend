from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from .database import Base

class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True) # ID from Supabase Auth
    workspace_id = Column(Integer, index=True) # Link to workspace
    filename = Column(String, index=True)
    upload_time = Column(DateTime, default=datetime.utcnow)

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True) # ID from Supabase Auth
    workspace_id = Column(Integer, index=True) # Link to workspace
    role = Column(String) # "user" or "ai"
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
