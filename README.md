# Project RAG - Backend API

A robust, high-performance REST API built with FastAPI that powers the Retrieval-Augmented Generation (RAG) capabilities of the application. This backend handles user authentication, vectorizing uploaded documents, persistent chat histories, and executing secure, workspace-isolated semantic searches.

## 🌟 Key Features

- **FastAPI Foundation:** Asynchronous, high-speed API endpoints with automatic interactive documentation (Swagger UI).
- **Multi-Tenant Workspaces:** Data level isolation so users only query context within their active workspace.
- **Advanced RAG Pipeline:** Powered by **LangChain** and **HuggingFace** (`all-MiniLM-L6-v2`) to chunk, embed, and retrieve document contexts accurately.
- **Vector Database Integration:** Built on **PostgreSQL** with the **pgvector** extension (via Supabase) for fast similarity searches.
- **JWT Authentication:** Strict middleware validating Supabase access tokens to secure all sensitive endpoints.
- **Relational Persistence:** Automatically saving documents and chat messages to Postgres using **SQLAlchemy**.

## 🛠️ Technology Stack

- **Core Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **AI & Retrieval:** [LangChain](https://python.langchain.com/) + HuggingFace Embeddings
- **Database & ORM:** PostgreSQL + [pgvector](https://github.com/pgvector/pgvector), [SQLAlchemy](https://www.sqlalchemy.org/)
- **Authentication:** `python-jose` (PyJWT) validating Supabase JWTs
- **Document Processing:** `PyPDFLoader`

## ⚙️ Setup and Installation Guide

### Prerequisites
Make sure you have [Python 3.9+](https://www.python.org/) installed on your system. You will also need a Supabase PostgreSQL database with the `pgvector` extension enabled.

### 1. Create a Virtual Environment
Navigate to the backend directory and set up an isolated Python environment:
```bash
# In the project_rag/backend directory
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Dependencies
Install the required Python packages into your active virtual environment:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
You need to connect the backend to your exact Supabase PostgreSQL instance.

Create a `.env` file in the root of the `backend` folder (this file is git-ignored) and add your database URL:
```env
# The connection string to your Supabase PostgreSQL database
# Usually in the format: postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
DATABASE_URL=your_supabase_postgres_connection_string
```

### 4. Initialize Database Models
Ensure your Supabase database has all tables created correctly. (You can run any local DB setup or migration scripts if provided, or SQLAlchemy will auto-create standard tables on first connect depending on `Base.metadata.create_all(bind=engine)` logic).

### 5. Run the Server
Start the local FastAPI development server using Uvicorn:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. You can test endpoints directly and interactively via the built-in Swagger UI at `http://localhost:8000/docs`.
 
