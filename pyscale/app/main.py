from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import git
import os
from sqlalchemy import create_engine, Column, Integer, Text, ForeignKey, TIMESTAMP
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy import text
from datetime import datetime
import shutil
from sqlalchemy import func
from sqlalchemy import event
import chardet

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

def init_extensions():
    with engine.connect() as connection:
        # Enable required extensions
        connection.execution_options(isolation_level="AUTOCOMMIT").execute(text("""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE EXTENSION IF NOT EXISTS ai;
            CREATE EXTENSION IF NOT EXISTS pgai;
        """))

# Initialize extensions
init_extensions()

class Repository(Base):
    __tablename__ = 'repositories'
    id = Column(Integer, primary_key=True)
    name = Column(Text, unique=True)
    url = Column(Text)
    cloned_at = Column(TIMESTAMP, default=datetime.utcnow)

class File(Base):
    __tablename__ = 'files'
    id = Column(Integer, primary_key=True)
    repo_id = Column(Integer, ForeignKey('repositories.id'))
    file_path = Column(Text)
    content = Column(Text)
    repo = relationship("Repository", back_populates="files")

Repository.files = relationship("File", order_by=File.id, back_populates="repo")

# Create tables
Base.metadata.create_all(bind=engine)

def setup_vectorizer(db):
    """Set up the vectorizer for the files table"""
    create_vectorizer_query = text("""
    SELECT ai.create_vectorizer(
        'files'::regclass,
        destination => 'file_embeddings',
        embedding => ai.embedding_openai('text-embedding-3-large', 3072),
        chunking => ai.chunking_recursive_character_text_splitter(
            'content',
            chunk_size => 500,
            chunk_overlap => 50
        ),
        formatting => ai.formatting_python_template('File: $file_path\n\nContent: $chunk'),
        indexing => ai.indexing_hnsw(
            min_rows => 1000,
            opclass => 'vector_cosine_ops'
        )
    );
    """)
    
    try:
        db.execute(create_vectorizer_query)
        db.commit()
    except Exception as e:
        print(f"Error setting up vectorizer: {str(e)}")
        db.rollback()
        raise

def is_binary_file(file_path):
    """Check if a file is binary"""
    with open(file_path, 'rb') as file:
        initial_bytes = file.read(1024)
        result = chardet.detect(initial_bytes)
        return result['encoding'] is None

def process_repo(repo_path: str, repo_id: int, db):
    """Process repository files - simplified as vectorizer handles embeddings"""
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.git')]
        
        for file_name in files:
            file_path = os.path.join(root, file_name)
            
            if is_binary_file(file_path):
                print(f"Skipping binary file: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    sanitized_content = content.replace('\x00', '')
                    
                    file_record = File(
                        repo_id=repo_id,
                        file_path=file_path,
                        content=sanitized_content
                    )
                    db.add(file_record)
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue
    
    db.commit()

class RepoInDB(BaseModel):
    url: str

class Question(BaseModel):
    query: str

def clone_repository(repo_url: str, repo_name: str):
    repo_path = f"./repos/{repo_name}"
    if os.path.exists(repo_path) and os.path.isdir(repo_path):
        shutil.rmtree(repo_path)
    git.Repo.clone_from(repo_url, repo_path)
    return repo_path

@app.post("/insert_repo/")
async def insert_repo(repo: RepoInDB):
    db = SessionLocal()
    try:
        repo_name = repo.url.split("/")[-1].replace(".git", "")
        
        existing_repo = db.query(Repository).filter_by(name=repo_name).first()
        if existing_repo:
            raise HTTPException(status_code=400, detail=f"Repository '{repo_name}' already exists.")
        
        db_repo = Repository(name=repo_name, url=repo.url)
        db.add(db_repo)
        db.commit()
        
        repo_path = clone_repository(repo.url, repo_name)
        process_repo(repo_path, db_repo.id, db)
        
        # Ensure vectorizer is set up
        setup_vectorizer(db)
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
    return {"message": "Repository inserted successfully"}

@app.post("/search/")
async def search(question: Question):
    db = SessionLocal()
    try:
        similarity_query = text("""
            SELECT 
                chunk,
                1 - (embedding <=> ai.openai_embed('text-embedding-3-large', :query_text)) AS similarity,
                file_path
            FROM file_embeddings
            WHERE 1 - (embedding <=> ai.openai_embed('text-embedding-3-large', :query_text)) > 0.7
            ORDER BY similarity DESC
            LIMIT 5
        """)

        result = db.execute(similarity_query, {
            "query_text": question.query
        }).fetchall()

        response_text = "\n".join([
            f"File: {row.file_path}\nContent: {row.chunk}\nRelevance: {row.similarity:.2f}\n"
            for row in result
        ])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

    return {"response": response_text}

@app.post("/ask_question/")
async def ask_question(question: Question):
    db = SessionLocal()
    try:
        # Use vectorizer for similarity search
        similarity_query = text("""
            WITH context AS (
                SELECT 
                    chunk,
                    file_path,
                    1 - (embedding <=> ai.openai_embed('text-embedding-3-large', :query_text)) AS similarity
                FROM file_embeddings
                WHERE 1 - (embedding <=> ai.openai_embed('text-embedding-3-large', :query_text)) > 0.7
                ORDER BY similarity DESC
                LIMIT 5
            )
            SELECT ai.openai_chat_complete(
                'gpt-4o-2024-08-06',
                jsonb_build_array(
                    jsonb_build_object('role', 'system', 'content', 
                        'You are a helpful assistant that answers questions about code repositories. Base your answers only on the provided context. If you cannot answer based on the context, say so clearly.'),
                    jsonb_build_object('role', 'user', 'content', format(
                        'Context from the repository:\n%s\n\nQuestion: %s\n\nBased on the above context, please provide a concise and accurate answer.',
                        string_agg(format('From %s: %s', file_path, chunk), E'\n'),
                        :question
                    ))
                )
            )->'choices'->0->'message'->>'content' AS response
            FROM context
            GROUP BY similarity
        """)

        result = db.execute(similarity_query, {
            "query_text": question.query,
            "question": question.query
        }).fetchall()

        if not result:
            return {"response": "No relevant information found in the repository."}

        return {"response": result[0][0]}  # Return the first (most relevant) response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/vectorizer_status")
async def get_vectorizer_status():
    db = SessionLocal()
    try:
        status_query = text("SELECT * FROM ai.vectorizer_status;")
        result = db.execute(status_query).fetchall()
        return {"status": [dict(row) for row in result]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()