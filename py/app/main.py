


from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel
import git
import os
from sqlalchemy import create_engine, Column, Integer, Text, ForeignKey, Float, TIMESTAMP
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy import text
from datetime import datetime
import numpy as np
from pgvector.sqlalchemy import Vector
import shutil

from sqlalchemy import func
from sqlalchemy import event

import chardet  # For character encoding detection


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Database setup
#DATABASE_URL = "sqlite:///./test.db"
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_SIZE = 3072


def configure_session(dbapi_connection, connection_record):
    with dbapi_connection.cursor() as cursor:
        # Set the Ollama host configuration
        cursor.execute("SELECT set_config('ai.ollama_host', 'http://ollama:11434', false);")
        # Set the OpenAI API key configuration
        cursor.execute("SELECT set_config('ai.openai_api_key', %s, false);", (OPENAI_API_KEY,))

def init_vector_extension():
    with engine.connect() as connection:
        connection.execution_options(isolation_level="AUTOCOMMIT").execute(text("CREATE EXTENSION IF NOT EXISTS ai CASCADE;"))


init_vector_extension()

# Attach the event listener to the engine
event.listen(engine, "connect", configure_session)

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

class Embedding(Base):
    __tablename__ = 'embeddings'
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'))
    embedding = Column(Vector(3072)) # Column(Vector(768))  # Change from Text to Vector(768)
    chunk = Column(Text)
    file = relationship("File", back_populates="embeddings")


Repository.files = relationship("File", order_by=File.id, back_populates="repo")
File.embeddings = relationship("Embedding", order_by=Embedding.id, back_populates="file")

Base.metadata.create_all(bind=engine)

# Embedding model placeholder
def generate_embedding(text):
    # Replace with actual model call
    return np.random.rand(768).tolist()

# API Models
class RepoInDB(BaseModel):
    url: str

class Question(BaseModel):
    query: str

# Helper functions
def clone_repository(repo_url: str, repo_name: str):
    repo_path = f"./repos/{repo_name}"
    # Check if the repository path already exists and delete it if it's not empty
    if os.path.exists(repo_path) and os.path.isdir(repo_path):
        shutil.rmtree(repo_path)
    # Clone the repository after ensuring the directory is removed
    git.Repo.clone_from(repo_url, repo_path)
    return repo_path

def insert_embedding(db, file_id: int, chunk: str):
    """
    Insert a single embedding into the database.
    
    Args:
        db: SQLAlchemy database session
        file_id (int): ID of the file this embedding belongs to
        chunk (str): Text chunk to embed
        api_key (str): OpenAI API key
        
    Returns:
        None
        
    Raises:
        SQLAlchemyError: If there's a database error
    """
    insert_query = text("""
        INSERT INTO embeddings (file_id, embedding, chunk)
        VALUES (
            :file_id,
            ai.openai_embed('text-embedding-3-large', :chunk_text, :api_key),
            :chunk_text
        )
    """)
    
    params = {
        "file_id": file_id,
        "chunk_text": chunk,
        "api_key": OPENAI_API_KEY
    }
    
    try:
        db.execute(insert_query, params)
    except Exception as e:
        print(f"Error inserting embedding: {str(e)}")
        raise

def insert_embedding_(db, file_id: int, chunk: str):



    insert_query = text("""                    
        INSERT INTO embeddings (file_id, embedding, chunk)
        VALUES (
            :file_id,
            ai.ollama_embed('mxbai-embed-large', :chunk_text),
            :chunk_text
        )
    """)

    db.execute(insert_query, {"file_id": file_id, "chunk_text": chunk})

def is_binary_file(file_path):
    """
    Checks if a file is binary by reading a portion of its content.
    
    Args:
        file_path (str): Path to the file to check.
    
    Returns:
        bool: True if the file is binary, False otherwise.
    """
    with open(file_path, 'rb') as file:
        # Read a small portion of the file for checking
        initial_bytes = file.read(1024)
        # Use chardet to detect if content is likely text
        result = chardet.detect(initial_bytes)
        # If confidence is high and encoding is detected, assume it's a text file
        return result['encoding'] is None  # Binary if no encoding detected

# Modify process_repo to use ollama_embed in SQL with checks for existing chunks
def process_repo(repo_path: str, repo_id: int, db):
    file_count = 0  # Initialize a counter for processed files

    for root, dirs, files in os.walk(repo_path):
        # Exclude .git directories from processing
        dirs[:] = [d for d in dirs if not d.startswith('.git')]

        for file_name in files:
            # Stop if the file limit is reached
            if file_count >= 100:
                print("Reached the limit of 100 files. Stopping processing.")
                return

            file_path = os.path.join(root, file_name)
            
            # Check if the file is binary and skip if so
            if is_binary_file(file_path):
                print(f"Skipping binary file: {file_path}")
                continue

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                sanitized_content = content.replace('\x00', '')  # Remove null characters from content
                
                # Create the file record if it does not exist
                file_record = File(repo_id=repo_id, file_path=file_path, content=sanitized_content)
                db.add(file_record)
                db.commit()

                # Increment the file counter
                file_count += 1

                # Chunk the sanitized content and check each chunk
                chunks = [sanitized_content[i:i+500] for i in range(0, len(sanitized_content), 500)]
                for chunk in chunks:
                    # Check if this chunk already exists for this file in the embeddings table
                    existing_chunk = db.query(Embedding).filter_by(file_id=file_record.id, chunk=chunk).first()
                    
                    if existing_chunk:
                        print(f"Skipping existing chunk for file {file_path}.")
                        continue
                    
                    # Insert embedding if the chunk doesn't already exist
                    insert_embedding(db, file_record.id, chunk)

    db.commit()



# APIs
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("static", "index.html"))


@app.post("/insert_repo/")
async def insert_repo(repo: RepoInDB):
    db = SessionLocal()
    try:
        repo_name = repo.url.split("/")[-1].replace(".git", "")
        
        # Check if the repository already exists
        existing_repo = db.query(Repository).filter_by(name=repo_name).first()
        if existing_repo:
            raise HTTPException(status_code=400, detail=f"Repository '{repo_name}' already exists.")
        
        # If not, proceed to add the new repository
        db_repo = Repository(name=repo_name, url=repo.url)
        db.add(db_repo)
        db.commit()
        
        repo_path = clone_repository(repo.url, repo_name)
        process_repo(repo_path, db_repo.id, db)
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
        # Generate query embedding using ai.openai_embed and cast as vector
        similarity_query = text("""
            WITH query_embedding AS (
                SELECT ai.openai_embed('text-embedding-3-large', :query_text, :api_key)::vector(3072) AS embedding
            )
            SELECT e.chunk,
                   1 - (e.embedding <=> (SELECT embedding FROM query_embedding)) AS similarity,
                   e.file_id
            FROM embeddings e
            WHERE 1 - (e.embedding <=> (SELECT embedding FROM query_embedding)) > 0.7
            ORDER BY similarity DESC
            LIMIT 5
        """)

        # Execute the query
        result = db.execute(similarity_query, {
            "query_text": question.query,
            "api_key": OPENAI_API_KEY
        }).fetchall()

        # Format the response text
        response_text = "\n".join([f"Related Content: {row.chunk} (Score: {row.similarity:.2f})" for row in result])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

    return {"response": response_text}


@app.post("/ask_question/")
async def ask_question(question: Question):
    db = SessionLocal()
    try:
        # Step 1: Refine the question using gpt-4o-mini
        refine_query = text("""
            SELECT ai.openai_chat_complete(
                'gpt-4o-mini',
                jsonb_build_array(
                    jsonb_build_object('role', 'system', 'content', 
                        'I have embedded content from a Git repository and need to answer a specific user question. Based on the user’s intent, please rephrase the question to specifically align the user intent with the context of a Git repository: main code modules, features, and intended use cases, using specific terminology.'),
                    jsonb_build_object('role', 'user', 'content', :original_question)
                ),
                api_key=>:api_key
            )->'choices'->0->'message'->>'content' AS refined_question
        """)

        # Execute the question refinement
        refined_question = db.execute(refine_query, {
            "original_question": question.query,
            "api_key": OPENAI_API_KEY
        }).scalar()

        print("Refined Question:", refined_question)

        # First, retrieve relevant chunks using vector similarity
        similarity_query = text("""
            WITH query_embedding AS (
                SELECT ai.openai_embed('text-embedding-3-large', :query_text, :api_key)::vector(3072) AS embedding
            )
            SELECT e.chunk,
                1 - (e.embedding <#> (SELECT embedding FROM query_embedding)) AS similarity,
                f.file_path
            FROM embeddings e
            JOIN files f ON e.file_id = f.id
            WHERE 1 - (e.embedding <#> (SELECT embedding FROM query_embedding)) > 0.7
            ORDER BY similarity DESC
            LIMIT 5
        """)

        # Execute the similarity search with the refined question
        chunks = db.execute(similarity_query, {
            "query_text": refined_question,
            "api_key": OPENAI_API_KEY
        }).fetchall()

        # print("Chunks:", chunks)

        # Prepare context from retrieved chunks
        context = "\n".join([
            f"[From {row.file_path}]: {row.chunk}" 
            for row in chunks
        ])

        # print("Context:", context)

        # Generate LLM response using the context
        llm_query = text("""
            SELECT ai.openai_chat(
                messages=>ARRAY[
                    '{"role": "system", "content": "You are a helpful assistant that answers questions about code repositories. Base your answers only on the provided context. If you cannot answer based on the context, say so clearly."}'::jsonb,
                    jsonb_build_object(
                        'role', 'user',
                        'content', format(
                            'Context from the repository:\n%s\n\nQuestion: %s\n\nBased on the above context, please provide a concise and accurate answer.',
                            :context,
                            :question
                        )
                    )
                ],
                model=>'gpt-4o-2024-08-06',
                temperature=>0.7,
                api_key=>:api_key
            ) as response
        """)

        llm_query = text("""
            SELECT ai.openai_chat_complete(
                'gpt-4o-2024-08-06',
                jsonb_build_array(
                    jsonb_build_object('role', 'system', 'content', 
                        'You are a helpful assistant that answers questions about code repositories. Base your answers only on the provided context. If you cannot answer based on the context, say so clearly.'),
                    jsonb_build_object('role', 'user', 'content', format(
                        'Context from the repository:\n%s\n\nQuestion: %s\n\nBased on the above context, please provide a concise and accurate answer.',
                        :context,
                        :question
                    ))
                ),
                api_key=>:api_key
            )->'choices'->0->'message'->>'content' AS response
        """)


        # Execute the LLM query
        llm_result = db.execute(llm_query, {
            "context": context,
            "question": question.query,
            "api_key": OPENAI_API_KEY
        }).scalar()

        # Format the complete response
        response = {
            "llm_response": llm_result,
            "relevant_chunks": [
                {
                    "content": row.chunk,
                    "file_path": row.file_path,
                    "similarity_score": float(row.similarity)
                }
                for row in chunks
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

    return response

@app.get("/get_repositories")
async def get_repositories():
    db = SessionLocal()
    try:
        repos = db.query(Repository).all()
        return {"repositories": [{"id": repo.id, "name": repo.name} for repo in repos]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/initialize_db/")
async def initialize_db():
    Base.metadata.create_all(bind=engine)
    return {"message": "Database initialized"}

@app.get("/inspect_data/")
async def inspect_data():
    db = SessionLocal()
    try:
        # Query to get the count of chunks for each repository
        result = (
            db.query(Repository.name, func.count(Embedding.id).label("chunk_count"))
            .join(File, Repository.id == File.repo_id)
            .join(Embedding, File.id == Embedding.file_id)
            .group_by(Repository.name)
            .all()
        )

        # Format the result as a list of dictionaries
        response_data = [
            {"repository_name": repo_name, "chunk_count": chunk_count}
            for repo_name, chunk_count in result
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

    return {"repositories": response_data}


# Run the FastAPI app with Uvicorn (in Dockerfile or command line)
# uvicorn main:app --host 0.0.0.0 --port 8000
