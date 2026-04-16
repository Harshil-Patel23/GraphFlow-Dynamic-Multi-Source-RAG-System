"""
Retriever setup and vector store configuration for Pinecone.
"""

import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from src.core.config import settings

# embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def _ensure_index_exists():
    """
    Ensure the Pinecone index exists, creating it if necessary.
    """
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    
    index_name = settings.PINECONE_INDEX_NAME
    
    # Check if index exists
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=384, # Dimensions for all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" # Default region, can be made configurable
            )
        )
        
        # Wait for index to be initialized
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"Index {index_name} created successfully.")
    else:
        print(f"Index {index_name} already exists.")


def retriever_chain(chunks: list[Document], session_id: str = "default"):
    """
    Initialize and store documents in Pinecone vector database.

    Vectors are stored in a namespace matching the session_id, ensuring
    that documents uploaded in one chat session are isolated from all others.

    Args:
        chunks: List of document chunks to store.
        session_id: Chat session UUID used as the Pinecone namespace.

    Returns:
        Boolean indicating success of the operation.
    """
    try:
        # Ensure index exists before storing
        _ensure_index_exists()

        # Store documents in Pinecone under session-specific namespace
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=settings.PINECONE_INDEX_NAME,
            namespace=session_id
        )

        print(f"Stored {len(chunks)} document chunks in Pinecone index: '{settings.PINECONE_INDEX_NAME}', namespace: '{session_id}'")
        return True
    except Exception as e:
        print(f"Error storing documents in Pinecone: {e}")
        return False


def get_retriever(session_id: str = "default"):
    """
    Get a retriever tool connected to the Pinecone vector store.

    The retriever is scoped to the given session_id namespace, so only
    documents uploaded in the same chat session are searchable.

    Args:
        session_id: Chat session UUID used as the Pinecone namespace.

    Returns:
        A LangChain retriever tool configured for the vector store.
    """
    try:
        # Ensure index exists before retrieving (in case it's the first run)
        _ensure_index_exists()

        # Connect to Pinecone index scoped to session namespace
        vectorstore = PineconeVectorStore(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=session_id
        )
        
        retriever = vectorstore.as_retriever()

        # Load document description
        if os.path.exists("description.txt"):
            with open("description.txt", "r", encoding="utf-8") as f:
                description = f.read()
        else:
            description = "uploaded documents"

        retriever_tool = create_retriever_tool(
            retriever,
            "retriever_customer_uploaded_documents",
            f"Use this tool **only** to answer questions about: {description}\n"
            "Don't use this tool to answer anything else."
        )

        return retriever_tool

    except Exception as e:
        print(f"Error initializing Pinecone retriever: {e}")
        # If it's the first run and no docs are uploaded, we might still want to return a tool or handle it gracefully
        raise Exception(e)


def delete_session_vectors(session_id: str) -> None:
    """
    Delete all vectors in a Pinecone namespace (session cleanup).

    Called when a conversation is deleted so orphaned embeddings don't
    accumulate and waste storage.

    Args:
        session_id: Chat session UUID whose namespace should be wiped.
    """
    try:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        index.delete(delete_all=True, namespace=session_id)
        print(f"Deleted all vectors in namespace: '{session_id}'")
    except Exception as e:
        print(f"Error deleting vectors for session {session_id}: {e}")
