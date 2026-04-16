"""
API routes for RAG operations and conversation management.

Endpoints:
  POST   /rag/query                          → run a RAG query in a named session
  POST   /rag/documents/upload               → upload a document (PDF/TXT)
  GET    /rag/conversations                  → list conversations for a user
  POST   /rag/conversations                  → create / update a conversation record
  GET    /rag/conversations/{session_id}/messages → load full message history
  DELETE /rag/conversations/{session_id}     → delete one conversation + its messages
  DELETE /rag/conversations/all              → clear ALL conversations for a user
"""

from fastapi import APIRouter, UploadFile, File, Header, Query
from langchain_core.messages import HumanMessage, AIMessage

from src.memory.chat_history_mongo import ChatHistory, MongoDBChatMessageHistory, collection as chat_collection
from src.memory.conversation_store import (
    list_conversations,
    upsert_conversation,
    delete_conversation,
    delete_all_conversations,
    ensure_indexes,
)
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder
from src.rag.retriever_setup import delete_session_vectors
from pydantic import BaseModel

router = APIRouter()


# ─── Startup: ensure MongoDB indexes exist ────────────────────────────────────

@router.on_event("startup")
async def _startup() -> None:
    """Create required MongoDB indexes when FastAPI starts."""
    await ensure_indexes()


# ─── RAG Query ────────────────────────────────────────────────────────────────

@router.post("/rag/query")
async def rag_query(req: QueryRequest):
    """
    Process a RAG query and return the generated response.

    The session_id (a UUID) identifies the conversation so messages are stored
    and retrieved correctly from MongoDB.  The first user message is also used
    as the conversation title if a username is supplied in the request.

    Args:
        req: QueryRequest containing 'query', 'session_id', and optionally
             'username' for conversation metadata.

    Returns:
        The last generated message from the RAG pipeline.
    """
    # Retrieve (or create) the MongoDB-backed history for this session
    chat_history: MongoDBChatMessageHistory = ChatHistory.get_session_history(req.session_id)

    # Persist the incoming user message
    await chat_history.add_message(HumanMessage(content=req.query))

    # Fetch the full conversation history to feed to the graph
    messages = await chat_history.get_messages()

    # Run the LangGraph RAG pipeline with the full history as context
    result = builder.invoke({"messages": messages, "session_id": req.session_id})
    output_text = result["messages"][-1].content

    # Persist the assistant's reply
    await chat_history.add_message(AIMessage(content=output_text))

    # ── Update conversation metadata ──────────────────────────────────────────
    # If username is provided, upsert a metadata record so the sidebar can list
    # this conversation.  The title is set to the first user message (truncated).
    username = getattr(req, "username", None)
    if username:
        # Set title only for brand-new conversations (single message in history)
        is_first_message = len(messages) == 1
        title = req.query if is_first_message else None
        await upsert_conversation(username, req.session_id, title=title)

    return {"result": result["messages"][-1]}


# ─── Document Upload ──────────────────────────────────────────────────────────

@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description"),
    session_id: str = Header("default", alias="X-Session-Id"),
):
    """
    Upload a PDF or TXT document for RAG processing.

    Documents are stored in a Pinecone namespace matching the session_id,
    ensuring per-session isolation.

    Args:
        file: The file object to upload.
        description: Human-readable description of the document (via header).
        session_id: Chat session UUID for vector namespace scoping (via header).

    Returns:
        Upload status dict.
    """
    status_upload = documents(description, file, session_id)
    return {"status": status_upload}


# ─── Conversation Management ──────────────────────────────────────────────────

@router.get("/rag/conversations")
async def get_conversations(username: str = Query(...)):
    """
    List all conversation metadata for a user, newest-updated first.

    Args:
        username: The authenticated user's username (query param).

    Returns:
        List of conversation dicts: {session_id, title, created_at, updated_at}.
    """
    conversations = await list_conversations(username)
    return {"conversations": conversations}


# Request body for creating / updating a conversation record
class ConversationUpsertRequest(BaseModel):
    username: str
    session_id: str
    title: str | None = None


@router.post("/rag/conversations")
async def create_conversation(req: ConversationUpsertRequest):
    """
    Create or update a conversation metadata record.

    Called by Streamlit when the user starts a New Chat so the session is
    registered before the first query is sent.

    Args:
        req: Body with username, session_id, and optional title.

    Returns:
        Confirmation dict.
    """
    await upsert_conversation(req.username, req.session_id, title=req.title)
    return {"status": "ok", "session_id": req.session_id}


@router.get("/rag/conversations/{session_id}/messages")
async def get_conversation_messages(session_id: str):
    """
    Load the full message history for a conversation.

    Used by Streamlit when the user clicks a past conversation in the sidebar
    to reload its messages into the chat area.

    Args:
        session_id: UUID identifying the conversation.

    Returns:
        List of message dicts: {type, content}.
    """
    # Query the chat_history collection directly for raw docs
    cursor = chat_collection.find(
        {"session_id": session_id},
        {"_id": 0, "type": 1, "content": 1},
    ).sort("timestamp", 1)

    docs = await cursor.to_list(length=1000)
    return {"messages": docs}


@router.delete("/rag/conversations/{session_id}")
async def remove_conversation(session_id: str):
    """
    Delete a single conversation – its metadata, chat messages, and Pinecone vectors.

    Args:
        session_id: UUID of the conversation to remove.

    Returns:
        Confirmation dict.
    """
    # Remove metadata record
    await delete_conversation(session_id)

    # Remove all chat messages belonging to this session
    history = ChatHistory.get_session_history(session_id)
    await history.clear()

    # Remove all vectors in the session's Pinecone namespace
    delete_session_vectors(session_id)

    return {"status": "deleted", "session_id": session_id}


@router.delete("/rag/conversations/all")
async def clear_all_user_conversations(username: str = Query(...)):
    """
    Delete ALL conversations for a user – metadata + chat messages.

    Called when the user clicks 'Clear History' in the Streamlit sidebar.

    Args:
        username: The authenticated user's username (query param).

    Returns:
        Number of conversations deleted.
    """
    # Collect session_ids, then bulk-delete metadata
    session_ids = await delete_all_conversations(username)

    # Delete all chat messages and Pinecone vectors for every session
    for sid in session_ids:
        history = ChatHistory.get_session_history(sid)
        await history.clear()
        delete_session_vectors(sid)

    return {"status": "cleared", "deleted_count": len(session_ids)}
