"""
API client for communicating with backend services.
"""

import logging
import requests

logger = logging.getLogger(__name__)

import os

# Backend service URLs
# RUST_BASE_URL = "http://localhost:8080/api"
# PYTHON_BASE_URL = "http://127.0.0.1:8000"
RUST_BASE_URL = os.environ.get("RUST_BASE_URL", "http://localhost:8080/api")
PYTHON_BASE_URL = os.environ.get("PYTHON_BASE_URL", "http://127.0.0.1:8000")


# ─── Authentication (Rust Service) ────────────────────────────────────────────

def create_user(username: str, password: str, api_token: str) -> bool:
    """Create a new user account via the Rust auth service."""
    headers = {"X-API-TOKEN": api_token, "Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{RUST_BASE_URL}/create_user",
            json={"username": username, "password": password},
            headers=headers,
        )
        return response.status_code == 200
    except requests.RequestException:
        logger.exception("create_user failed")
        return False


def login_user(username: str, password: str, api_token: str) -> dict | None:
    """Authenticate user login via the Rust auth service."""
    headers = {"X-API-TOKEN": api_token, "Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{RUST_BASE_URL}/login",
            json={"username": username, "password": password},
            headers=headers,
        )
        return response.json() if response.status_code == 200 else None
    except requests.RequestException:
        logger.exception("login_user failed")
        return None


def get_api_token() -> str | None:
    """Get a system-level API token from the Rust service."""
    try:
        response = requests.post(f"{RUST_BASE_URL}/init")
        return response.json().get("token") if response.status_code == 200 else None
    except requests.RequestException:
        logger.exception("get_api_token failed")
        return None


# ─── RAG Operations (FastAPI Service) ─────────────────────────────────────────

def query_backend(query: str, session_id: str, username: str) -> str:
    """Send a query to the RAG backend, linking it to the user's history."""
    url = f"{PYTHON_BASE_URL}/rag/query"
    try:
        response = requests.post(
            url,
            json={"query": query, "session_id": session_id, "username": username},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()["result"]["content"]
        return f"Error: {response.status_code} - {response.text}"
    except requests.RequestException as e:
        return f"Network error: {str(e)}"


def document_upload_rag(file, description: str, session_id: str = "default") -> bool:
    """Upload a document to the RAG system, scoped to a session."""
    headers = {"X-Description": description, "X-Session-Id": session_id}
    url = f"{PYTHON_BASE_URL}/rag/documents/upload"
    try:
        if file:
            files = {"file": (file.name, file, file.type)}
            response = requests.post(url, files=files, headers=headers)
            return response.status_code == 200
        return False
    except requests.RequestException:
        logger.exception("document_upload_rag failed")
        return False


# ─── Conversation Management (FastAPI Service) ────────────────────────────────

def get_user_conversations(username: str) -> list:
    """Fetch the list of past conversations for the user."""
    url = f"{PYTHON_BASE_URL}/rag/conversations"
    try:
        response = requests.get(url, params={"username": username})
        return response.json().get("conversations", []) if response.status_code == 200 else []
    except requests.RequestException:
        logger.exception("get_user_conversations failed")
        return []


def get_conversation_messages(session_id: str) -> list:
    """Fetch all messages for a specific conversation session."""
    url = f"{PYTHON_BASE_URL}/rag/conversations/{session_id}/messages"
    try:
        response = requests.get(url)
        return response.json().get("messages", []) if response.status_code == 200 else []
    except requests.RequestException:
        logger.exception("get_conversation_messages failed")
        return []


def delete_conversation(session_id: str) -> bool:
    """Delete a single conversation session."""
    url = f"{PYTHON_BASE_URL}/rag/conversations/{session_id}"
    try:
        response = requests.delete(url)
        return response.status_code == 200
    except requests.RequestException:
        logger.exception("delete_conversation failed")
        return False


def clear_user_history(username: str) -> bool:
    """Delete ALL conversations for a user."""
    url = f"{PYTHON_BASE_URL}/rag/conversations/all"
    try:
        response = requests.delete(url, params={"username": username})
        return response.status_code == 200
    except requests.RequestException:
        logger.exception("clear_user_history failed")
        return False
