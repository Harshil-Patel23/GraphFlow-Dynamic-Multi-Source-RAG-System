"""
Conversation store backed by MongoDB.

This module manages *conversation metadata* – a lightweight record per chat
session (session_id, username, title, timestamps).  Full message history is
kept separately in the `chat_history` collection (see chat_history_mongo.py).

Collection schema (MongoDB document):
    {
        "session_id":  str,   # UUID that ties this record to chat_history docs
        "username":    str,   # owner of the conversation
        "title":       str,   # first user message truncated to 60 chars
        "created_at":  datetime,
        "updated_at":  datetime,
    }
"""

from datetime import datetime, timezone

from src.db.mongo_client import db

# Collection that stores per-session conversation metadata
_conversations = db["conversations"]

# Maximum length of the auto-generated title (first user message)
TITLE_MAX_LEN = 60


async def ensure_indexes() -> None:
    """
    Create indexes on first use:
    - unique index on session_id (each session_id maps to exactly one record)
    - index on username (fast lookup of all conversations for a user)
    """
    await _conversations.create_index("session_id", unique=True)
    await _conversations.create_index("username")


async def list_conversations(username: str) -> list[dict]:
    """
    Return all conversation metadata for *username*, newest first.

    Args:
        username: The authenticated user's username.

    Returns:
        List of dicts with keys: session_id, title, created_at, updated_at.
    """
    cursor = _conversations.find(
        {"username": username},
        # Exclude MongoDB's internal _id from the result
        {"_id": 0, "session_id": 1, "title": 1, "created_at": 1, "updated_at": 1},
    ).sort("updated_at", -1)   # newest-updated first (like ChatGPT)

    return await cursor.to_list(length=500)


async def upsert_conversation(
    username: str,
    session_id: str,
    title: str | None = None,
) -> None:
    """
    Insert a new conversation record or update its title/timestamp if it
    already exists.

    Args:
        username:   Owner of the conversation.
        session_id: UUID identifying this chat session.
        title:      Human-readable title (first user message, truncated).
                    If None the title is left unchanged on update.
    """
    now = datetime.now(tz=timezone.utc)

    # Build the $set payload – only include title when it is provided
    set_fields: dict = {"updated_at": now}
    if title is not None:
        # Truncate to TITLE_MAX_LEN and strip whitespace
        set_fields["title"] = title[:TITLE_MAX_LEN].strip()

    # setOnInsert runs only when a new document is created (upsert=True)
    await _conversations.update_one(
        {"session_id": session_id},
        {
            "$set": set_fields,
            "$setOnInsert": {
                "username": username,
                "session_id": session_id,
                "created_at": now,
            },
        },
        upsert=True,
    )


async def delete_conversation(session_id: str) -> None:
    """
    Remove a single conversation's metadata record.
    (Chat messages are deleted separately via MongoDBChatMessageHistory.clear.)

    Args:
        session_id: UUID of the conversation to remove.
    """
    await _conversations.delete_one({"session_id": session_id})


async def delete_all_conversations(username: str) -> list[str]:
    """
    Remove ALL conversation metadata records belonging to *username*.

    Returns:
        List of session_ids that were deleted (used by the caller to also
        wipe chat messages from the chat_history collection).
    """
    # Collect session_ids first so the caller can clean up chat_history too
    cursor = _conversations.find(
        {"username": username},
        {"_id": 0, "session_id": 1},
    )
    docs = await cursor.to_list(length=500)
    session_ids = [d["session_id"] for d in docs]

    # Bulk-delete all metadata records for this user
    await _conversations.delete_many({"username": username})

    return session_ids


async def get_conversation_session_ids(username: str) -> list[str]:
    """
    Return only the list of session_ids for a user (helper for bulk operations).

    Args:
        username: The authenticated user's username.

    Returns:
        List of session_id strings.
    """
    cursor = _conversations.find(
        {"username": username},
        {"_id": 0, "session_id": 1},
    )
    docs = await cursor.to_list(length=500)
    return [d["session_id"] for d in docs]
