"""
Chat page for the Streamlit application with conversation history sidebar.
"""

import streamlit as st
import uuid
import sys
import os

# Add the streamlit_app directory to sys.path so utils can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.api_client import (
    query_backend,
    document_upload_rag,
    get_user_conversations,
    get_conversation_messages,
    clear_user_history,
    delete_conversation
)

# ─── Configuration ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LangGraph Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a clean, light "ChatGPT-like" sidebar
st.markdown("""
    <style>
        /* General button styling */
        .stButton > button { width: 100%; border-radius: 8px; font-weight: 500; }
        
        /* Sidebar container styling - Light Gray */
        [data-testid="stSidebar"] { 
            background-color: #F8F9FB; 
            border-right: 1px solid #E0E0E0;
        }

        /* Chat history items in sidebar */
        .sidebar-chat-item {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            cursor: pointer;
            border: 1px solid #E0E0E0;
            background: white;
            color: #1F1F1F;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }
        .sidebar-chat-item:hover { 
            background: #F0F2F6; 
            border-color: #D0D0D0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ─────────────────────────────────────────────

if "jwt_token" not in st.session_state:
    st.warning("Please login first.")
    st.switch_page("home.py")
    st.stop()

username = st.session_state.get("username", "user")

# current_session_id tracks which conversation is active
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

# chat_history stores the messages for the active session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# state to refresh sidebar list
if "refresh_sidebar" not in st.session_state:
    st.session_state.refresh_sidebar = True


# ─── Sidebar Functions ────────────────────────────────────────────────────────

def start_new_chat():
    """Reset the session state for a fresh chat session."""
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.refresh_sidebar = True


def load_chat(session_id: str):
    """Load messages from a past conversation session."""
    messages = get_conversation_messages(session_id)
    # Convert from dict {"type": "human", "content": "..."} to tuple ("user", "...")
    formatted_history = []
    for m in messages:
        role = "user" if m["type"] in ["human", "user"] else "assistant"
        formatted_history.append((role, m["content"]))
    
    st.session_state.chat_history = formatted_history
    st.session_state.current_session_id = session_id


# ─── Sidebar UI ───────────────────────────────────────────────────────────────

with st.sidebar:
    # st.title("🤖 Assistant")
    
    # 1. Document Upload (Moved to top per request)
    st.subheader("📂 Upload Context")
    uploaded_file = st.file_uploader("Add a PDF/TXT", type=["pdf", "txt"])
    if uploaded_file:
        desc = st.text_input("Describe the file", placeholder="e.g. Project specs")
        if st.button("Upload"):
            if desc:
                with st.spinner("Processing document..."):
                    if document_upload_rag(uploaded_file, desc, st.session_state.current_session_id):
                        st.success("Context added!")
                    else:
                        st.error("Upload failed.")
            else:
                st.warning("Description required.")

    st.divider()
    
    # 2. History List
    st.subheader("📜 Recent Chats")
    conversations = get_user_conversations(username)
    
    if not conversations:
        st.info("No past conversations yet.")
    else:
        for conv in conversations:
            sid = conv["session_id"]
            title = conv.get("title") or "Untitled Chat"
            
            # Highlight current active chat
            is_active = (sid == st.session_state.current_session_id)
            btn_label = f"{'⭐ ' if is_active else ''}{title}"
            
            if st.button(btn_label, key=f"btn_{sid}", use_container_width=True):
                load_chat(sid)
                st.rerun()

    st.divider()

    # 3. Settings/Clean-up (Bottom)
    col_clear, col_logout = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear All"):
            if clear_user_history(username):
                start_new_chat()
                st.rerun()
    with col_logout:
        if st.button("🔒 Logout"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.switch_page("home.py")


# ─── Main Chat Area ───────────────────────────────────────────────────────────

# Layout Title and "New Chat" button in a single row
header_col1, header_col2 = st.columns([8, 2])
with header_col1:
    st.title("💬 Chat Session")
with header_col2:
    st.write(" ") # Vertical alignment spacer
    if st.button("➕ New Chat", type="primary"):
        start_new_chat()
        st.rerun()

st.caption(f"Logged in as: **{username}** | Session: `{st.session_state.current_session_id[:8]}`")


# Display historical messages
for role, text in st.session_state.chat_history:
    st.chat_message(role).write(text)

# Input area
user_input = st.chat_input("Ask me anything...")

if user_input:
    # 1. Immediately display user message
    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").write(user_input)

    # 2. Get response from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_backend(
                user_input, 
                st.session_state.current_session_id,
                username
            )
            st.write(response)

    # 3. Save assistant response
    st.session_state.chat_history.append(("assistant", response))
    
    # 4. Refresh sidebar (so new title appears if it was the first message)
    if len(st.session_state.chat_history) <= 2:
        st.session_state.refresh_sidebar = True
        st.rerun()
