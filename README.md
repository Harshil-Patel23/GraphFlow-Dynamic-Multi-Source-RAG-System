# GraphFlow - Agentic AI Chatbot

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.5.4-orange.svg)](https://python.langchain.com/langgraph/)
[![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-blue.svg)](https://www.pinecone.io/)
[![Rust](https://img.shields.io/badge/Rust-Auth--Backend-orange.svg)](https://www.rust-lang.org/)

## 📋 Overview

**GraphFlow** is a sophisticated, end-to-end Retrieval-Augmented Generation (RAG) system built with an **Agentic AI architecture**. It dynamically routes queries, retrieves context-aware information from a **Pinecone** vector database, and generates high-quality responses using advanced LLM orchestration via **LangGraph**.

The system is designed for scalability and production readiness, featuring a robust **Rust-based authentication service**, a **MongoDB** persistent chat history, and a modern **Streamlit** user interface.

---

## 🎯 Key Features

### 🧠 Intelligent Query Routing

- **Adaptive Classification**: Real-time classification of queries into three types:
  - **Index**: Specific answers retrieved from your uploaded documents.
  - **General**: Broad knowledge answers from the LLM's internal weights.
  - **Web Search**: Real-time information fetched via **Tavily**.

### 📚 Advanced RAG Pipeline (Pinecone Powered)

- **High-Performance Retrieval**: Uses **Pinecone** for extremely fast similarity searches.
- **Local Embeddings**: Utilizes `all-MiniLM-L6-v2` for efficient, high-quality document vectorization.
- **Relevance Grading**: Automated nodes to evaluate retrieved context before generation.
- **Query Rewriting**: Self-correcting loop to optimize queries for better retrieval.

### 🤖 Agentic Orchestration

- **LangGraph Workflow**: A stateful multi-agent system managing the complex logic of query analysis, retrieval, and verification.
- **ReAct Framework**: Implements Reasoning and Acting patterns for autonomous decision-making.

### 💾 Persistence & Security

- **Rust Auth Service**: High-performance, secure authentication backend.
- **MongoDB History**: Full conversation persistence allowing users to resume chats across sessions.
- **Session Tracking**: Individual context isolation per user and session.

---

## 🏗️ Architecture

For an in-depth look at our RAG Data pipeline, Request Lifecycle, and Database ERD diagrams, please view our [**Architecture Documentation (ARCHITECTURE.md)**](ARCHITECTURE.md).

---

## 🛠️ Technology Stack

| Component             | Technology                       | Role                         |
| --------------------- | -------------------------------- | ---------------------------- |
| **LLM Orchestration** | LangChain / LangGraph            | Workflow & Agentic Logic     |
| **Vector Database**   | Pinecone                         | Document Storage & Retrieval |
| **Authentication**    | Rust                             | Secure User Management       |
| **Persistence**       | MongoDB                          | Chat History & Session Data  |
| **Backend API**       | FastAPI                          | Core Service Layer           |
| **Frontend UI**       | Streamlit                        | Iterative User Interface     |
| **Embeddings**        | HuggingFace (`all-MiniLM-L6-v2`) | Document Vectorization       |
| **Web Search**        | Tavily                           | Real-time Context Retrieval  |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- Rust & Cargo (for auth service)
- MongoDB (Local or Atlas)
- Pinecone Account (Free Tier works)
- Groq / Tavily API Keys

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/Harshil-Patel23/GraphFlow.git
cd GraphFlow

# Setup Python Backend
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Setup Rust Auth (in separate terminal)
cd rust_auth
cargo build --release
```

### 3. Environment Setup

Create a `.env` file in the root directory:

```env
# AI Services
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=adaptive-rag-index

# Database
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=adaptive_rag

# Security
SECRET_KEY=your_super_secret_key
```

### 4. Running the App

**Terminal 1: FastAPI Backend**

```bash
python -m uvicorn src.main:app --reload --port 8000
```

**Terminal 2: Streamlit Frontend**

```bash
streamlit run streamlit_app/home.py
```

**Terminal 3: Rust Auth (Optional/Dev)**

```bash
cd rust_auth
cargo run
```

---

## 📂 Project Structure

```
GraphFlow/
├── src/                    # Core RAG Logic
│   ├── api/                # FastAPI Routes
│   ├── core/               # App Configuration & Logging
│   ├── llms/               # LLM Provider Wrappers
│   ├── memory/             # MongoDB History Logic
│   ├── rag/                # LangGraph & Vector DB Setup
│   └── tools/              # Graph Decision Tools
├── streamlit_app/          # Frontend Pages
│   ├── home.py             # Login / Landing
│   └── pages/chat.py       # Main Chat & File Upload
├── rust_auth/              # Rust Authentication Backend
├── requirements.txt        # Python Dependencies
└── .env                    # Environment Variables
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author**: [Harshil Patel](https://github.com/Harshil-Patel23)  
**Status**: Active & Production Ready ✅
