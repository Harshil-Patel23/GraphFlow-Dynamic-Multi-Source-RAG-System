# System Architecture & Design

This document provides a detailed overview of the GraphFlow system architecture, data flow, and component relationships.

## 1. High-Level Architecture

GraphFlow leverages a modern microservice-inspired architecture separating the frontend, backend, and an extremely fast authentication service. 

```mermaid
graph TD
    User((User)) --> Streamlit[Streamlit Frontend]
    Streamlit --> FastAPI[FastAPI Backend]
    Streamlit --> RustAuth[Rust Auth Service]
    
    subgraph "Data Storage"
        FastAPI --> MongoDB[(MongoDB: Chat History)]
        FastAPI --> Pinecone[(Pinecone / Qdrant : Vector DB)]
        RustAuth --> MongoDB
    end

    subgraph "Agentic Pipeline (LangGraph)"
        FastAPI --> Routing[Query Router]
        Routing --> RetrievalAgent[Retrieval Agent]
        Routing --> GeneralAgent[General Agent]
        Routing --> WebSearch[Web Search - Tavily]
        RetrievalAgent --> VectorDB[(Vector DB)]
        RetrievalAgent --> Grading[Relevance Grading]
        Grading --> Gen[Generate Response]
        GeneralAgent --> Gen
        WebSearch --> Gen
    end
```

### Key Components

1. **Frontend (Streamlit):** Provides a rapid, customizable interface for users to chat and upload context documents.
2. **Core Backend (FastAPI):** Handles business logic, document chunking, and orchestration of the AI pipeline.
3. **Authentication (Rust):** Written in Rust for maximum performance and security. Handles JWT issuance and validation.
4. **Agent Orchestrator (LangGraph):** The brain of the system, determining if a query needs vector retrieval, general LLM knowledge, or live web search.
5. **Persistence Models:**
   - **MongoDB:** Stores users, persistent sessions, and message chat history.
   - **Pinecone/Qdrant:** Vector databases storing high-dimensional embeddings of user documents.

---

## 2. RAG Data Flow (Ingestion & Retrieval)

When a user uploads a document, it follows a strict pipeline to ensure it is accurately embedded and stored.

```mermaid
graph LR
    A[Upload Document] --> B[Text Splitter / Chunker]
    B --> C[Embedding Model]
    C --> D[(Vector DB)]
    E[User Query] --> F[Embedding Model]
    F --> G[Similarity Search]
    D --> G
    G --> H[Top-K Snippets]
    H --> I[LLM Generation]
    E --> I
    I --> J[Final Answer]
```

---

## 3. Request Lifecycle (Sequence Diagram)

This outlines exactly what happens when a user sends a chat message. The system is designed to evaluate, verify, and automatically correct its retrieval strategy.

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Streamlit UI
    participant Backend as FastAPI
    participant Auth as Rust Auth
    participant Core as LangGraph Core
    participant DB as MongoDB/Pinecone
    participant LLM as Groq/Tavily

    User->>Frontend: Send Message
    Frontend->>Auth: Validate JWT Token
    Auth-->>Frontend: Token Valid
    Frontend->>Backend: POST /chat (message, session_id)
    Backend->>Core: Invoke Graph State
    Core->>DB: Fetch Chat History (MongoDB)
    Core->>LLM: Classify Intent (Router Node)
    
    alt is "Retrieval"
        Core->>DB: Query Vector DB
        DB-->>Core: Top-K Contexts
        Core->>LLM: Grade Context Relevance
    else is "Web Search"
        Core->>LLM: Tavily Search API
    end
    
    Core->>LLM: Generate Answer with Context
    LLM-->>Core: Final Answer Text
    Core->>DB: Save to Chat History
    Core-->>Backend: Final State Output
    Backend-->>Frontend: JSON Response
    Frontend-->>User: Display Message
```

---

## 4. Entity-Relationship Diagram

The core database uses MongoDB for flexibility, but follows a strict relational pattern mentally to align user sessions and privacy.

```mermaid
erDiagram
    USERS {
        string _id PK
        string email
        string hashed_password
        date created_at
    }
    SESSIONS {
        string session_id PK
        string user_id FK
        date last_active
    }
    CHATS {
        string message_id PK
        string session_id FK
        string role "user/assistant"
        string content
        date timestamp
    }
    DOCUMENTS {
        string doc_id PK
        string user_id FK
        string filename
        int chunk_count
    }

    USERS ||--o{ SESSIONS : "has"
    SESSIONS ||--o{ CHATS : "contains"
    USERS ||--o{ DOCUMENTS : "uploads"
```

## Security & Isolation Strategy

- **Authentication:** Rust handles token signing to prevent unauthorized access.
- **Data Isolation:** The Vector DB leverages **namespace** isolation (typically utilizing `session_id`) to guarantee that context from one user's chat is physically incapable of leaking into another's.
