"""
Graph builder module for the adaptive RAG system.
"""

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph


from src.rag.retriever_setup import get_retriever
from src.config.settings import Config
from src.llms.openai import llm
from src.models.grade import Grade
from src.models.route_identifier import RouteIdentifier
from src.models.state import State
from src.tools.graph_tools import routing_tool, doc_tool

config = Config()


def contextualize_query(state: State):
    """
    Rewrite ambiguous follow-up questions into standalone queries
    using conversation history so downstream nodes receive clear context.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated state with latest_query set to the standalone question.
    """
    messages = state["messages"]
    current_question = messages[-1].content

    # If there's no prior history, the question is already standalone
    if len(messages) <= 1:
        print("="*140)
        print("[contextualize] No prior history — using question as-is")
        print("="*140)
        return {"latest_query": current_question, "chat_history": "No prior history."}

    # Build a readable chat history string from prior messages
    # Limit to last 20 messages to avoid token overflow on long conversations
    chat_history_lines = []
    for msg in messages[:-1]:  # everything except the latest question
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        chat_history_lines.append(f"{role}: {msg.content}")
    chat_history_str = "\n".join(chat_history_lines[-20:])

    contextualize_prompt = PromptTemplate(
        template=config.prompt("contextualize_prompt"),
        input_variables=["chat_history", "question"]
    )
    chain = contextualize_prompt | llm
    result = chain.invoke({
        "chat_history": chat_history_str,
        "question": current_question
    })

    standalone_question = result.content.strip()
    print("="*140)
    print(f"[contextualize] '{current_question}' → '{standalone_question}'")
    print("="*140)

    # Return both the standalone question AND the clean chat history so
    # downstream nodes (query_classifier, generate) can use it without
    # being polluted by intermediate retriever messages.
    return {"latest_query": standalone_question, "chat_history": chat_history_str}


# # Node implementations
def query_classifier(state: State):
    """
    Classify the query to determine if it's related to indexed documents.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated state with route and latest_query.
    """
    # Use the contextualized standalone question (set by contextualize_query)
    question = state.get("latest_query") or state["messages"][-1].content

    # Pass chat history so the classifier can detect conversational questions
    # (e.g. "what is my name?" when the user already stated it in chat)
    chat_history = state.get("chat_history", "No prior history.")

    llm_with_structured_output = llm.with_structured_output(RouteIdentifier)
    classify_prompt = PromptTemplate(
        template=config.prompt("classify_prompt"),
        input_variables=["question", "chat_history"]
    )

    chain = classify_prompt | llm_with_structured_output
    result = chain.invoke({"question": question, "chat_history": chat_history})
    valid_routes = ["index", "general", "search"]
    route = result.route if result.route in valid_routes else "index"
    print("="*140)
    print(f"[query_classifier] Route: {route}")
    print("="*140)
    return {"messages": state["messages"], "route": route, "latest_query": question}
    

def general_llm(state: State):
    """
    Fetch general common knowledge result from the LLM.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages from LLM.
    """
    result = llm.invoke(state["messages"])
    print("="*140)
    print("inside general llm")
    print("="*140)
    print(result)
    print("="*140)
    return {"messages": result}


def retriever_node(state: State):
    """
    Retrieve results directly from the vector store.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages with retrieved document content.
    """
    query = state["latest_query"]
    session_id = state.get("session_id", "default")
    retriever_tool = get_retriever(session_id)
    
    try:
        # Search the vector store directly without an agent loop limit
        raw_docs = retriever_tool.invoke(query)

        # Extract clean page_content text from each Document chunk.
        # Using str(raw_docs) produces unreadable Python object dumps
        # (e.g. "Document(page_content='...', metadata={...})") which
        # confuses the LLM and causes it to hallucinate missing data.
        if isinstance(raw_docs, list):
            context = "\n\n---\n\n".join(
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in raw_docs
            )
        else:
            # Fallback: already a plain string (e.g. error path)
            context = str(raw_docs)

    except Exception as e:
        print(f"Retrieval error: {e}")
        context = "No relevant context found."

    new_message = AIMessage(
        content=context
    )

    print("="*140)
    print("Result from retriever_node (Document Context):")
    print("="*140)
    print(new_message.content)
    print("="*140)

    return {
        "messages": [new_message]
    }


def grade(state: State):
    """
    Grade the results retrieved from vector stores.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated state with binary_score.
    """
    grading_prompt = PromptTemplate(
        template=config.prompt("grading_prompt"),
        input_variables=["question", "context"]
    )
    context = state["messages"][-1].content
    question = state["latest_query"]

    llm_with_grade = llm.with_structured_output(Grade)

    chain_graded = grading_prompt | llm_with_grade
    result = chain_graded.invoke({"question": question, "context": context})

    print("="*140)
    print("Grading result: ",result)
    print("="*140)

    return {"messages": state["messages"], "binary_score": result.binary_score}


def rewrite_query(state: State):
    """
    Rewrite the query to get better retrieval results.

    Args:
        state (State): State of the question.

    Returns:
        dict: Updated latest_query.
    """
    query = state["latest_query"]
    rewrite_prompt = PromptTemplate(
        template=config.prompt("rewrite_prompt"),
        input_variables=["query"]
    )
    chain = rewrite_prompt | llm
    result = chain.invoke({"query": query})

    print("="*140)
    print("Rewritten query: ", result)
    print("="*140)

    return {
        "latest_query": result.content
    }


def generate(state: State):
    """
    Generate the final answer for the user.

    Args:
        state (State): State of the question.

    Returns:
        dict: Generated response.
    """
    question = state["latest_query"]
    context = state["messages"][-1].content
    chat_history = state.get("chat_history", "No prior history.")

    generate_prompt = PromptTemplate(
        template=config.prompt("generate_prompt"),
        input_variables=["question", "context", "chat_history"]
    )

    generate_chain = generate_prompt | llm

    result = generate_chain.invoke({
        "question": question,
        "context": context,
        "chat_history": chat_history
    })

    print("="*140)
    print("Generated answer: ")
    print("="*140)
    print(result)
    print("="*140)

    return {"messages": [{"role": "assistant", "content": result.content}]}


def web_search(state: State):
    """
    Search the web for the rewritten query.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Search results as messages.
    """
    # Initialize the Tavily tool
    search_tool = TavilySearchResults()

    # Search a query
    result = search_tool.invoke(state["latest_query"])

    print("="*140)
    print("Raw web search results: ")
    print("="*140)
    print(result)
    print("="*140)

    contents = [item["content"] for item in result if "content" in item]
    print("="*140)
    print(contents)
    print("="*140)
    return {
        "messages": [{"role": "assistant", "content": "\n\n".join(contents)}]
    }


# Build the graph
graph = StateGraph(State)

graph.add_node("contextualize", contextualize_query)
graph.add_node("query_analysis", query_classifier)
graph.add_node("retriever", retriever_node)
graph.add_node("grade", grade)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("general_llm", general_llm)

graph.add_edge(START, "contextualize")
graph.add_edge("contextualize", "query_analysis")
graph.add_edge("web_search", "generate")
graph.add_edge("retriever", "grade")
graph.add_edge("rewrite", "retriever")
graph.add_conditional_edges("query_analysis", routing_tool)
graph.add_conditional_edges("grade", doc_tool)
graph.add_edge("generate", END)
graph.add_edge("general_llm", END)

builder = graph.compile()