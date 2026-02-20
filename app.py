import streamlit as st
import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# API Key Check & LLM Setup
@st.cache_resource
def get_llm():
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("❌ **Add `OPENAI_API_KEY` to Streamlit Cloud Secrets!**")
        st.stop()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

llm = get_llm()

# Agent State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "append"]
    next: str

# Agent Functions (Simple prompts - no external tools needed)
@st.cache_resource
def researcher_agent():
    return llm.bind_tools([],
        system_message="You are RESEARCHER. Research deeply and provide detailed facts only.")

@st.cache_resource
def analyst_agent():
    return llm.bind_tools([],
        system_message="You are ANALYST. Analyze information and extract key insights.")

@st.cache_resource
def writer_agent():
    return llm.bind_tools([],
        system_message="You are WRITER. Create clear, concise final reports.")

def supervisor(state):
    """Routes to correct agent based on task"""
    last_msg = state["messages"][-1].content.lower()
    if any(word in last_msg for word in ["research", "find", "data"]):
        return {"next": "researcher"}
    elif any(word in last_msg for word in ["analyze", "insight", "summary"]):
        return {"next": "analyst"}
    else:
        return {"next": "writer"}

def call_agent(state, agent):
    """Call agent and return response"""
    result = agent.invoke(state["messages"])
    return {"messages": [result]}

# Build Graph
@st.cache_resource
def create_graph():
    workflow = StateGraph(state_schema=AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("researcher", lambda state: call_agent(state, researcher_agent()))
    workflow.add_node("analyst", lambda state: call_agent(state, analyst_agent()))
    workflow.add_node("writer", lambda state: call_agent(state, writer_agent()))
    
    # Add edges
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "researcher": "researcher",
            "analyst": "analyst", 
            "writer": "writer"
        }
    )
    workflow.add_edge("writer", END)
    
    return workflow.compile()

# Streamlit UI
st.title("🤖 Multi-Agent Platform")
st.caption("Researcher → Analyst → Writer | Persistent memory | OpenAI-powered")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Chat input
if prompt := st.chat_input("Ask anything... e.g., 'Research AI agents'"):
    
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agents collaborating..."):
            # Run multi-agent workflow
            graph = create_graph()
            result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
            
            # Display final response
            final_msg = result["messages"][-1]
            st.markdown(final_msg.content)
            st.session_state.messages.append(final_msg)

# Sidebar Instructions
with st.sidebar:
    st.markdown("### 📋 Quick Start")
    st.code("""
1. Settings → Secrets → Add:
   OPENAI_API_KEY="sk-proj-..."

2. Try these tasks:
   • "Research context drift"
   • "Analyze AI agent frameworks" 
   • "Write PdM report"
    """, language="text")
    st.caption("Built for Divya Mittal | IOCL Research")
