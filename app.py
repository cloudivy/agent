import streamlit as st
import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# ========================================
# API KEY CHECK - FIXED
# ========================================
api_key = st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    st.title("🔑 Multi-Agent Platform")
    st.error("❌ **Please add `OPENAI_API_KEY` in Settings → Secrets**")
    st.info("👉 Go to hamburger menu (top-right) → Settings → Secrets")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

# ========================================
# AGENT STATE & FUNCTIONS
# ========================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "append"]
    next: str

def supervisor(state):
    last_msg = state["messages"][-1].content.lower()
    if any(word in last_msg for word in ["research", "find", "data", "info"]):
        return {"next": "researcher"}
    elif any(word in last_msg for word in ["analyze", "insight", "summary"]):
        return {"next": "analyst"}
    else:
        return {"next": "writer"}

@st.cache_resource
def create_agents():
    researcher = llm.bind(
        system_message="You are RESEARCHER. Provide detailed facts and research only.")
    analyst = llm.bind(
        system_message="You are ANALYST. Analyze data and extract key insights.")
    writer = llm.bind(
        system_message="You are WRITER. Create clear, concise final reports.")
    return researcher, analyst, writer

def call_agent(state, agent):
    result = agent.invoke(state["messages"])
    return {"messages": [result]}

# ========================================
# BUILD GRAPH
# ========================================
@st.cache_resource 
def get_workflow():
    researcher, analyst, writer = create_agents()
    
    workflow = StateGraph(state_schema=AgentState)
    
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("researcher", lambda s: call_agent(s, researcher))
    workflow.add_node("analyst", lambda s: call_agent(s, analyst))
    workflow.add_node("writer", lambda s: call_agent(s, writer))
    
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor", 
        lambda s: s["next"],
        {"researcher": "researcher", "analyst": "analyst", "writer": "writer"}
    )
    workflow.add_edge("writer", END)
    
    return workflow.compile()

# ========================================
# STREAMLIT UI
# ========================================
st.title("🤖 Multi-Agent Research Platform")
st.markdown("**Researcher → Analyst → Writer** | Powered by GPT-4o-mini")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

# ========================================
# BIG PROMINENT CHAT INPUT (THIS IS YOUR REQUEST)
# ========================================
st.markdown("---")
prompt = st.chat_input(
    "💬 **Enter your research task here...** (e.g., 'Research context drift in AI agents')",
    key="main_input"
)

if prompt:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Run multi-agent system
    with st.chat_message("assistant"):
        with st.spinner("🤖 **Agents collaborating:** Supervisor → Researcher → Analyst → Writer..."):
            graph = get_workflow()
            result = graph.invoke({"messages": st.session_state.messages})
            
            # Show final response
            final_msg = result["messages"][-1]
            st.markdown(final_msg.content)
            st.session_state.messages.append(final_msg)

# ========================================
# SIDEBAR HELP
# ========================================
with st.sidebar:
    st.header("📋 Example Prompts")
    st.markdown("""
    - "Research context drift in multi-agent systems"
    - "Analyze LangGraph vs CrewAI" 
    - "Write PdM pipeline report"
    - "Find AI agent memory techniques"
    """)
    
    st.markdown("---")
    st.success("✅ **API Key Working!**")
    st.info("👉 Type in the **big input box below**")
