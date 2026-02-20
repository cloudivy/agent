import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import os

# ========================================
# API KEY CHECK
# ========================================
st.title("🤖 Multi-Agent Platform")
st.markdown("**Supervisor → Researcher → Analyst → Writer**")

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("❌ **Add `OPENAI_API_KEY` in Settings → Secrets**")
    st.markdown("---")
    st.markdown("""
    **Steps:**
    1. Click hamburger menu (☰) top-right
    2. Settings → Secrets 
    3. Add: `OPENAI_API_KEY="sk-proj-your-key"`
    """)
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

# ========================================
# CHAT HISTORY
# ========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# ========================================
# BIG CLEAR CHAT INPUT (YOUR REQUEST)
# ========================================
st.markdown("---")
st.markdown("### 💬 **Enter your task below**")

prompt = st.chat_input(
    "Type here... e.g., 'Research context drift', 'Analyze AI frameworks', 'Write PdM report'",
    key="chat_input"
)

# ========================================
# PROCESS INPUT - SIMPLIFIED MULTI-AGENT
# ========================================
if prompt:
    # Add user message to chat
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agents working..."):
            
            # SUPERVISOR: Route task
            supervisor_prompt = f"""
            You are SUPERVISOR. Route this task: "{prompt}"
            
            Choose ONE agent:
            - RESEARCHER (if research/data needed)  
            - ANALYST (if analysis/insights needed)
            - WRITER (if report/summary needed)
            
            Respond with ONLY: "RESEARCHER" or "ANALYST" or "WRITER"
            """
            supervisor_role = llm.invoke(supervisor_prompt).content.strip()
            
            st.markdown(f"**📋 Supervisor routed to:** *{supervisor_role}*")
            
            # AGENT WORKFLOW
            if "RESEARCH" in supervisor_role.upper():
                research_prompt = f"""You are RESEARCHER. Research "{prompt}" and provide detailed facts only."""
                response = llm.invoke(research_prompt)
                
            elif "ANALYS" in supervisor_role.upper():
                analysis_prompt = f"""You are ANALYST. Analyze "{prompt}" and extract key insights."""
                response = llm.invoke(analysis_prompt)
                
            else:  # WRITER
                writer_prompt = f"""You are WRITER. Create clear final report for "{prompt}"."""
                response = llm.invoke(writer_prompt)
            
            # Show result
            st.markdown(response.content)
            st.session_state.messages.append(AIMessage(content=response.content))

# ========================================
# SIDEBAR - EXAMPLE PROMPTS
# ========================================
with st.sidebar:
    st.header("🎯 Example Tasks")
    examples = [
        "Research context drift in AI agents",
        "Analyze LangGraph vs CrewAI", 
        "Write pipeline PdM report",
        "Research agent memory techniques"
    ]
    
    for example in examples:
        if st.button(example, key=example):
            st.session_state.messages = [HumanMessage(content=example)]
            st.rerun()
    
    st.markdown("---")
    st.success("✅ **Your API key is working!**")
