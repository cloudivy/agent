import streamlit as st
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "append"]
    next: str

@st.cache_resource
def load_researcher():
    prompt = "You are a Researcher. Research the topic deeply and provide facts."
    return create_react_agent(llm, [], prompt)  # Add tools later

@st.cache_resource
def load_analyst():
    prompt = "You are an Analyst. Analyze the research and extract key insights."
    return create_react_agent(llm, [], prompt)

@st.cache_resource
def load_writer():
    prompt = "You are a Writer. Summarize insights into a clear final report."
    return create_react_agent(llm, [], prompt)

def supervisor_node(state: AgentState):
    msg = state["messages"][-1].content
    if "research" in msg.lower():
        return {"next": "researcher"}
    elif "analyze" in msg.lower():
        return {"next": "analyst"}
    else:
        return {"next": "writer"}

def agent_node(state: AgentState, agent):
    result = agent.invoke(state)
    return {"messages": result["messages"]}

# Chat history persistence
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session_1"

st.title("🚀 Simple Multi-Agent Platform")
st.caption("Supervisor routes to Researcher → Analyst → Writer for end-to-end tasks.")

# Chat display
for msg in st.session_state.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

# User input
if prompt := st.chat_input("Enter task e.g., 'Research AI agents'"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agents collaborating..."):
            # Build graph
            workflow = StateGraph(state_schema=AgentState)
            workflow.add_node("supervisor", supervisor_node)
            workflow.add_node("researcher", lambda s: agent_node(s, load_researcher()))
            workflow.add_node("analyst", lambda s: agent_node(s, load_analyst()))
            workflow.add_node("writer", lambda s: agent_node(s, load_writer()))

            workflow.add_edge(START, "supervisor")
            workflow.add_conditional_edges("supervisor", lambda s: s["next"])
            workflow.add_edge("writer", END)

            app = workflow.compile()

            # Run
            for chunk in app.stream({"messages": [HumanMessage(content=prompt)]}, {"configurable": {"thread_id": st.session_state.thread_id}}):
                if "messages" in chunk:
                    for msg in chunk["messages"]:
                        if isinstance(msg, AIMessage):
                            st.markdown(msg.content)
                            st.session_state.messages.append(msg)
