import streamlit as st
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.visual import draw_ascii  # Simple ASCII graph

# Local free LLM (run `ollama pull llama3.2` first)
@st.cache_resource
def get_llm():
    return ChatOllama(model="llama3.2", temperature=0)

llm = get_llm()

# PdM Tool
@tool
def check_sensor_history(sensor_id: str) -> str:
    """Check pipeline sensor history for predictive maintenance."""
    history = {
        "42": "Past: 3 high vib alerts, SCC risk rising.",
        "default": "Normal trends."
    }
    return history.get(sensor_id, "No history.")

llm_with_tools = llm.bind_tools([check_sensor_history])

# Agent State (memory + comms)
class PdMState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], operator.add]
    memory: List[str]
    next_agent: str

# Supervisor: Tool calling + routing
def supervisor(state: PdMState) -> PdMState:
    last_msg = state["messages"][-1]
    response = llm_with_tools.invoke(state["messages"])
    state["messages"] += [response]
    content = response.content.lower()
    state["next_agent"] = "expert" if "critical" in content or "high" in content else END
    state["memory"].append(f"Routed {last_msg.content[:50]}...")
    return state

# Expert: Final action
def expert_agent(state: PdMState) -> PdMState:
    expert_msg = AIMessage(content="Expert: Schedule drone inspection & notify ops team.")
    state["messages"] += [expert_msg]
    state["memory"].append("Action: Inspection dispatched")
    state["next_agent"] = END
    return state

# Build & compile graph
workflow = StateGraph(PdMState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("expert", expert_agent)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor", 
    lambda s: s["next_agent"],
    {"expert": "expert", END: END}
)
workflow.add_edge("expert", END)
pdm_app = workflow.compile()

# Streamlit UI
st.title("🛢️ Pipeline PdM Agent Visualizer")
st.markdown("**Free local Ollama + LangGraph + Tools/Memory/Comms**")

# ASCII Graph Viz
with st.expander("Agent Workflow Graph"):
    st.code(draw_ascii(pdm_app), language="text")

# Input
alert = st.text_input("Enter Pipeline Alert (e.g., 'Sensor 42 high vibration'):")
sensor = st.text_input("Sensor ID:", "42")

if st.button("🚀 Run Agent Simulation", type="primary"):
    initial_state = {
        "messages": [HumanMessage(content=f"{alert} [Sensor: {sensor}]")],
        "memory": st.session_state.get("memory", []),
        "next_agent": ""
    }
    with st.spinner("Agent thinking..."):
        result = pdm_app.invoke(initial_state)
    
    # Results
    st.subheader("📱 Agent Messages")
    for msg in result["messages"]:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.write(msg.content)
    
    st.subheader("🧠 Agent Memory")
    st.write(result["memory"])
    
    # Persist memory
    if "memory" not in st.session_state:
        st.session_state.memory = []
    st.session_state.memory.extend(result["memory"])

# Ollama Status
if st.button("Check Ollama"):
    try:
        test = llm.invoke([HumanMessage(content="Status?")])
        st.success("✅ Ollama running!")
    except:
        st.error("❌ Install Ollama & run `ollama pull llama3.2`")

st.info("💡 Deploy: GitHub → Streamlit Cloud (free)")

