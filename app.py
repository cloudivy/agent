import streamlit as st
from groq import Groq

st.set_page_config(page_title="Groq LLM Chat", page_icon="🤖", layout="wide")

st.sidebar.title("🔧 Settings")
api_key = st.sidebar.text_input("Groq API Key", value="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", type="password")
model = st.sidebar.selectbox("Model", 
    ["llama3-8b-8192", "llama3-70b-8192", "llama-3.3-70b-versatile", 
     "mixtral-8x7b-32768", "gemma2-9b-it", "llama3-groq-70b-8192-tool-use-preview"])

# Test connection button
if st.sidebar.button("🧪 Test API", use_container_width=True):
    if api_key:
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model, 
                messages=[{"role": "user", "content": "Confirm: Groq API working?"}],
                max_tokens=20
            )
            st.sidebar.success(f"✅ Connected! Response: {response.choices[0].message.content}")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

# Chat session
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ready for ML research, pipelines, or agent sims chats."}]

st.title("🤖 Fast Groq LLM Chat")
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Ask about context drift, anomaly detection, or anything..."):
    if not api_key.startswith("gsk_"):
        st.error("Enter valid Groq API key (starts with gsk_) in sidebar!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            client = Groq(api_key=api_key)
            stream = client.chat.completions.create(
                model=model,
                messages=st.session_state.messages,
                temperature=0.7,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear chat
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = [{"role": "assistant", "content": "Chat cleared! Start fresh."}]
    st.rerun()
