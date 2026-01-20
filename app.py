import streamlit as st
from groq import Groq

st.set_page_config(page_title="Groq LLM Chat", page_icon="🤖", layout="wide")

st.sidebar.title("🔧 Settings")
api_key = st.sidebar.text_input("Groq API Key", value="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", type="password")
model = st.sidebar.selectbox("Model", 
    ["llama3-8b-8192", "llama3-70b-8192", "llama-3.3-70b-versatile", 
     "mixtral-8x7b-32768", "gemma2-9b-it"])

# Test connection (moved before chat)
if st.sidebar.button("🧪 Test API", use_container_width=True):
    if api_key and api_key.startswith("gsk_"):
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model, 
                messages=[{"role": "user", "content": "Groq API working?"}],
                max_tokens=20
            )
            st.sidebar.success(f"✅ Success: {response.choices[0].message.content}")
        except Exception as e:
            st.sidebar.error(f"❌ Test failed: {str(e)}")
    else:
        st.sidebar.warning("Enter valid key (starts with gsk_)")

# Chat session
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi researcher! Chat about LLMs, pipelines, or agents."}]

st.title("🤖 Groq LLM Chat")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input - FIXED: Client init inside try + key check
if prompt := st.chat_input("Your query..."):
    if not api_key or not api_key.startswith("gsk_"):
        st.error("⚠️ Add valid Groq API key in sidebar first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                client = Groq(api_key=api_key)  # Init here, protected
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
            except Exception as e:
                error_msg = f"Error: {str(e)}. Check key/model/quotas."
                message_placeholder.error(error_msg)
                full_response = error_msg
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = [{"role": "assistant", "content": "Cleared! Ready."}]
    st.rerun()

