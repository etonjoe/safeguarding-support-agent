
import streamlit as st
import os
import tempfile

from safeguarding_logic import (
    load_and_split_pdf,
    create_vector_store,
    initialize_llm,
    create_agent_executor
)

# --- Streamlit App Layout ---
st.set_page_config(page_title="Safeguarding Support Agent", layout="wide")
st.title("🏫 Safeguarding Support Agent (Nottingham Schools)")
st.caption("AI Assistant trained on local safeguarding policies")

# --- API Key Check using st.secrets ---
if 'GOOGLE_API_KEY' not in st.secrets:
    st.error("🚨 Google API Key not found. Please set it in your Streamlit secrets.")
    st.stop()

# Set API key as environment variable for compatibility
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# --- Session State Management ---
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! How can I help you with safeguarding procedures today? Please upload a policy PDF first."
    }]

# --- Sidebar Upload ---
with st.sidebar:
    st.header("📁 Policy Document")
    uploaded_file = st.file_uploader("Upload Safeguarding Policy PDF", type="pdf")

    if uploaded_file is not None:
        if st.button("Load and Process Policy"):
            with st.spinner("Processing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    docs = load_and_split_pdf(tmp_file_path)
                    if docs:
                        st.session_state.vector_store = create_vector_store(docs)

                        if st.session_state.llm is None:
                            st.session_state.llm = initialize_llm()

                        if st.session_state.llm and st.session_state.vector_store:
                            st.session_state.agent_executor = create_agent_executor(
                                st.session_state.llm, st.session_state.vector_store
                            )
                            st.success("Policy processed. Agent is ready.")
                            st.session_state.messages = [{
                                "role": "assistant",
                                "content": "Policy loaded successfully! How can I assist you?"
                            }]
                        else:
                            st.error("Failed to initialize LLM or Vector Store.")
                    else:
                        st.error("Could not extract text from the PDF.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)

    if st.session_state.agent_executor:
        st.success("✅ Agent Ready")
    else:
        st.warning("⚠️ Agent not ready. Please upload and process a policy PDF.")

# --- Chat Interface ---
st.header("💬 Chat with the Safeguarding Assistant")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question about the policy or describe a situation..."):
    if not st.session_state.agent_executor:
        st.error("Please upload and process a policy document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent_executor.invoke({
                    "input": prompt,
                    "chat_history": []  # Extend this later if needed
                })
                answer = response['output']
            except Exception as e:
                answer = "Sorry, I encountered an error. Please try again."
                st.error(str(e))

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
