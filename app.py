import streamlit as st
from src.search import RAGSearch

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------
# Custom CSS Styling
# ---------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        color: #4CAF50;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }
    .chat-container {
        border-radius: 12px;
        padding: 20px;
        background-color: #f9f9f9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Header
# ---------------------------
st.markdown('<div class="main-title">📚 RAG AI Document Assistant</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Ask questions about your documents using Retrieval Augmented Generation</div>',
    unsafe_allow_html=True
)

# ---------------------------
# Load RAG System
# ---------------------------
@st.cache_resource
def load_rag():
    return RAGSearch()

rag = load_rag()

# ---------------------------
# Chat History
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:

    st.title("⚙️ System Info")

    st.markdown("**Embedding Model**")
    st.code("all-MiniLM-L6-v2")

    st.markdown("**LLM Model**")
    st.code("llama-3.1-8b-instant")

    st.markdown("**Vector Database**")
    st.code("FAISS")

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown(
        """
        **RAG Pipeline**

        User Query  
        ↓  
        Embedding  
        ↓  
        Vector Search (FAISS)  
        ↓  
        Retrieve Context  
        ↓  
        Groq LLM  
        ↓  
        Generated Answer
        """
    )

# ---------------------------
# Chat Container
# ---------------------------
chat_container = st.container()

with chat_container:

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ---------------------------
# User Input
# ---------------------------
query = st.chat_input("Ask something about your documents...")

if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            answer = rag.search_and_answer(query)

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )