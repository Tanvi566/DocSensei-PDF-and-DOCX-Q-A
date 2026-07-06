import os
import tempfile
import time

import streamlit as st

from utils.loader import load_document
from utils.splitter import split_documents
from utils.vectorstore import create_vectorstore
from utils.qa import ask_question

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="DocSensei",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.title("⚙️ DocSensei Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or DOCX",
    type=["pdf", "docx"]
)

strategy = st.sidebar.selectbox(
    "Chunking Strategy",
    [
        "Strategy A (500 / 50)",
        "Strategy B (1000 / 100)"
    ]
)

top_k = st.sidebar.slider(
    "Retrieved Chunks",
    min_value=1,
    max_value=10,
    value=3
)

if st.sidebar.button("🗑 Clear Session"):
    st.session_state.clear()
    st.rerun()

# ---------------------------------------------------
# Main Page
# ---------------------------------------------------

st.title("📄 DocSensei")
st.write(
    "Upload a PDF or DOCX, then ask questions grounded only in that document."
)

# ---------------------------------------------------
# Upload Handling
# ---------------------------------------------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    needs_indexing = (
        "vectorstore" not in st.session_state
        or st.session_state.get("current_file") != uploaded_file.name
        or st.session_state.get("strategy") != strategy
    )

    if needs_indexing:

        start_time = time.time()

        with st.spinner("📚 Reading document and creating vector database..."):

            docs = load_document(file_path)

            if strategy == "Strategy A (500 / 50)":

                chunks = split_documents(
                    docs,
                    chunk_size=500,
                    chunk_overlap=50
                )

            else:

                chunks = split_documents(
                    docs,
                    chunk_size=1000,
                    chunk_overlap=100
                )

            vectorstore = create_vectorstore(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.current_file = uploaded_file.name
            st.session_state.strategy = strategy
            st.session_state.pages = len(docs)
            st.session_state.chunks = len(chunks)
            st.session_state.index_time = round(
                time.time() - start_time,
                2
            )

        st.success("✅ Document indexed successfully!")

    # ---------------------------------------------------
    # Sidebar Statistics
    # ---------------------------------------------------

    st.sidebar.markdown("---")

    st.sidebar.subheader("📊 Document Statistics")

    st.sidebar.metric(
        "Pages",
        st.session_state.pages
    )

    st.sidebar.metric(
        "Chunks",
        st.session_state.chunks
    )

    st.sidebar.metric(
        "Index Time",
        f"{st.session_state.index_time}s"
    )

    st.sidebar.info(
        f"Current Strategy:\n\n{strategy}"
    )

    # ---------------------------------------------------
    # Question Answering
    # ---------------------------------------------------

    question = st.text_input(
        "💬 Ask a question about your document"
    )

    if question:

        with st.spinner("🤖 Thinking..."):

            answer, sources = ask_question(
                st.session_state.vectorstore,
                question,
                top_k
            )

        st.markdown("## 🤖 Answer")

        st.info(answer)

        st.markdown("## 📚 Sources")

        for index, source in enumerate(sources, start=1):

            page = source.metadata.get("page", "Unknown")

            if page != "Unknown":
                page += 1

            with st.expander(
                f"Source {index} | Page {page}"
            ):

                preview = source.page_content

                if len(preview) > 500:
                    preview = preview[:500] + "..."

                st.write(preview)

else:

    st.info(
        "👈 Upload a PDF or DOCX from the sidebar to begin."
    )