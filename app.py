import os
import tempfile
import time

import streamlit as st

from utils.loader import load_document
from utils.splitter import split_documents
from utils.vectorstore import create_vectorstore
from utils.qa import ask_question

# =====================================================================
# Page configuration
# =====================================================================

st.set_page_config(
    page_title="DocSensei",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# Theme — restrained, single-accent dark UI
# =====================================================================

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --bg:            #121319;
  --bg-elevated:   #191a22;
  --surface:       #1e2029;
  --surface-hover: #252732;
  --border:        #2a2c37;
  --border-strong: #383b49;
  --fg:            #e8e9ee;
  --muted:         #9195a3;
  --muted-2:       #5d606e;
  --accent:        #6366f1;
  --accent-soft:   rgba(99,102,241,.14);
  --accent-strong: #818cf8;
  --success:       #34d399;
  --danger:        #f87171;
}

html, body, [class*="css"]{ font-family:'Inter', sans-serif; }

[data-testid="stAppViewContainer"]{ background: var(--bg); color: var(--fg); }
[data-testid="stHeader"]{ background: transparent; }
#MainMenu, footer, [data-testid="stDecoration"]{ visibility:hidden; height:0; }
.block-container{ padding-top: 1.4rem; max-width: 860px; }

/* Keep the sidebar reopen control visible & themed — never hide this */
[data-testid="stToolbar"]{ background: transparent; }
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[title="Open sidebar"],
button[kind="header"]{
  visibility: visible !important;
  opacity: 1 !important;
  display: flex !important;
  z-index: 999999 !important;
}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button{
  background: var(--surface) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 8px !important;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg{ color: var(--fg) !important; fill: var(--fg) !important; }

::-webkit-scrollbar{ width:6px; height:6px; }
::-webkit-scrollbar-track{ background: transparent; }
::-webkit-scrollbar-thumb{ background: var(--border-strong); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover{ background: var(--muted-2); }

/* ================= Sidebar ================= */
[data-testid="stSidebar"]{ background: var(--bg-elevated); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .block-container{ padding-top: 1.5rem; }
[data-testid="stSidebarUserContent"] label,
[data-testid="stSidebarUserContent"] p{ color: var(--muted); font-size:.85rem; }

.brand{
  display:flex; align-items:center; gap:.7rem;
  padding-bottom: 1.1rem; margin-bottom: 1.1rem; border-bottom: 1px solid var(--border);
}
.brand .mark{
  width:34px; height:34px; border-radius:9px; flex-shrink:0;
  display:flex; align-items:center; justify-content:center;
  background: var(--surface); border: 1px solid var(--border-strong);
  color: var(--accent-strong); font-size:15px; font-weight:800;
}
.brand .title{ font-weight:700; font-size:.98rem; color:var(--fg); letter-spacing:-.01em; line-height:1.2; }
.brand .subtitle{ font-size:.72rem; color:var(--muted-2); }

.section-label{
  font-size:.66rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em;
  color: var(--muted-2); margin: 1.3rem 0 .55rem 0;
}

[data-testid="stFileUploaderDropzone"]{
  background: var(--surface); border: 1px dashed var(--border-strong); border-radius: 10px;
}
[data-testid="stFileUploaderDropzone"]:hover{ border-color: var(--accent); }
[data-testid="stFileUploaderDropzone"] button{
  background: var(--surface-hover) !important; color: var(--fg) !important;
  border: 1px solid var(--border-strong) !important; border-radius: 7px !important; font-size:.82rem !important;
}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span{ color: var(--muted-2) !important; }

[data-testid="stSelectbox"] div[data-baseweb="select"] > div{
  background: var(--surface); border-color: var(--border-strong); border-radius: 8px; color: var(--fg); font-size:.86rem;
}
[data-testid="stSlider"] div[role="slider"]{ background-color: var(--accent); border-color: var(--accent); }
[data-testid="stTickBar"]{ color: var(--muted-2); }

.stButton > button{
  width:100%; background: var(--surface); color: var(--fg);
  border: 1px solid var(--border-strong); border-radius: 8px; font-weight:600; font-size:.84rem;
  padding:.48rem .8rem; transition: all .12s ease;
}
.stButton > button:hover{ border-color: var(--muted-2); background: var(--surface-hover); }
.stButton > button:focus{ box-shadow: none !important; }

.doc-card{
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding:.65rem .75rem; margin-bottom:.75rem;
}
.doc-card .name{ font-size:.82rem; font-weight:600; color:var(--fg); word-break:break-word; line-height:1.3; }
.doc-card .meta{ font-size:.7rem; color:var(--muted-2); margin-top:.1rem; }

.stat-grid{ display:grid; grid-template-columns: 1fr 1fr; gap:.5rem; margin-bottom:.7rem; }
.stat-chip{ background: var(--surface); border:1px solid var(--border); border-radius:10px; padding:.55rem .65rem; }
.stat-chip .label{ font-size:.62rem; color:var(--muted-2); text-transform:uppercase; letter-spacing:.06em; }
.stat-chip .value{ font-size:1rem; font-weight:700; color:var(--fg); font-family:'JetBrains Mono', monospace; margin-top:.1rem; }

.strategy-pill{
  display:inline-flex; align-items:center; gap:.35rem; font-size:.72rem; font-weight:600;
  background: var(--accent-soft); color: var(--accent-strong);
  border: 1px solid rgba(99,102,241,.3); padding:.3rem .6rem; border-radius:7px;
}

/* ================= Top status bar ================= */
.topbar{
  display:flex; align-items:center; justify-content:space-between;
  padding-bottom: 1.1rem; margin-bottom: 1.3rem; border-bottom: 1px solid var(--border);
}
.topbar .left{ display:flex; align-items:center; gap:.6rem; }
.topbar .mark{
  width:30px; height:30px; border-radius:8px; display:flex; align-items:center; justify-content:center;
  background: var(--surface); border:1px solid var(--border-strong); color: var(--accent-strong); font-weight:800; font-size:13px;
}
.topbar .name{ font-weight:700; font-size:.95rem; color:var(--fg); }
.topbar .doc{ font-size:.76rem; color:var(--muted-2); }
.status-dot{
  display:inline-flex; align-items:center; gap:.4rem; font-size:.74rem; color: var(--muted);
  background: var(--surface); border:1px solid var(--border); padding:.3rem .65rem; border-radius:999px;
}
.status-dot .dot{ width:6px; height:6px; border-radius:50%; background: var(--success); }

/* ================= Hero / empty state ================= */
.hero-wrap{ text-align:center; padding: 3.4rem 1rem 1.6rem 1rem; }
.hero-mark{
  width:52px; height:52px; border-radius:13px; margin: 0 auto 1.4rem auto;
  display:flex; align-items:center; justify-content:center;
  background: var(--surface); border:1px solid var(--border-strong); color: var(--accent-strong); font-weight:800; font-size:22px;
}
.hero-title{ font-size:1.85rem; font-weight:800; color:var(--fg); margin:0 0 .6rem 0; letter-spacing:-.02em; }
.hero-sub{ color:var(--muted); font-size:.95rem; max-width:480px; margin:0 auto; line-height:1.6; }
.feature-row{ display:flex; gap:.65rem; justify-content:center; margin-top:2rem; flex-wrap:wrap; }
.feature-chip{
  background: var(--surface); border:1px solid var(--border); border-radius:12px;
  padding:.85rem 1.1rem; font-size:.78rem; color:var(--muted); min-width:155px; text-align:left;
}
.feature-chip b{ color:var(--fg); display:block; font-size:.84rem; margin-bottom:.25rem; font-weight:700; }

/* ================= Chat ================= */
[data-testid="stChatMessage"]{
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding:.2rem .25rem; margin-bottom: .85rem;
}
[data-testid="stChatMessageAvatarUser"]{ background: var(--border-strong) !important; }
[data-testid="stChatMessageAvatarAssistant"]{ background: var(--accent) !important; }

[data-testid="stChatInput"]{ background: var(--surface); border: 1px solid var(--border-strong); border-radius: 12px; }
[data-testid="stChatInput"] textarea{ color: var(--fg) !important; font-size:.9rem; }
[data-testid="stChatInput"] textarea::placeholder{ color: var(--muted-2) !important; }
[data-testid="stBottomBlockContainer"]{ background: transparent; }

[data-testid="stExpander"]{
  background: var(--bg-elevated); border: 1px solid var(--border) !important; border-radius: 10px !important;
}
[data-testid="stExpander"] summary{ color: var(--muted) !important; font-size:.82rem; font-weight:600; }
[data-testid="stExpander"] summary:hover{ color: var(--accent-strong) !important; }

.chunk-card{
  background: var(--surface); border: 1px solid var(--border); border-left: 2px solid var(--accent);
  border-radius: 8px; padding:.65rem .8rem; margin-bottom:.5rem;
}
.chunk-head{ display:flex; justify-content:space-between; align-items:center; margin-bottom:.35rem; }
.chunk-head .tag{ font-size:.7rem; font-weight:600; color:var(--muted); }
.chunk-badge{
  background: var(--accent-soft); color: var(--accent-strong); border:1px solid rgba(99,102,241,.3);
  font-size:.64rem; font-weight:600; padding:.12rem .5rem; border-radius:6px;
}
.chunk-body{
  font-family:'JetBrains Mono', monospace; font-size:.74rem; color: var(--muted); line-height:1.6;
  white-space: pre-wrap;
}

.typing{ display:flex; gap:5px; align-items:center; padding:.4rem .2rem; }
.typing span{ width:6px; height:6px; border-radius:50%; background: var(--muted-2); animation: blink 1.2s infinite ease-in-out; }
.typing span:nth-child(2){ animation-delay:.2s; }
.typing span:nth-child(3){ animation-delay:.4s; }
@keyframes blink{ 0%,80%,100%{ transform:scale(.6); opacity:.35; } 40%{ transform:scale(1); opacity:1; } }

hr{ border-color: var(--border); }
[data-testid="stAlert"]{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; color: var(--fg); }
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)

# =====================================================================
# Session state
# =====================================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================================================================
# Sidebar
# =====================================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="mark">DS</div>
            <div>
                <div class="title">DocSensei</div>
                <div class="subtitle">Document Q&amp;A Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">Indexing</div>', unsafe_allow_html=True)

    strategy = st.selectbox(
        "Chunking Strategy",
        ["Strategy A (500 / 50)", "Strategy B (1000 / 100)"],
    )

    top_k = st.slider(
        "Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=3,
    )

    st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        new_chat_clicked = st.button("New Chat")
    with col_b:
        clear_clicked = st.button("Clear All")

    if new_chat_clicked:
        st.session_state.chat_history = []
        st.rerun()

    if clear_clicked:
        st.session_state.clear()
        st.rerun()

# =====================================================================
# Document processing
# =====================================================================

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1],
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

        with st.spinner("Reading document and building the vector index…"):
            docs = load_document(file_path)

            if strategy == "Strategy A (500 / 50)":
                chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
            else:
                chunks = split_documents(docs, chunk_size=1000, chunk_overlap=100)

            vectorstore = create_vectorstore(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.current_file = uploaded_file.name
            st.session_state.strategy = strategy
            st.session_state.pages = len(docs)
            st.session_state.chunks = len(chunks)
            st.session_state.index_time = round(time.time() - start_time, 2)
            st.session_state.chat_history = []

    # -----------------------------------------------------------------
    # Sidebar: document insights
    # -----------------------------------------------------------------

    with st.sidebar:
        st.markdown('<div class="section-label">Document Insights</div>', unsafe_allow_html=True)

        size_kb = round(uploaded_file.size / 1024, 1)

        st.markdown(
            f"""
            <div class="doc-card">
                <div class="name">{uploaded_file.name}</div>
                <div class="meta">{size_kb} KB</div>
            </div>
            <div class="stat-grid">
                <div class="stat-chip">
                    <div class="label">Pages</div>
                    <div class="value">{st.session_state.pages}</div>
                </div>
                <div class="stat-chip">
                    <div class="label">Chunks</div>
                    <div class="value">{st.session_state.chunks}</div>
                </div>
                <div class="stat-chip">
                    <div class="label">Index Time</div>
                    <div class="value">{st.session_state.index_time}s</div>
                </div>
                <div class="stat-chip">
                    <div class="label">Top K</div>
                    <div class="value">{top_k}</div>
                </div>
            </div>
            <span class="strategy-pill">{strategy}</span>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------
    # Top status bar (replaces the old big centered hero once a doc is loaded)
    # -----------------------------------------------------------------

    st.markdown(
        f"""
        <div class="topbar">
            <div class="left">
                <div class="mark">DS</div>
                <div>
                    <div class="name">DocSensei</div>
                    <div class="doc">{uploaded_file.name}</div>
                </div>
            </div>
            <div class="status-dot"><span class="dot"></span>Indexed &amp; ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------
    # Helper: render one source chunk card
    # -----------------------------------------------------------------

    def render_chunk(index, page, preview):
        st.markdown(
            f"""
            <div class="chunk-card">
                <div class="chunk-head">
                    <span class="tag">Chunk {index}</span>
                    <span class="chunk-badge">Page {page}</span>
                </div>
                <div class="chunk-body">{preview}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------
    # Replay chat history
    # -----------------------------------------------------------------

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            sources = message.get("sources")
            if sources:
                with st.expander(f"View {len(sources)} source chunk(s)"):
                    for i, source in enumerate(sources, start=1):
                        render_chunk(i, source["page"], source["preview"])

    # -----------------------------------------------------------------
    # Chat input
    # -----------------------------------------------------------------

    question = st.chat_input("Ask a question about your document…")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown(
                '<div class="typing"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True,
            )

            answer, sources = ask_question(
                st.session_state.vectorstore,
                question,
                top_k,
            )

            placeholder.markdown(answer)

            source_data = []
            if sources:
                with st.expander(f"View {len(sources)} source chunk(s)"):
                    for i, source in enumerate(sources, start=1):
                        page = source.metadata.get("page", "Unknown")
                        if page != "Unknown":
                            page = page + 1

                        preview = source.page_content
                        if len(preview) > 500:
                            preview = preview[:500] + "…"

                        source_data.append({"page": page, "preview": preview})
                        render_chunk(i, page, preview)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": source_data}
        )

else:
    # -------------------------------------------------------------
    # Empty state / landing hero
    # -------------------------------------------------------------

    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-mark">DS</div>
            <div class="hero-title">Talk to your documents</div>
            <div class="hero-sub">
                Upload a PDF or DOCX from the sidebar and ask questions.
                Every answer is grounded only in what's inside the file —
                no guessing, no fabricated facts.
            </div>
            <div class="feature-row">
                <div class="feature-chip"><b>Grounded</b>Answers cite only your document</div>
                <div class="feature-chip"><b>Configurable</b>Adjustable chunking strategy</div>
                <div class="feature-chip"><b>Traceable</b>Every answer shows its source chunks</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
