import os
import json
import streamlit as st
from datetime import datetime

# Import modular components
from src.config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    EMBEDDING_OPTIONS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_TEMPERATURE,
    get_api_key,
)
from src.document_loader import extract_documents_from_files
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store, search_with_relevance
from src.rag_engine import answer_question

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NexusRAG | Precision Document QA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom modern CSS for glassmorphism, badge pills, and clean styling
st.markdown("""
<style>
    /* Metric and status badges */
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-info {
        background-color: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Citation card styling */
    .citation-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .confidence-pill {
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        color: #FFFFFF;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .snippet-text {
        font-family: monospace;
        font-size: 0.82rem;
        color: #CBD5E1;
        background-color: #0F172A;
        padding: 8px;
        border-radius: 4px;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # List of {"role": "user"|"assistant", "content": str, "citations": []}

if "chat_history_tuples" not in st.session_state:
    st.session_state.chat_history_tuples = []  # List of (user_str, assistant_str)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "indexed_docs_stats" not in st.session_state:
    st.session_state.indexed_docs_stats = None


# -----------------------------------------------------------------------------
# 3. Sidebar - Configurations & Ingestion
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    st.caption("Configure intelligence parameters & documents")

    # --- API Key Section ---
    st.markdown("### 🔑 Authentication")
    detected_key = get_api_key()
    user_api_key = st.text_input(
        "Google Gemini API Key",
        value=detected_key or "",
        type="password",
        help="Get a free key from Google AI Studio (https://aistudio.google.com/)",
        placeholder="Enter AI Studio API Key..."
    )

    if user_api_key:
        st.markdown('<span class="metric-badge badge-success">✓ Key Configured</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="metric-badge badge-warning">⚠️ API Key Required</span>', unsafe_allow_html=True)

    st.markdown("---")

    # --- RAG Model & Precision Settings ---
    st.markdown("### 🧠 Model Architecture")
    selected_model = st.selectbox(
        "Reasoning LLM",
        options=AVAILABLE_MODELS,
        index=0,
        help="Gemini 1.5 Pro provides highest analytical reasoning; Flash offers lowest latency."
    )

    embedding_label = st.selectbox(
        "Embedding Provider",
        options=list(EMBEDDING_OPTIONS.keys()),
        index=0,
        help="Google text-embedding-004 is recommended for highest retrieval precision."
    )
    selected_embedding_provider = EMBEDDING_OPTIONS[embedding_label]

    # --- Precision Tuning (Accordion) ---
    with st.expander("🎯 Precision & Grounding Settings", expanded=False):
        temperature = st.slider(
            "Temperature (Strictness)",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_TEMPERATURE),
            step=0.05,
            help="Lower values (0.0 - 0.1) enforce deterministic, fact-grounded responses without hallucination."
        )

        top_k = st.slider(
            "Retrieved Passages (Top-K)",
            min_value=1,
            max_value=10,
            value=int(DEFAULT_TOP_K),
            step=1,
            help="Number of most relevant context chunks passed to the LLM."
        )

        min_confidence = st.slider(
            "Min Relevance Threshold (%)",
            min_value=0,
            max_value=80,
            value=0,
            step=5,
            help="Filter out retrieved passages below this confidence match percentage."
        )

        chunk_size = st.number_input(
            "Chunk Size (Chars)",
            min_value=200,
            max_value=2000,
            value=int(DEFAULT_CHUNK_SIZE),
            step=50,
            help="Target size for text splitting."
        )

        chunk_overlap = st.number_input(
            "Chunk Overlap (Chars)",
            min_value=0,
            max_value=500,
            value=int(DEFAULT_CHUNK_OVERLAP),
            step=25,
            help="Preserves continuity across split boundaries."
        )

    st.markdown("---")

    # --- Document Ingestion Section ---
    st.markdown("### 📁 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Source Documents",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True,
        help="Upload one or multiple documents to build your vector knowledge base."
    )

    col_proc, col_clear = st.columns(2)

    with col_proc:
        process_clicked = st.button("⚡ Index Docs", use_container_width=True, type="primary")

    with col_clear:
        clear_index_clicked = st.button("🗑️ Reset", use_container_width=True)

    if clear_index_clicked:
        st.session_state.vector_store = None
        st.session_state.indexed_docs_stats = None
        st.session_state.messages = []
        st.session_state.chat_history_tuples = []
        st.success("Vector store and chat history cleared!")
        st.rerun()

    if process_clicked:
        if not user_api_key and selected_embedding_provider == "google":
            st.error("Please enter a Google API Key before indexing with Google Embeddings.")
        elif not uploaded_files:
            st.warning("Please select at least one document to index.")
        else:
            with st.spinner("Extracting, chunking, and embedding documents..."):
                try:
                    # 1. Extract documents
                    raw_docs, extraction_stats = extract_documents_from_files(uploaded_files)

                    if not raw_docs:
                        st.error("No valid text could be extracted from the uploaded files.")
                    else:
                        # 2. Load embedding model
                        emb_model = get_embedding_model(
                            provider=selected_embedding_provider,
                            api_key=user_api_key
                        )

                        # 3. Create FAISS Vector Index
                        vector_store, chunk_count = create_vector_store(
                            documents=raw_docs,
                            embeddings=emb_model,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )

                        st.session_state.vector_store = vector_store
                        st.session_state.indexed_docs_stats = {
                            "files_count": len(uploaded_files),
                            "file_names": [f.name for f in uploaded_files],
                            "total_chunks": chunk_count,
                            "total_characters": extraction_stats["total_characters"],
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                        st.sidebar.success(f"Successfully indexed {chunk_count} chunks from {len(uploaded_files)} file(s)!")
                        st.rerun()

                except Exception as e:
                    st.sidebar.error(f"Indexing Failed: {str(e)}")

    # Index Status Display in Sidebar
    if st.session_state.indexed_docs_stats:
        stats = st.session_state.indexed_docs_stats
        st.markdown("#### 📊 Knowledge Base Status")
        st.markdown(
            f'<span class="metric-badge badge-success">Active: {stats["files_count"]} File(s)</span>'
            f'<span class="metric-badge badge-info">{stats["total_chunks"]} Chunks</span>',
            unsafe_allow_html=True
        )
        with st.expander("Indexed Files", expanded=False):
            for name in stats["file_names"]:
                st.write(f"- 📄 `{name}`")
            st.caption(f"Indexed at: {stats['timestamp']}")

    st.markdown("---")

    # Chat management
    if st.session_state.messages:
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history_tuples = []
            st.rerun()

        # Export chat
        chat_export = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="📥 Export Chat (JSON)",
            data=chat_export,
            file_name=f"rag_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )


# -----------------------------------------------------------------------------
# 4. Main Interface - Header & Status
# -----------------------------------------------------------------------------
st.title("📚 NexusRAG: High-Precision Document QA Intelligence")
st.markdown(
    "Ask questions about your uploaded documents with **100% strictly grounded answers**, "
    "verified source page citations, and zero-hallucination guardrails."
)

# Status notification banner
if st.session_state.vector_store is None:
    st.info("👈 **Get Started**: Enter your Google Gemini API Key and upload your PDF or text documents in the sidebar, then click **Index Docs**.")
else:
    stats = st.session_state.indexed_docs_stats
    st.markdown(
        f'<span class="metric-badge badge-success">✓ Knowledge Base Online</span>'
        f'<span class="metric-badge badge-info">{stats["files_count"]} Document(s)</span>'
        f'<span class="metric-badge badge-info">{stats["total_chunks"]} Chunks</span>'
        f'<span class="metric-badge badge-success">Model: {selected_model}</span>',
        unsafe_allow_html=True
    )

# Starter Suggestion Prompts
if st.session_state.vector_store is not None and len(st.session_state.messages) == 0:
    st.markdown("##### 💡 Suggested Questions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Summarize the main topics of the document(s)", use_container_width=True):
            st.session_state.pending_prompt = "Summarize the main topics and key conclusions of the uploaded document(s)."
            st.rerun()
        if st.button("🔍 Extract all key figures, statistics, and numerical metrics", use_container_width=True):
            st.session_state.pending_prompt = "Extract all key numbers, dates, statistics, and metrics stated in the documents with exact page citations."
            st.rerun()
    with col2:
        if st.button("⚡ What are the primary objectives or action items?", use_container_width=True):
            st.session_state.pending_prompt = "What are the primary objectives, deliverables, or action items outlined in the documents?"
            st.rerun()
        if st.button("⚠️ List any critical limitations, risks, or conditions noted", use_container_width=True):
            st.session_state.pending_prompt = "List all risks, exceptions, constraints, or warnings specified in the text."
            st.rerun()


# -----------------------------------------------------------------------------
# 5. Conversation History Rendering
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    citations = msg.get("citations", [])

    with st.chat_message(role):
        st.markdown(content)

        if citations:
            with st.expander(f"📑 Verified Sources & Evidence ({len(citations)} citations)", expanded=False):
                for idx, cite in enumerate(citations, 1):
                    source = cite.get("source", "Unknown")
                    page = cite.get("page", 1)
                    conf = cite.get("confidence_score", 0.0)
                    snippet = cite.get("snippet", "")
                    full = cite.get("full_content", "")

                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">
                            <span>📄 Source {idx}: <strong>{source}</strong> (Page {page})</span>
                            <span class="confidence-pill">{conf}% Match</span>
                        </div>
                        <div class="snippet-text">{snippet}</div>
                    </div>
                    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. User Query Handling & RAG Execution
# -----------------------------------------------------------------------------
# Handle either chat input or starter button click
prompt_from_input = st.chat_input("Ask any question regarding your uploaded documents...")
pending_prompt = st.session_state.pop("pending_prompt", None)

active_prompt = prompt_from_input or pending_prompt

if active_prompt:
    if not user_api_key:
        st.error("Please enter a Google Gemini API Key in the sidebar to ask questions.")
    elif st.session_state.vector_store is None:
        st.warning("Please upload and index documents in the sidebar first.")
    else:
        # 1. Append and render User message
        st.session_state.messages.append({
            "role": "user",
            "content": active_prompt,
            "citations": []
        })
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # 2. Retrieve & Generate Assistant Answer
        with st.chat_message("assistant"):
            with st.spinner("Searching vector index and synthesizing grounded answer..."):
                try:
                    # Retrieve scored documents
                    min_thresh = float(min_confidence) / 100.0
                    scored_docs = search_with_relevance(
                        vector_store=st.session_state.vector_store,
                        query=active_prompt,
                        k=top_k,
                        min_score_threshold=min_thresh
                    )

                    # Execute grounded generation
                    result = answer_question(
                        query=active_prompt,
                        scored_docs=scored_docs,
                        chat_history_tuples=st.session_state.chat_history_tuples,
                        model_name=selected_model,
                        api_key=user_api_key,
                        temperature=temperature,
                    )

                    answer_text = result["answer"]
                    citations = result["citations"]

                    # Display answer
                    st.markdown(answer_text)

                    # Display citations
                    if citations:
                        with st.expander(f"📑 Verified Sources & Evidence ({len(citations)} citations)", expanded=False):
                            for idx, cite in enumerate(citations, 1):
                                source = cite.get("source", "Unknown")
                                page = cite.get("page", 1)
                                conf = cite.get("confidence_score", 0.0)
                                snippet = cite.get("snippet", "")

                                st.markdown(f"""
                                <div class="citation-card">
                                    <div class="citation-header">
                                        <span>📄 Source {idx}: <strong>{source}</strong> (Page {page})</span>
                                        <span class="confidence-pill">{conf}% Match</span>
                                    </div>
                                    <div class="snippet-text">{snippet}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # Update session state history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations
                    })
                    st.session_state.chat_history_tuples.append((active_prompt, answer_text))

                except Exception as e:
                    error_msg = f"❌ An error occurred during answer generation: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "citations": []
                    })
