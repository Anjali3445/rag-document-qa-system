"""Streamlit web app:  streamlit run app.py"""
import tempfile
from pathlib import Path
import streamlit as st
from rag import RAGPipeline

st.set_page_config(page_title="RAG Document QA", page_icon="📄", layout="wide")
st.title("📄 RAG-Based Document QA System")
st.caption("Upload documents, then ask questions. Answers are grounded in your files.")

with st.sidebar:
    st.header("Settings")
    files = st.file_uploader("Upload documents", type=["txt", "md", "pdf", "docx"],
                             accept_multiple_files=True)
    use_samples = st.checkbox("Use sample documents", value=not files)
    backend = st.selectbox("Retriever", ["tfidf", "embeddings"],
                           help="'embeddings' needs: pip install sentence-transformers")
    k = st.slider("Chunks to retrieve (top-k)", 1, 8, 3)
    chunk_size = st.slider("Chunk size (words)", 50, 300, 120, 10)
    use_llm = st.checkbox("Use LLM if API key is set", value=True)
    build = st.button("Build / Rebuild index", type="primary")

if build or "rag" not in st.session_state:
    paths = []
    if files:
        tmp = Path(tempfile.mkdtemp())
        for f in files:
            (tmp / f.name).write_bytes(f.getbuffer())
        paths.append(tmp)
    if use_samples or not files:
        paths.append(Path("data/documents"))
    rag = RAGPipeline(backend, chunk_size, chunk_size // 4, k, use_llm)
    st.session_state.stats = rag.ingest(paths)
    st.session_state.rag = rag
    st.session_state.history = []

rag = st.session_state.rag
rag.k, rag.use_llm = k, use_llm
st.sidebar.success(f"Indexed {st.session_state.stats['documents']} docs / "
                   f"{st.session_state.stats['chunks']} chunks")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        st.caption(f"mode: {turn['mode']} | retriever: {turn['retriever']}")
        with st.expander("Sources"):
            for s in turn["sources"]:
                st.markdown(f"**{s['source']}** · chunk {s['chunk_id']} · score {s['score']}")
                st.write(s["text"])

if q := st.chat_input("Ask a question about your documents..."):
    st.session_state.history.append(rag.ask(q))
    st.rerun()
