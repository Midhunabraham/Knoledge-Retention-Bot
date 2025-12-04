# app.py
"""
Knowledge Retention & Training Bot
Streamlit + ChromaDB (local vectors) + Ollama (local LLM)
"""

import os
import io
import uuid
from typing import List, Dict, Tuple

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import requests
import traceback

# ---------------------------
# App Constants & Helpers
# ---------------------------
APP_TITLE = "Knowledge Retention & Training Bot (Prototype)"
PERSIST_DIR = ".chroma"
COLLECTION_NAME = "company_knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 4
MAX_CHARS_PER_CHUNK = 1000        # chunk truncation before summarization
MAX_TOTAL_CONTEXT_CHARS = 500    # max context sent to Ollama
SUMMARIZE_THRESHOLD = 600         # summarize chunks longer than this
OLLAMA_URL = "http://localhost:11434/api"
# sentence-transformers is optional but recommended
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = EMBED_MODEL):
    """
    Returns (model, embedding_function_for_chroma)
    """
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. Run: pip install sentence-transformers"
        )
    model = SentenceTransformer(model_name)

    class _EmbedFunc(embedding_functions.EmbeddingFunction):
        def __init__(self):
            # The base class doesn't take args in current chroma versions; we just subclass for compatibility
            super().__init__()

        def __call__(self, texts: List[str]) -> List[List[float]]:
            # SentenceTransformer.encode returns numpy array by default; convert to list of lists
            vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return vecs.tolist()

    return model, _EmbedFunc()


@st.cache_resource(show_spinner=False)
def get_chroma_client(persist_dir: str = PERSIST_DIR):
    """
    Create a local persistent Chroma client. Directory will be created if missing.
    """
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def split_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], []
    size = 0
    for p in paras:
        if size + len(p) > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [p], len(p)
        else:
            buf.append(p)
            size += len(p)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def read_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)


def extract_text_from_upload(upload) -> Tuple[str, str]:
    name = upload.name
    data = upload.read()
    ext = os.path.splitext(name.lower())[1]
    if ext in [".pdf"]:
        return read_pdf(io.BytesIO(data)), name
    else:
        try:
            return data.decode("utf-8", errors="ignore"), name
        except Exception:
            return "", name


def ensure_collection(client: chromadb.ClientAPI, embed_fn) -> chromadb.Collection:
    """
    Get or create a collection and attach the embedding function so that Chroma will call it on add().
    """
    try:
        # try to get existing
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        col = client.create_collection(COLLECTION_NAME, embedding_function=embed_fn)
        return col
    # if we have it already, ensure the embedding function is set (some versions accept this arg)
    try:
        col = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    except Exception:
        # fallback: use the collection as is
        pass
    return col


def add_document(chroma_col: chromadb.Collection, text: str, meta: Dict):
    if not text.strip():
        return 0
    chunks = split_text(text)
    ids = [str(uuid.uuid4()) for _ in chunks]
    # When collection was created with embedding_function, Chroma will compute embeddings automatically.
    chroma_col.add(documents=chunks, metadatas=[meta] * len(chunks), ids=ids)
    return len(chunks)


def retrieve(chroma_col: chromadb.Collection, query: str, top_k: int = DEFAULT_TOP_K):
    res = chroma_col.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]  # remove "ids"
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    
    # generate fake IDs if needed
    ids = [str(i) for i in range(len(docs))]

    return list(zip(docs, metas, dists, ids))


def call_ollama(system_prompt: str, user_prompt: str, model: str = "mistral", timeout: int = 120) -> str:
    """
    Calls local Ollama REST API. Tries /api/chat first (multi-turn), falls back to /api/generate.
    """
    base = OLLAMA_URL
    payload_chat = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    try:
        # prefer /api/chat (multi-turn)
        resp = requests.post(f"{base}/chat", json=payload_chat, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Ollama /api/chat returns {"message": {"content": "..."}}
        if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
            return data["message"]["content"]
        # fallback: some Ollama versions return 'output' or list; try to safely extract text
        if isinstance(data, dict):
            return str(data)
        return resp.text
    except Exception as e_chat:
        # try /api/generate fallback
        try:
            payload_gen = {"model": model, "prompt": f"{system_prompt}\n\n{user_prompt}", "stream": False}
            resp = requests.post(f"{base}/generate", json=payload_gen, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # generate endpoint structures vary; attempt to extract content
            if isinstance(data, dict):
                # common returned fields include 'responses' or 'text' etc
                if "text" in data:
                    return data["text"]
                if "responses" in data and isinstance(data["responses"], list) and data["responses"]:
                    return data["responses"][0].get("text", str(data["responses"][0]))
            return resp.text
        except Exception as e_gen:
            # Return both errors for debugging
            tb = traceback.format_exc()
            return f"❗ Unable to call Ollama (chat error: {e_chat}; generate error: {e_gen}).\nTraceback:\n{tb}"


def summarize_chunk(chunk: str) -> str:
    """
    Summarize the chunk using Ollama if it's too long
    """
    if len(chunk) <= SUMMARIZE_THRESHOLD:
        return chunk
    system_prompt = "You are a helpful assistant. Summarize the following text in concise key points."
    user_prompt = f"Text:\n{chunk}"
    summary = call_ollama(system_prompt, user_prompt)
    # be safe: return truncated summary
    return summary[:MAX_CHARS_PER_CHUNK]


def build_prompt(question: str, hits: List[Tuple[str, Dict, float, str]]) -> Tuple[str, str]:
    sources_block = []
    total_chars = 0
    for i, (doc, meta, dist, _id) in enumerate(hits, start=1):
        title = meta.get("source", "Uploaded Doc") if isinstance(meta, dict) else "Uploaded Doc"
        chunk_text = summarize_chunk(doc)
        if total_chars + len(chunk_text) > MAX_TOTAL_CONTEXT_CHARS:
            remaining = MAX_TOTAL_CONTEXT_CHARS - total_chars
            if remaining <= 0:
                break
            chunk_text = chunk_text[:remaining]
        sources_block.append(f"[S{i}] {title}\n{chunk_text}")
        total_chars += len(chunk_text)
        if total_chars >= MAX_TOTAL_CONTEXT_CHARS:
            break

    context = "\n\n".join(sources_block) if sources_block else "No context available."
    system = (
        "You are an internal company knowledge assistant. Answer clearly and concisely. "
        "Cite sources as [S1], [S2]. If the answer is not in the provided context, say you don't know."
    )
    user = f"Question: {question}\n\nContext:\n{context}"
    return system, user


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
st.title(APP_TITLE)
st.caption("RAG prototype — upload company docs/logs, then ask questions. Documents are embedded locally.")

with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("Top-K Context Chunks", 1, 5, 2)
    st.markdown("---")
    st.subheader("Add Documents")
    uploads = st.file_uploader(
        "Upload PDFs, TXT, logs, code (multi-select)",
        type=["pdf", "txt", "md", "log", "c", "h", "py", "json"],
        accept_multiple_files=True,
    )
    ingest_btn = st.button("Ingest to Knowledge Base", use_container_width=True)
    clear_btn = st.button("Clear Knowledge Base (danger)", type="secondary", use_container_width=True)

# Initialize client & embedder
with st.spinner("Initializing vector database and embedder..."):
    try:
        model, embed_fn = get_embedder(EMBED_MODEL)
    except Exception as e:
        st.error(f"Embedder initialization failed: {e}")
        st.stop()

    try:
        client = get_chroma_client(PERSIST_DIR)
        collection = ensure_collection(client, embed_fn)
    except Exception as e:
        st.error(f"Chroma initialization failed: {e}")
        st.stop()

if clear_btn:
    with st.spinner("Clearing vector store..."):
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = ensure_collection(client, embed_fn)
        st.success("Knowledge base cleared.")

if ingest_btn and uploads:
    added_total = 0
    for up in uploads:
        text, fname = extract_text_from_upload(up)
        meta = {"source": fname}
        added = add_document(collection, text, meta)
        added_total += added
    st.success(f"Ingested {len(uploads)} files into {added_total} chunks.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask about your uploaded knowledge…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and drafting answer…"):
            hits = retrieve(collection, question, top_k=top_k)
            if not hits or all(h[0] == [] for h in hits):
                st.warning("No relevant context found. Please upload documents or broaden your question.")
                answer = "No relevant context found."
            else:
                system, user = build_prompt(question, hits)
                answer = call_ollama(system, user)
                st.markdown(answer)

                if hits:
                    with st.expander("Sources used"):
                        for i, (doc, meta, dist, _id) in enumerate(hits, start=1):
                            title = meta.get("source", f"Doc {_id[:8]}") if isinstance(meta, dict) else f"Doc {_id[:8]}"
                            st.markdown(f"**[S{i}]** {title} — distance: {dist:.4f}")
                            st.code(doc[:MAX_CHARS_PER_CHUNK] + ("…" if len(doc) > MAX_CHARS_PER_CHUNK else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown(
    """
---
**Tips**
- Upload a few highly relevant PDFs/logs first. Then ask targeted questions.
- If answers look generic, increase Top-K or add more detailed documents.
- Works fully offline with Ollama — make sure `ollama serve` (or `ollama` daemon) is running.
"""
)
