import os
import uuid
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Tuple
import streamlit as st
from core.config import PERSIST_DIR, COLLECTION_NAME, EMBED_MODEL, DEFAULT_TOP_K
from core.file_readers import split_text

# Safe import for sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = EMBED_MODEL):
    """Returns (model, embedding_function_for_chroma)"""
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed. Run: pip install sentence-transformers")
    
    model = SentenceTransformer(model_name)

    class _EmbedFunc(embedding_functions.EmbeddingFunction):
        def __init__(self):
            super().__init__()

        def __call__(self, texts: List[str]) -> List[List[float]]:
            vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return vecs.tolist()

    return model, _EmbedFunc()

@st.cache_resource(show_spinner=False)
def get_chroma_client(persist_dir: str = PERSIST_DIR):
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)

def ensure_collection(client: chromadb.ClientAPI, embed_fn) -> chromadb.Collection:
    try:
        col = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    except Exception:
        col = client.create_collection(COLLECTION_NAME, embedding_function=embed_fn)
    return col

def add_document(chroma_col: chromadb.Collection, text: str, meta: Dict):
    # 1. Check if this file is already in the DB
    source_name = meta.get("source", "")
    existing = chroma_col.get(where={"source": source_name})
    
    if existing["ids"]:
        # File exists! Return 0 so we don't add it again.
        return 0
        
    # 2. If not, add it normally
    if not text.strip():
        return 0
        
    chunks = split_text(text)
    ids = [str(uuid.uuid4()) for _ in chunks]
    chroma_col.add(documents=chunks, metadatas=[meta] * len(chunks), ids=ids)
    return len(chunks)

def retrieve(chroma_col: chromadb.Collection, query: str, top_k: int = DEFAULT_TOP_K):
    res = chroma_col.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids = [str(i) for i in range(len(docs))]
    return list(zip(docs, metas, dists, ids))