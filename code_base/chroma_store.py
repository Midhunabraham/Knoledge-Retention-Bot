import os
import uuid
import chromadb
from core.config import COLLECTION_NAME, MAX_CHARS_PER_CHUNK


def get_chroma_client(persist_dir):
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def ensure_collection(client, embed_fn):
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    except:
        return client.create_collection(COLLECTION_NAME, embedding_function=embed_fn)


def add_document(chroma_col, text, meta):
    chunks, buf, size = [], [], 0
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    for p in paras:
        if size + len(p) > MAX_CHARS_PER_CHUNK and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [p], len(p)
        else:
            buf.append(p)
            size += len(p)

    if buf:
        chunks.append("\n\n".join(buf))

    ids = [str(uuid.uuid4()) for _ in chunks]
    chroma_col.add(documents=chunks, metadatas=[meta] * len(chunks), ids=ids)
    return len(chunks)


def retrieve(chroma_col, query, top_k):
    res = chroma_col.query(query_texts=[query], n_results=top_k)
    return list(zip(res["documents"][0], res["metadatas"][0]))
