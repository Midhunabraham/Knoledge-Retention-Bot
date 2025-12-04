import os
import time
import json
import pickle
import numpy as np
import requests
from tqdm import tqdm
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------
OLLAMA_API = "http://localhost:11434/api/embed"
MODEL = "llama-mini-embed"  # if available
DATA_DIR = r"C:\Users\midhu\KnowledgeBot"   # Folder with PDFs and TXTs
DB_FAISS_PATH = "faiss_index"
TEMP_SAVE_PATH = "embeddings_temp.pkl"
BATCH_SIZE = 1
TIMEOUT = 20  # seconds
RETRIES = 3

# ---------------------------------------
# LOAD FILES
# ---------------------------------------
def load_documents():
    docs = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    docs.append(f.read())
            elif file.endswith(".pdf"):
                pdf_reader = PdfReader(path)
                text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                docs.append(text)
    return docs

# ---------------------------------------
# CHUNKING
# ---------------------------------------
def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, length_function=len)
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))
    print(f"📄 Total chunks created: {len(chunks)}")
    return chunks

# ---------------------------------------
# EMBEDDING API (Batch + Retry)
# ---------------------------------------
def embed_batch(texts):
    for attempt in range(RETRIES):
        try:
            payload = {"model": "nomic-embed-text", "input": texts}
            response = requests.post(OLLAMA_API, json=payload, timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                if "embeddings" in data:
                    return np.array(data["embeddings"], dtype="float32")
            print(f"⚠️ Ollama error: {response.text}")
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    return None

# ---------------------------------------
# MAIN BUILDER
# ---------------------------------------
def build_knowledge_db():
    documents = load_documents()
    chunks = chunk_documents(documents)

    embeddings, docs_with_meta = [], []
    start_index = 0

    # Resume progress if available
    if os.path.exists(TEMP_SAVE_PATH):
        print("🔄 Resuming from previous progress...")
        with open(TEMP_SAVE_PATH, "rb") as f:
            saved = pickle.load(f)
            embeddings, docs_with_meta = saved["embeddings"], saved["docs"]
            start_index = len(embeddings)

    total_batches = (len(chunks) - start_index + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(start_index, len(chunks), BATCH_SIZE), desc="🔍 Embedding chunks"):
        batch = chunks[i:i + BATCH_SIZE]
        batch_vectors = embed_batch(batch)
        print(embed_batch([chunks[0]]))
        if batch_vectors is not None:
            for j, vector in enumerate(batch_vectors):
                embeddings.append(vector)
                docs_with_meta.append(Document(page_content=batch[j], metadata={"id": i + j}))
        else:
            print(f"⚠️ Skipping batch starting at chunk {i}")

        # Save progress every 5 batches
        if i % (BATCH_SIZE * 5) == 0:
            with open(TEMP_SAVE_PATH, "wb") as f:
                pickle.dump({"embeddings": embeddings, "docs": docs_with_meta}, f)
            print("💾 Progress saved temporarily...")

    # Final save
    if embeddings:
        faiss_index = FAISS.from_embeddings(embeddings, docs_with_meta)
        faiss_index.save_local(DB_FAISS_PATH)
        print(f"✅ FAISS index saved at {DB_FAISS_PATH} with {len(embeddings)} vectors")

        # Clean up temp file
        if os.path.exists(TEMP_SAVE_PATH):
            os.remove(TEMP_SAVE_PATH)
    else:
        print("⚠️ No embeddings created. Check Ollama server.")

# ---------------------------------------
if __name__ == "__main__":
    build_knowledge_db()
