import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

DB_FAISS_PATH = "faiss_index"
OLLAMA_API = "http://localhost:11434/api/chat"

# -------------------------------
# Load FAISS DB
# -------------------------------
def load_db():
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# -------------------------------
# Query FAISS + Ollama
# -------------------------------
def query_bot(query, k=3):
    db = load_db()
    docs = db.similarity_search(query, k=k)
    context = "\n\n".join([d.page_content for d in docs])

    payload = {
        "model": "llama2:7b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    }

    response = requests.post(OLLAMA_API, json=payload, stream=False)
    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        return f"❌ Error from Ollama: {response.text}"

# -------------------------------
if __name__ == "__main__":
    while True:
        query = input("\nAsk me something (or 'exit'): ")
        if query.lower() == "exit":
            break
        answer = query_bot(query)
        print("\n🤖 Answer:", answer)
