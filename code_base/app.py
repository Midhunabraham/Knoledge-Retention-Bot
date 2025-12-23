import streamlit as st
from core.config import APP_TITLE, EMBED_MODEL, PERSIST_DIR, COLLECTION_NAME, MAX_CHARS_PER_CHUNK
from core import file_readers, chroma_store, prompt_builder, llm_ollama

# ---------------------------
# Streamlit UI Setup
# ---------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
st.title(APP_TITLE)
st.caption("RAG prototype — upload company docs/logs, then ask questions. Documents are embedded locally.")

# ---------------------------
# Sidebar & Settings
# ---------------------------
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

# ---------------------------
# Initialization
# ---------------------------
with st.spinner("Initializing vector database and embedder..."):
    try:
        model, embed_fn = chroma_store.get_embedder(EMBED_MODEL)
    except Exception as e:
        st.error(f"Embedder initialization failed: {e}")
        st.stop()

    try:
        client = chroma_store.get_chroma_client(PERSIST_DIR)
        collection = chroma_store.ensure_collection(client, embed_fn)
    except Exception as e:
        st.error(f"Chroma initialization failed: {e}")
        st.stop()

# ---------------------------
# Event Handling
# ---------------------------
if clear_btn:
    with st.spinner("Clearing vector store..."):
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = chroma_store.ensure_collection(client, embed_fn)
        st.success("Knowledge base cleared.")

if ingest_btn and uploads:
    added_total = 0
    for up in uploads:
        text, fname = file_readers.extract_text_from_upload(up)
        meta = {"source": fname}
        added = chroma_store.add_document(collection, text, meta)
        added_total += added
    st.success(f"Ingested {len(uploads)} files into {added_total} chunks.")

# ---------------------------
# Chat Interface
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask about your uploaded knowledge…")

if question:
    # User Message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and drafting answer…"):
            # 1. Retrieve
            hits = chroma_store.retrieve(collection, question, top_k=top_k)
            
            if not hits or all(h[0] == [] for h in hits):
                st.warning("No relevant context found. Please upload documents or broaden your question.")
                answer = "No relevant context found."
            else:
                # 2. Build Prompt (includes summarization of chunks)
                system_msg, user_msg = prompt_builder.build_prompt(question, hits)
                
                # 3. Generate Answer
                answer = llm_ollama.call_ollama(system_msg, user_msg)
                
                st.markdown(answer)

                # Show Sources
                if hits:
                    with st.expander("Sources used"):
                        for i, (doc, meta, dist, _id) in enumerate(hits, start=1):
                            title = meta.get("source", f"Doc {_id[:8]}") if isinstance(meta, dict) else f"Doc {_id[:8]}"
                            st.markdown(f"**[S{i}]** {title} — distance: {dist:.4f}")
                            st.code(doc[:MAX_CHARS_PER_CHUNK] + ("…" if len(doc) > MAX_CHARS_PER_CHUNK else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("""---
**Tips**: Works fully offline with Ollama. Make sure `ollama serve` is running.
""")