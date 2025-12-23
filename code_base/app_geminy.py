import streamlit as st
from core.config import APP_TITLE, EMBED_MODEL, PERSIST_DIR, COLLECTION_NAME, MAX_CHARS_PER_CHUNK
from core import file_readers, chroma_store, prompt_builder, llm_gemini

# ---------------------------
# Streamlit UI Setup
# ---------------------------
st.set_page_config(page_title=f"{APP_TITLE} - Gemini", page_icon="♊", layout="wide")
st.title(f"♊ {APP_TITLE}")
st.caption("Powered by Google Gemini 2.5/3.0 + Local Vector Store")

# ---------------------------
# Sidebar & Settings
# ---------------------------
with st.sidebar:
	
    st.markdown("---")
    st.subheader("Connect Data Sources")
    
    # GMAIL BUTTON
    if st.button("📥 Import Recent Emails (Gmail)"):
        with st.spinner("Connecting to Gmail... (Check popup window)"):
            try:
                # Lazy import to avoid errors if module is missing
                from core import gmail_connector
                
                status, emails = gmail_connector.fetch_recent_emails(max_count=20)
                
                if status == "MISSING_CREDS":
                    st.error("❌ 'credentials.json' not found! Please download it from Google Cloud Console.")
                elif status == "SUCCESS":
                    count = 0
                    for email in emails:
                        # Add to Vector DB
                        chroma_store.add_document(
                            collection, 
                            email['text'], 
                            email['meta']
                        )
                        count += 1
                    st.success(f"✅ Successfully indexed {count} emails!")
                    
            except Exception as e:
                st.error(f"Gmail Error: {e}")

    # ... continue with "Add Documents" section ...
    st.subheader("API Setup")
    
    # 1. Try to load from secrets.toml first
    secret_key = st.secrets.get("GEMINI_API_KEY")
    
    if secret_key:
        st.success("✅ API Key loaded from secrets.toml")
        active_key = secret_key
    else:
        # Fallback to manual input if secrets.toml is missing or empty
        api_key_input = st.text_input("Gemini API Key", type="password", help="Get one at aistudio.google.com")
        active_key = api_key_input

    if not active_key:
        st.warning("⚠️ Please provide a Gemini API Key to proceed.")

    st.markdown("---")
    st.subheader("Settings")
    
    # 2. UPDATED MODEL LIST based on your check_models.py results
    gemini_model = st.selectbox(
        "Model", 
        [
            "gemini-2.5-flash",       # Fast & Efficient (Default)
            "gemini-2.5-pro",         # High Intelligence
            "gemini-2.0-flash",       # Stable Backup
            "gemini-3-pro-preview",   # Experimental / Most Powerful
        ]
    )
    
    top_k = st.slider("Top-K Context Chunks", 1, 10, 3)
    
    st.markdown("---")
    st.subheader("Add Documents")
    uploads = st.file_uploader(
    "Upload Knowledge (Docs, Code, Images, Excel)",
    type=[
        "pdf", "docx", "pptx", "xlsx", "xls",  # Documents
        "txt", "md", "log", "json",            # Text
        "py", "c", "h", "js", "html",          # Code
        "jpg", "jpeg", "png"                   # Images
    ],
    accept_multiple_files=True,
    )
    ingest_btn = st.button("Ingest Docs", use_container_width=True)
    clear_btn = st.button("Clear Database", type="secondary", use_container_width=True)

# ---------------------------
# Logic: DB & Ingestion
# ---------------------------
if active_key:
    try:
        llm_gemini.configure_gemini(active_key)
    except Exception as e:
        st.error(f"API Configuration failed: {e}")

# Initialize Vector DB
with st.spinner("Loading Vector Database..."):
    try:
        model, embed_fn = chroma_store.get_embedder(EMBED_MODEL)
        client = chroma_store.get_chroma_client(PERSIST_DIR)
        collection = chroma_store.ensure_collection(client, embed_fn)
    except Exception as e:
        st.error(f"Database Error: {e}")
        st.stop()

if clear_btn:
    try:
        client.delete_collection(COLLECTION_NAME)
        collection = chroma_store.ensure_collection(client, embed_fn)
        st.success("Knowledge base cleared.")
    except Exception as e:
        st.error(f"Error clearing DB: {e}")

if ingest_btn and uploads:
    total_chunks = 0
    progress_bar = st.progress(0)
    for idx, up in enumerate(uploads):
        text, fname = file_readers.extract_text_from_upload(up)
        meta = {"source": fname}
        count = chroma_store.add_document(collection, text, meta)
        total_chunks += count
        progress_bar.progress((idx + 1) / len(uploads))
    st.success(f"Ingested {len(uploads)} files ({total_chunks} chunks).")

# ---------------------------
# Chat Interface
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask Gemini about your documents..."):
    if not active_key:
        st.error("Please enter a Gemini API Key in the sidebar or secrets.toml.")
        st.stop()

    # User Message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Retrieve (Local Vectors)
            hits = chroma_store.retrieve(collection, question, top_k=top_k)
            
            if not hits or all(h[0] == [] for h in hits):
                st.warning("No relevant context found in documents.")
                answer = "I couldn't find any relevant information in your uploaded documents."
            else:
                # 2. Build Context (using Gemini to summarize chunks if needed)
                sources_block = []
                total_chars = 0
                from core.config import MAX_TOTAL_CONTEXT_CHARS
                
                for i, (doc, meta, dist, _id) in enumerate(hits, start=1):
                    title = meta.get("source", "Doc")
                    # Use Gemini to summarize long chunks
                    chunk_sum = llm_gemini.summarize_chunk(doc, api_key=active_key)
                    
                    if total_chars + len(chunk_sum) > MAX_TOTAL_CONTEXT_CHARS:
                        break
                    
                    sources_block.append(f"[S{i}] {title}\n{chunk_sum}")
                    total_chars += len(chunk_sum)

                context_text = "\n\n".join(sources_block)
                
                system_msg = "You are a helpful expert. Answer based strictly on the context below."
                user_msg = f"Context:\n{context_text}\n\nQuestion: {question}"
                
                # 3. Generate
                answer = llm_gemini.call_gemini(system_msg, user_msg, model_name=gemini_model)
                
                st.markdown(answer)
                
                # Show Sources
                with st.expander("View Sources"):
                    for i, (doc, meta, dist, _) in enumerate(hits, start=1):
                        st.caption(f"Source {i}: {meta.get('source')} (Dist: {dist:.3f})")
                        st.text(doc[:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})