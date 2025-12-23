import streamlit as st
import base64
from core.config import APP_TITLE, EMBED_MODEL, PERSIST_DIR, COLLECTION_NAME, GROQ_MODEL
from core import file_readers, chroma_store, prompt_builder, llm_groq
from core.ui import render_header

# Clear chat if logo clicked
if st.query_params.get("clear_chat"):
    st.session_state.messages = []
    st.query_params.clear()

st.set_page_config(
    page_title="SyBot",
    layout="wide"
)

# ---------------------------
# Streamlit UI Setup
# ---------------------------

# ---------- LOGO ----------
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_image("assets/logo.png")

st.markdown(
    f"""
    <style>
        .sybot-logo {{
            position: fixed;
            top: 12px;
            left: 12px;
            width: 56px;
            z-index: 1000;
            cursor: pointer;
        }}
    </style>

    <a href="?clear_chat=true">
        <img class="sybot-logo" src="data:image/png;base64,{logo_base64}">
    </a>
    """,
    unsafe_allow_html=True
)

render_header()
# ---------------------------
# Sidebar & Settings
# ---------------------------
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_image("assets/logo.png")
with st.sidebar:

# ---------- SIDEBAR FIXED LOGO ----------
    st.markdown(
        f"""
        <style>
            /* Fix sidebar logo at top */
            [data-testid="stSidebar"] {{
                padding-top: 90px;
            }}

            .sidebar-logo {{
                position: fixed;
                top: 12px;
                left: 16px;
                width: 80px;
                z-index: 200;
                background-color: white;
                padding: 6px 10px;
                border-radius: 8px;
            }}
        </style>

        <div class="sidebar-logo">
            <img src="data:image/png;base64,{logo_base64}" width="140">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.subheader("API Setup")
    
    # 1. Try to load from secrets.toml first
    secret_key = st.secrets.get("GROQ_API_KEY")
    
    if secret_key:
        st.success("✅ API Key loaded")
        active_key = secret_key
    else:
        api_key_input = st.text_input("Groq API Key", type="password", help="Get one at console.groq.com")
        active_key = api_key_input

    if not active_key:
        st.warning("⚠️ Please provide a Groq API Key.")

    st.markdown("---")
    st.subheader("Settings")
    
    # Model Selection
    selected_model = st.selectbox(
        "Model", 
        [
            "llama-3.3-70b-versatile",   # Newest, smartest
            "llama-3.1-70b-versatile",   # Previous stable
            "llama-3.1-8b-instant",      # Super fast
            "mixtral-8x7b-32768"         # Large context
        ],
        index=0
    )
    
    top_k = st.slider("Top-K Context Chunks", 1, 10, 3)

    st.markdown("---")
    st.subheader("Data Connectors")
    
    # --- GMAIL INTEGRATION ---
    if st.button("📥 Import Recent Emails (Gmail)"):
        # We need to initialize the DB client first to add documents
        try:
            _, embed_fn_temp = chroma_store.get_embedder(EMBED_MODEL)
            client_temp = chroma_store.get_chroma_client(PERSIST_DIR)
            collection_temp = chroma_store.ensure_collection(client_temp, embed_fn_temp)
            
            with st.spinner("Connecting to Gmail... (Check browser popup)"):
                # Import here to avoid startup errors if libraries are missing
                from core import gmail_connector
                
                status, emails = gmail_connector.fetch_recent_emails(max_count=20)
                
                if status == "MISSING_CREDS":
                    st.error("❌ 'credentials.json' missing! Download it from Google Cloud Console.")
                elif status == "SUCCESS":
                    count = 0
                    for email in emails:
                        chroma_store.add_document(collection_temp, email['text'], email['meta'])
                        count += 1
                    st.success(f"✅ Indexed {count} emails!")
                else:
                    st.error("Failed to fetch emails.")
                    
        except ImportError:
            st.error("Google libraries missing. Run: pip install google-auth-oauthlib google-api-python-client")
        except Exception as e:
            st.error(f"Gmail Error: {e}")
            
    st.markdown("---")
    st.subheader("Add Documents")
    
    # File Uploader (All Formats)
    uploads = st.file_uploader(
        "Upload Knowledge (Docs, Code, Images, Excel)",
        type=[
            "pdf", "docx", "pptx", "xlsx", "xls",  # Office Docs
            "txt", "md", "log", "json",            # Text
            "py", "c", "h", "js", "html",          # Code
            "jpg", "jpeg", "png"                   # Images (OCR)
        ],
        accept_multiple_files=True,
    )
    
    ingest_btn = st.button("Ingest Docs", use_container_width=True)
    clear_btn = st.button("Clear Database", type="secondary", use_container_width=True)

# ---------------------------
# Logic: DB & Ingestion
# ---------------------------
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

if question := st.chat_input("Ask SyBot about your documents or emails..."):
    if not active_key:
        st.error("Please enter a Groq API Key in the sidebar.")
        st.stop()

    # User Message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("SyBot is thinking..."):
            # 1. Retrieve
            hits = chroma_store.retrieve(collection, question, top_k=top_k)
            
            if not hits or all(h[0] == [] for h in hits):
                st.warning("No relevant context found.")
                answer = "I couldn't find any relevant information in your documents or emails."
            else:
                # 2. Build Prompt (using Groq for summarization)
                sources_block = []
                total_chars = 0
                from core.config import MAX_TOTAL_CONTEXT_CHARS
                
                for i, (doc, meta, dist, _id) in enumerate(hits, start=1):
                    title = meta.get("source", "Doc")
                    # Summarize
                    chunk_sum = llm_groq.summarize_chunk(doc, api_key=active_key)
                    
                    if total_chars + len(chunk_sum) > MAX_TOTAL_CONTEXT_CHARS:
                        break
                    
                    sources_block.append(f"[S{i}] {title}\n{chunk_sum}")
                    total_chars += len(chunk_sum)

                context_text = "\n\n".join(sources_block)
                
                system_msg = "You are a helpful expert. Answer strictly based on the context provided."
                user_msg = f"Context:\n{context_text}\n\nQuestion: {question}"
                
                # 3. Generate
                answer = llm_groq.call_groq(system_msg, user_msg, model=selected_model, api_key=active_key)
                
                st.markdown(answer)
                
                # Show Sources
                with st.expander("View Sources"):
                    for i, (doc, meta, dist, _) in enumerate(hits, start=1):
                        st.caption(f"Source {i}: {meta.get('source')} (Dist: {dist:.3f})")
                        st.text(doc[:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})