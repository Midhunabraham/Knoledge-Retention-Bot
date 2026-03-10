import streamlit as st
import base64
import hmac
import uuid
import json
import os
from datetime import datetime
from core.config import APP_TITLE, EMBED_MODEL, PERSIST_DIR, COLLECTION_NAME, GROQ_MODEL
from core import file_readers, chroma_store, prompt_builder, llm_groq
from core.ui import render_header

# Define paths for temporary storage
TEMP_DB_DIR = "temp_storage"
TEMP_SESSIONS_FILE = "temp_sessions.json"

# Ensure temp directories exist
os.makedirs(TEMP_DB_DIR, exist_ok=True)

def check_password():
    """Returns `True` if the user had a correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Check if password exists in session state before accessing it
        if "password" in st.session_state:
            if hmac.compare_digest(st.session_state["password"], st.secrets["passwords"]["office_access"]):
                st.session_state["password_correct"] = True
                st.session_state["login_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                del st.session_state["password"]  # Don't keep password in memory
                # Initialize session ID for temporary storage
                if "session_id" not in st.session_state:
                    st.session_state.session_id = str(uuid.uuid4())
                # Initialize ingestion mode (default: permanent)
                if "ingestion_mode" not in st.session_state:
                    st.session_state.ingestion_mode = "permanent"
            else:
                st.session_state["password_correct"] = False
                st.session_state["login_attempts"] = st.session_state.get("login_attempts", 0) + 1

    # Initialize session state variables
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "ingestion_mode" not in st.session_state:
        st.session_state.ingestion_mode = "permanent"

    # 1. If password is correct, return True
    if st.session_state.get("password_correct", False):
        return True

    # ----- ENHANCED LOGIN PAGE UI -----
    # Create a centered login form with better styling
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Logo/Header
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #4CAF50; margin-bottom: 10px;">🔒 SyBot</h1>
            <p style="color: #666; font-size: 16px;">Secure Knowledge Management System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>Secure Login</h2>", unsafe_allow_html=True)
        
        # Password input
        password = st.text_input(
            "Enter Access Password",
            type="password",
            key="password",
            help="Contact administrator if you don't have the password"
        )
        
        # Login button with custom styling
        login_col1, login_col2, login_col3 = st.columns([1, 2, 1])
        with login_col2:
            if st.button("🚀 Login to SyBot", use_container_width=True, type="primary"):
                password_entered()
        
        # Show login attempts warning
        if st.session_state.login_attempts > 0:
            st.warning(f"❌ Incorrect password. Attempt {st.session_state.login_attempts}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # # Help section
        # with st.expander("ℹ️ Need Help?", expanded=False):
        #     st.markdown("""
        #     **Login Instructions:**
        #     - Contact your system administrator to obtain the access password
        #     - Make sure you're authorized to access SyBot
        #     - After login, you can choose between temporary or permanent document storage
            
        #     **Security Features:**
        #     - Password is never stored or logged
        #     - Session-based authentication
        #     - Encrypted document storage
        #     - Automatic session timeout
        #     """)
        
        # Footer
        st.markdown("""
        <div style="text-align: center; margin-top: 40px; color: #888; font-size: 14px;">
            <p>© 2026 SyBot Knowledge System</p>
            <p style="font-size: 12px;">v1.0 | Secure Access Required</p>
        </div>
        """, unsafe_allow_html=True)

    # 3. Handle Invalid Password after button click
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        # Already shown in the warning above
        pass

    # 4. Stop App Execution (Return False)
    return False

# ... rest of your existing functions (get_temp_collection, save_temp_session, etc.) remain the same ...

def get_temp_collection(session_id, embed_fn):
    """Get or create a temporary collection for the session."""
    temp_persist_dir = os.path.join(TEMP_DB_DIR, session_id)
    client = chroma_store.get_chroma_client(temp_persist_dir)
    collection_name = f"temp_collection_{session_id[:8]}"
    
    # Try to get existing collection or create new one
    try:
        collection = client.get_collection(collection_name, embedding_function=embed_fn)
    except:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embed_fn
        )
    
    return client, collection

def save_temp_session(session_id, files_ingested):
    """Save temporary session information."""
    try:
        if os.path.exists(TEMP_SESSIONS_FILE):
            with open(TEMP_SESSIONS_FILE, 'r') as f:
                sessions = json.load(f)
        else:
            sessions = {}
        
        sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "files_ingested": files_ingested,
            "last_accessed": datetime.now().isoformat()
        }
        
        with open(TEMP_SESSIONS_FILE, 'w') as f:
            json.dump(sessions, f, indent=2)
    except Exception as e:
        st.error(f"Error saving session: {e}")

def get_temp_session_info(session_id):
    """Get information about a temporary session."""
    try:
        if os.path.exists(TEMP_SESSIONS_FILE):
            with open(TEMP_SESSIONS_FILE, 'r') as f:
                sessions = json.load(f)
            return sessions.get(session_id, {})
    except:
        pass
    return {}

def cleanup_old_sessions(max_age_hours=24):
    """Clean up old temporary sessions."""
    try:
        if not os.path.exists(TEMP_SESSIONS_FILE):
            return
        
        with open(TEMP_SESSIONS_FILE, 'r') as f:
            sessions = json.load(f)
        
        now = datetime.now()
        sessions_to_keep = {}
        
        for session_id, session_data in sessions.items():
            created_at = datetime.fromisoformat(session_data["created_at"])
            age_hours = (now - created_at).total_seconds() / 3600
            
            if age_hours < max_age_hours:
                sessions_to_keep[session_id] = session_data
            else:
                # Remove the session directory
                temp_dir = os.path.join(TEMP_DB_DIR, session_id)
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
        
        with open(TEMP_SESSIONS_FILE, 'w') as f:
            json.dump(sessions_to_keep, f, indent=2)
            
    except Exception as e:
        print(f"Error cleaning up sessions: {e}")

# ------------------------------------------------------------------
# MAIN APP EXECUTION STARTS HERE
# ------------------------------------------------------------------

# Clean up old sessions on startup
cleanup_old_sessions()

# Check password BEFORE loading anything else
if not check_password():
    st.stop()  # <--- This halts the app here if login fails

# Show login success message once
if "login_time" in st.session_state and not st.session_state.get("login_shown", False):
    st.success(f"✅ Login successful! Welcome to SyBot. Logged in at: {st.session_state.login_time}")
    st.session_state.login_shown = True
    st.rerun()

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(
    page_title="SyBot",
    layout="wide"
)

# Clear chat if logo clicked
if st.query_params.get("clear_chat"):
    st.session_state.messages = []
    st.query_params.clear()

# ---------------------------
# Helper: Load image as base64
# ---------------------------
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_image("assets/logo.png")

# ---------------------------
# Streamlit UI Setup
# ---------------------------

# ---------- HEADER ----------
def render_header():
    st.markdown("<h1 style='text-align:center'>Welcome to SyBot</h1>", unsafe_allow_html=True)

render_header()

# ---------------------------
# Sidebar & Settings
# ---------------------------
with st.sidebar:
    # ---------- SIDEBAR FIXED LOGO + NEW CHAT BUTTON ----------
    st.markdown(
        f"""
        <div style="position:fixed; top:12px; left:12px; display:flex; align-items:center; gap:8px;">
            <a href="?clear_chat=true">
                <img src="data:image/png;base64,{logo_base64}" width="60">
            </a>
            <a href="?clear_chat=true">
                <button style="
                    padding:6px 10px;
                    border-radius:6px;
                    border:none;
                    background-color:#4CAF50;
                    color:white;
                    cursor:pointer;
                    font-weight:bold;
                ">New Chat</button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------- COLLAPSIBLE HELP ----------
    with st.expander("Help ❓"):
        st.markdown(
            """
            **Welcome to SyBot Help!**

            **Capabilities:**
            - Upload and process documents: PDF, DOCX, PPTX, XLSX, TXT, CSV, Images (OCR supported)
            - Answer questions based on uploaded documents
            - Provide summaries and key points
            - Cross-reference information across multiple documents
            - Integrate with email and other systems for document-based queries

            **Ingestion Modes:**
            - **Permanent**: Store documents in the main database for future use
            - **Temporary**: Store documents in a session-specific temporary storage
              (data is automatically cleaned up after 24 hours)

            **How to use:**
            1. Select ingestion mode (temporary or permanent)
            2. Upload your document in the main area.
            3. Ask your question in the input box.
            4. SyBot will analyze the document and provide an answer.
            """
        )

    # ---------- OTHER SIDEBAR ELEMENTS ----------
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
    
    # ---------- INGESTION MODE SELECTION ----------
    st.subheader("Ingestion Mode")
    
    # Toggle between temporary and permanent ingestion
    ingestion_mode = st.radio(
        "Select ingestion mode:",
        ["permanent", "temporary"],
        format_func=lambda x: "📁 Permanent Storage" if x == "permanent" else "⏰ Temporary Session",
        index=0 if st.session_state.get("ingestion_mode", "permanent") == "permanent" else 1,
        key="ingestion_mode_selector"
    )
    
    # Update session state
    st.session_state.ingestion_mode = ingestion_mode
    
    if ingestion_mode == "temporary":
        # Ensure session_id exists
        if not st.session_state.session_id:
            st.session_state.session_id = str(uuid.uuid4())
            
        st.info(f"""
        **Temporary Session Mode**
        
        - Session ID: `{st.session_state.session_id[:8]}...`
        - Data will be available only in this session
        - Automatically cleaned up after 24 hours
        - Not accessible by other users
        """)
        
        # Show session info
        session_info = get_temp_session_info(st.session_state.session_id)
        if session_info:
            created_time = datetime.fromisoformat(session_info["created_at"]).strftime("%Y-%m-%d %H:%M")
            st.caption(f"Session created: {created_time}")
            if session_info.get("files_ingested"):
                st.caption(f"Files ingested: {session_info['files_ingested']}")
    else:
        st.success("**Permanent Mode**: Documents will be stored in main database")
    
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
        try:
            _, embed_fn_temp = chroma_store.get_embedder(EMBED_MODEL)
            
            # Choose collection based on ingestion mode
            if st.session_state.ingestion_mode == "temporary":
                client_temp, collection_temp = get_temp_collection(
                    st.session_state.session_id, 
                    embed_fn_temp
                )
            else:
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
                    
                    # Update session info for temporary mode
                    if st.session_state.ingestion_mode == "temporary":
                        session_info = get_temp_session_info(st.session_state.session_id)
                        files_ingested = session_info.get("files_ingested", 0) + count
                        save_temp_session(st.session_state.session_id, files_ingested)
                    
                    mode_text = "temporary session" if st.session_state.ingestion_mode == "temporary" else "permanent database"
                    st.success(f"✅ Indexed {count} emails to {mode_text}!")
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
            "jpg", "jpeg", "png", "eml"                   # Images (OCR)
        ],
        accept_multiple_files=True,
    )
    
    # Ingest button with mode indicator
    ingest_label = "Ingest to Temporary Session" if st.session_state.ingestion_mode == "temporary" else "Ingest to Permanent DB"
    ingest_btn = st.button(ingest_label, use_container_width=True)
    
    # Clear button - shows appropriate label based on mode
    if st.session_state.ingestion_mode == "temporary":
        clear_btn = st.button("Clear Temporary Session", type="secondary", use_container_width=True)
    else:
        clear_btn = st.button("Clear Database", type="secondary", use_container_width=True)

# ---------------------------
# Logic: DB & Ingestion
# ---------------------------
# Initialize appropriate Vector DB based on mode
with st.spinner("Loading Vector Database..."):
    try:
        model, embed_fn = chroma_store.get_embedder(EMBED_MODEL)
        
        if st.session_state.ingestion_mode == "temporary":
            # Ensure session_id exists
            if not st.session_state.session_id:
                st.session_state.session_id = str(uuid.uuid4())
                
            # Use temporary collection for this session
            client, collection = get_temp_collection(st.session_state.session_id, embed_fn)
            st.info(f"📝 Using temporary session storage (ID: {st.session_state.session_id[:8]}...)")
        else:
            # Use permanent collection
            client = chroma_store.get_chroma_client(PERSIST_DIR)
            collection = chroma_store.ensure_collection(client, embed_fn)
            st.success("✅ Using permanent database storage")
            
    except Exception as e:
        st.error(f"Database Error: {e}")
        st.stop()

# Handle clear button based on mode
if clear_btn:
    try:
        if st.session_state.ingestion_mode == "temporary":
            # Clear temporary collection
            temp_persist_dir = os.path.join(TEMP_DB_DIR, st.session_state.session_id)
            if os.path.exists(temp_persist_dir):
                import shutil
                shutil.rmtree(temp_persist_dir)
                st.success("Temporary session cleared.")
                # Recreate empty collection
                client, collection = get_temp_collection(st.session_state.session_id, embed_fn)
        else:
            # Clear permanent collection
            client.delete_collection(COLLECTION_NAME)
            collection = chroma_store.ensure_collection(client, embed_fn)
            st.success("Knowledge base cleared.")
    except Exception as e:
        st.error(f"Error clearing DB: {e}")

# Handle ingestion based on mode
if ingest_btn and uploads:
    total_chunks = 0
    progress_bar = st.progress(0)
    
    # Show mode indicator
    mode_text = "temporary session" if st.session_state.ingestion_mode == "temporary" else "permanent database"
    st.info(f"Ingesting {len(uploads)} file(s) to {mode_text}...")
    
    for idx, up in enumerate(uploads):
        text, fname = file_readers.extract_text_from_upload(up)
        meta = {"source": fname, "ingestion_mode": st.session_state.ingestion_mode}
        
        if st.session_state.ingestion_mode == "temporary":
            meta["session_id"] = st.session_state.session_id
            meta["ingestion_time"] = datetime.now().isoformat()
        
        count = chroma_store.add_document(collection, text, meta)
        total_chunks += count
        progress_bar.progress((idx + 1) / len(uploads))
    
    # Update session info for temporary mode
    if st.session_state.ingestion_mode == "temporary":
        session_info = get_temp_session_info(st.session_state.session_id)
        files_ingested = session_info.get("files_ingested", 0) + len(uploads)
        save_temp_session(st.session_state.session_id, files_ingested)
    
    st.success(f"Ingested {len(uploads)} files ({total_chunks} chunks) to {mode_text}.")

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
                    # Add ingestion mode indicator to source
                    ingestion_mode = meta.get("ingestion_mode", "permanent")
                    mode_indicator = "⏰" if ingestion_mode == "temporary" else "📁"
                    
                    # Summarize
                    chunk_sum = llm_groq.summarize_chunk(doc, api_key=active_key)
                    
                    if total_chars + len(chunk_sum) > MAX_TOTAL_CONTEXT_CHARS:
                        break
                    
                    sources_block.append(f"{mode_indicator} [S{i}] {title}\n{chunk_sum}")
                    total_chars += len(chunk_sum)

                context_text = "\n\n".join(sources_block)
                
                system_msg = "You are a helpful expert. Answer strictly based on the context provided."
                user_msg = f"Context:\n{context_text}\n\nQuestion: {question}"
                
                # 3. Generate
                answer = llm_groq.call_groq(system_msg, user_msg, model=selected_model, api_key=active_key)
                
                st.markdown(answer)
                
                # Show Sources with mode indicators
                with st.expander("View Sources"):
                    for i, (doc, meta, dist, _) in enumerate(hits, start=1):
                        ingestion_mode = meta.get("ingestion_mode", "permanent")
                        mode_text = "(Temporary)" if ingestion_mode == "temporary" else "(Permanent)"
                        st.caption(f"Source {i} {mode_text}: {meta.get('source')} (Dist: {dist:.3f})")
                        st.text(doc[:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})