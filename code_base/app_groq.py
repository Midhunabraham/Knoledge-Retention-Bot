import streamlit as st
import base64
import hmac
import uuid
import json
import os
import io
import re
import pandas as pd
import duckdb
from datetime import datetime
from core.config import APP_TITLE, EMBED_MODEL, PERSIST_DIR, COLLECTION_NAME, GROQ_MODEL
from core import file_readers, chroma_store, prompt_builder, llm_groq
from core.ui import render_header

# ── Groq free tier hard limits ──
# 12,000 TPM ≈ 48,000 chars total per request (system + user + context).
# We reserve ~8,000 chars for system prompt + question + answer headroom.
# Data context budget: 20,000 chars max.
GROQ_MAX_CONTEXT_CHARS = 20000

# Rows shown per page when browsing Excel results
EXCEL_PAGE_SIZE = 30

# Define paths for temporary storage
TEMP_DB_DIR = "temp_storage"
TEMP_SESSIONS_FILE = "temp_sessions.json"
os.makedirs(TEMP_DB_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Password / Auth
# ------------------------------------------------------------------
def check_password():
    def password_entered():
        if "password" in st.session_state:
            if hmac.compare_digest(st.session_state["password"], st.secrets["passwords"]["office_access"]):
                st.session_state["password_correct"] = True
                st.session_state["login_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                del st.session_state["password"]
                if "session_id" not in st.session_state:
                    st.session_state.session_id = str(uuid.uuid4())
                if "ingestion_mode" not in st.session_state:
                    st.session_state.ingestion_mode = "permanent"
            else:
                st.session_state["password_correct"] = False
                st.session_state["login_attempts"] = st.session_state.get("login_attempts", 0) + 1

    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "ingestion_mode" not in st.session_state:
        st.session_state.ingestion_mode = "permanent"

    if st.session_state.get("password_correct", False):
        return True

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #4CAF50; margin-bottom: 10px;">🔒 SyBot</h1>
            <p style="color: #666; font-size: 16px;">Secure Knowledge Management System</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>Secure Login</h2>", unsafe_allow_html=True)

        # ── Use st.form so pressing Enter submits the login ──
        with st.form("login_form", clear_on_submit=True):
            st.text_input("Enter Access Password", type="password", key="password",
                          help="Contact administrator if you don't have the password")
            login_col1, login_col2, login_col3 = st.columns([1, 2, 1])
            with login_col2:
                submitted = st.form_submit_button(
                    "🚀 Login to SyBot",
                    use_container_width=True,
                    type="primary"
                )
            if submitted:
                password_entered()

        if st.session_state.login_attempts > 0:
            st.warning(f"❌ Incorrect password. Attempt {st.session_state.login_attempts}")
        st.markdown("""
        <div style="text-align: center; margin-top: 40px; color: #888; font-size: 14px;">
            <p>© 2026 SyBot Knowledge System</p>
            <p style="font-size: 12px;">v1.0 | Secure Access Required</p>
        </div>
        """, unsafe_allow_html=True)
    return False


# ------------------------------------------------------------------
# Temp Session Helpers
# ------------------------------------------------------------------
def get_temp_collection(session_id, embed_fn):
    temp_persist_dir = os.path.join(TEMP_DB_DIR, session_id)
    client = chroma_store.get_chroma_client(temp_persist_dir)
    collection_name = f"temp_collection_{session_id[:8]}"
    try:
        collection = client.get_collection(collection_name, embedding_function=embed_fn)
    except:
        collection = client.create_collection(name=collection_name, embedding_function=embed_fn)
    return client, collection

def save_temp_session(session_id, files_ingested):
    try:
        sessions = {}
        if os.path.exists(TEMP_SESSIONS_FILE):
            with open(TEMP_SESSIONS_FILE, 'r') as f:
                sessions = json.load(f)
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
    try:
        if os.path.exists(TEMP_SESSIONS_FILE):
            with open(TEMP_SESSIONS_FILE, 'r') as f:
                sessions = json.load(f)
            return sessions.get(session_id, {})
    except:
        pass
    return {}

def cleanup_old_sessions(max_age_hours=24):
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
                temp_dir = os.path.join(TEMP_DB_DIR, session_id)
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
        with open(TEMP_SESSIONS_FILE, 'w') as f:
            json.dump(sessions_to_keep, f, indent=2)
    except Exception as e:
        print(f"Error cleaning up sessions: {e}")


# ------------------------------------------------------------------
# Excel Helpers
# ------------------------------------------------------------------
EXCEL_EXTENSIONS = ['.xlsx', '.xls']

def is_excel_source(meta):
    return any(meta.get("source", "").lower().endswith(ext) for ext in EXCEL_EXTENSIONS)

def get_all_excel_chunks(collection, excel_source):
    try:
        res = collection.get(where={"source": excel_source}, include=["documents", "metadatas"])
        return [(doc, meta, 0.0, "") for doc, meta in zip(res["documents"], res["metadatas"])]
    except Exception as e:
        st.warning(f"Could not retrieve all Excel chunks: {e}")
        return []

def detect_pagination(query: str):
    """
    Returns ('next'|'prev'|'page'|'first'|None, count_or_page_number_or_None)
    """
    q = query.lower().strip()
    if re.search(r'\b(next|more|continue|show more|load more)\b', q):
        m = re.search(r'next\s+(\d+)', q)
        return 'next', int(m.group(1)) if m else EXCEL_PAGE_SIZE
    if re.search(r'\b(prev|previous|back|before)\b', q):
        return 'prev', EXCEL_PAGE_SIZE
    m = re.search(r'\bpage\s+(\d+)\b', q)
    if m:
        return 'page', int(m.group(1))
    if re.search(r'\b(first|start|beginning|restart|reset)\b', q):
        return 'first', None
    return None, None

def df_page_to_context(df_page: pd.DataFrame, sheet: str,
                        offset: int, total_matched: int,
                        total_file: int, page_size: int) -> str:
    """Format one page of results as a context string for the LLM."""
    end = min(offset + page_size, total_matched)
    total_pages = max(1, -(-total_matched // page_size))   # ceiling div
    header = (
        f"--- Sheet: {sheet} ---\n"
        f"Total rows in file: {total_file} | "
        f"Rows matching query: {total_matched} | "
        f"Showing rows {offset + 1}–{end} "
        f"(Page {offset // page_size + 1} of {total_pages})\n"
        f"Columns: {', '.join(df_page.columns.tolist())}\n"
        f"{'(More rows available — say next to see more)' if end < total_matched else '(All matched rows shown)'}\n"
        f"--- Data ---\n"
    )
    data_str = df_page.to_string(index=False, max_rows=None, max_cols=None)
    combined = header + data_str
    # Hard cap to stay within Groq limit
    if len(combined) > GROQ_MAX_CONTEXT_CHARS:
        combined = combined[:GROQ_MAX_CONTEXT_CHARS]
    return combined


# ------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------
cleanup_old_sessions()

if not check_password():
    st.stop()

if "login_time" in st.session_state and not st.session_state.get("login_shown", False):
    st.success(f"✅ Login successful! Welcome to SyBot. Logged in at: {st.session_state.login_time}")
    st.session_state.login_shown = True
    st.rerun()

st.set_page_config(page_title="SyBot", layout="wide")

if st.query_params.get("clear_chat"):
    st.session_state.messages = []
    st.query_params.clear()

def get_base64_image(path):
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, path)
    with open(full_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_image("assets/logo.png")

def render_header():
    st.markdown("<h1 style='text-align:center'>Welcome to SyBot</h1>", unsafe_allow_html=True)

render_header()

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="position:fixed; top:12px; left:12px; display:flex; align-items:center; gap:8px;">
            <a href="?clear_chat=true">
                <img src="data:image/png;base64,{logo_base64}" width="60">
            </a>
            <a href="?clear_chat=true">
                <button style="padding:6px 10px;border-radius:6px;border:none;
                    background-color:#4CAF50;color:white;cursor:pointer;font-weight:bold;">
                    New Chat
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Help ❓"):
        st.markdown("""
        **Welcome to SyBot Help!**

        **Capabilities:**
        - Upload and process documents: PDF, DOCX, PPTX, XLSX, TXT, CSV, Images (OCR supported)
        - Answer questions based on uploaded documents
        - Paginate through Excel results — say **"next"** or **"next 30"** to browse

        **Ingestion Modes:**
        - **Permanent**: Store documents in the main database
        - **Temporary**: Session-only storage, cleaned up after 24 hours

        **How to use:**
        1. Select ingestion mode
        2. Upload your document
        3. Ask your question
        4. Say **"next"** to see more rows from Excel files
        """)

    st.subheader("API Setup")
    try:
        secret_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        secret_key = None
    if secret_key:
        st.success("✅ API Key loaded")
        active_key = secret_key
    else:
        api_key_input = st.text_input("Groq API Key", type="password", help="Get one at console.groq.com")
        active_key = api_key_input

    if not active_key:
        st.warning("⚠️ Please provide a Groq API Key.")

    st.markdown("---")
    st.subheader("Ingestion Mode")
    ingestion_mode = st.radio(
        "Select ingestion mode:",
        ["permanent", "temporary"],
        format_func=lambda x: "📁 Permanent Storage" if x == "permanent" else "⏰ Temporary Session",
        index=0 if st.session_state.get("ingestion_mode", "permanent") == "permanent" else 1,
        key="ingestion_mode_selector"
    )
    st.session_state.ingestion_mode = ingestion_mode

    if ingestion_mode == "temporary":
        if not st.session_state.session_id:
            st.session_state.session_id = str(uuid.uuid4())
        st.info(f"""
        **Temporary Session Mode**
        - Session ID: `{st.session_state.session_id[:8]}...`
        - Auto-cleaned after 24 hours
        """)
        session_info = get_temp_session_info(st.session_state.session_id)
        if session_info:
            created_time = datetime.fromisoformat(session_info["created_at"]).strftime("%Y-%m-%d %H:%M")
            st.caption(f"Session created: {created_time}")
            if session_info.get("files_ingested"):
                st.caption(f"Files ingested: {session_info['files_ingested']}")
    else:
        st.success("**Permanent Mode**: Stored in main database")

    st.markdown("---")
    st.subheader("Settings")
    selected_model = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ],
        index=0
    )
    top_k = st.slider("Top-K Context Chunks", 1, 20, 10)

    st.markdown("---")
    st.subheader("Data Connectors")
    if st.button("📥 Import Recent Emails (Gmail)"):
        try:
            _, embed_fn_temp = chroma_store.get_embedder(EMBED_MODEL)
            if st.session_state.ingestion_mode == "temporary":
                client_temp, collection_temp = get_temp_collection(st.session_state.session_id, embed_fn_temp)
            else:
                client_temp = chroma_store.get_chroma_client(PERSIST_DIR)
                collection_temp = chroma_store.ensure_collection(client_temp, embed_fn_temp)
            with st.spinner("Connecting to Gmail..."):
                from core import gmail_connector
                status, emails = gmail_connector.fetch_recent_emails(max_count=20)
                if status == "MISSING_CREDS":
                    st.error("❌ 'credentials.json' missing!")
                elif status == "SUCCESS":
                    count = 0
                    for email in emails:
                        chroma_store.add_document(collection_temp, email['text'], email['meta'])
                        count += 1
                    if st.session_state.ingestion_mode == "temporary":
                        session_info = get_temp_session_info(st.session_state.session_id)
                        save_temp_session(st.session_state.session_id,
                                          session_info.get("files_ingested", 0) + count)
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
    uploads = st.file_uploader(
        "Upload Knowledge (Docs, Code, Images, Excel)",
        type=["pdf", "docx", "pptx", "xlsx", "xls", "txt", "md", "log", "json",
              "py", "c", "h", "js", "html", "jpg", "jpeg", "png", "eml"],
        accept_multiple_files=True,
    )
    ingest_label = "Ingest to Temporary Session" if st.session_state.ingestion_mode == "temporary" else "Ingest to Permanent DB"
    ingest_btn = st.button(ingest_label, use_container_width=True)
    if st.session_state.ingestion_mode == "temporary":
        clear_btn = st.button("Clear Temporary Session", type="secondary", use_container_width=True)
    else:
        clear_btn = st.button("Clear Database", type="secondary", use_container_width=True)


# ------------------------------------------------------------------
# DB Init
# ------------------------------------------------------------------
with st.spinner("Loading Vector Database..."):
    try:
        model, embed_fn = chroma_store.get_embedder(EMBED_MODEL)
        if st.session_state.ingestion_mode == "temporary":
            if not st.session_state.session_id:
                st.session_state.session_id = str(uuid.uuid4())
            client, collection = get_temp_collection(st.session_state.session_id, embed_fn)
            st.info(f"📝 Using temporary session storage (ID: {st.session_state.session_id[:8]}...)")
        else:
            client = chroma_store.get_chroma_client(PERSIST_DIR)
            collection = chroma_store.ensure_collection(client, embed_fn)
            st.success("✅ Using permanent database storage")
    except Exception as e:
        st.error(f"Database Error: {e}")
        st.stop()

if clear_btn:
    try:
        if st.session_state.ingestion_mode == "temporary":
            temp_persist_dir = os.path.join(TEMP_DB_DIR, st.session_state.session_id)
            if os.path.exists(temp_persist_dir):
                import shutil
                shutil.rmtree(temp_persist_dir)
                st.success("Temporary session cleared.")
                client, collection = get_temp_collection(st.session_state.session_id, embed_fn)
        else:
            client.delete_collection(COLLECTION_NAME)
            collection = chroma_store.ensure_collection(client, embed_fn)
            st.success("Knowledge base cleared.")
    except Exception as e:
        st.error(f"Error clearing DB: {e}")

# Session state for Excel store + pagination
if "excel_file_store" not in st.session_state:
    st.session_state.excel_file_store = {}       # {filename: raw_bytes}
if "excel_filtered_df" not in st.session_state:
    st.session_state.excel_filtered_df = {}      # {filename: DataFrame of ALL matched rows}
if "excel_page_offset" not in st.session_state:
    st.session_state.excel_page_offset = {}      # {filename: int current offset}
if "excel_last_query" not in st.session_state:
    st.session_state.excel_last_query = {}       # {filename: str last filter query}
if "excel_df_cache" not in st.session_state:
    st.session_state.excel_df_cache = {}         # {filename: {"sheet_name": str, "df": DataFrame}}
if "excel_duckdb_con" not in st.session_state:
    st.session_state.excel_duckdb_con = {}       # {filename: duckdb.Connection}

if ingest_btn and uploads:
    total_chunks = 0
    progress_bar = st.progress(0)
    mode_text = "temporary session" if st.session_state.ingestion_mode == "temporary" else "permanent database"
    st.info(f"Ingesting {len(uploads)} file(s) to {mode_text}...")
    # Check which files are already ingested (avoid re-processing on re-click)
    try:
        existing_sources = {
            m.get("source") for m in collection.get(include=["metadatas"])["metadatas"]
        }
    except Exception:
        existing_sources = set()

    skipped = 0
    for idx, up in enumerate(uploads):
        fname_check = up.name
        if fname_check in existing_sources and not any(
            fname_check.lower().endswith(ext) for ext in EXCEL_EXTENSIONS
        ):
            st.caption(f"⏭️ Skipped `{fname_check}` — already in database.")
            skipped += 1
            progress_bar.progress((idx + 1) / len(uploads))
            continue

        text, fname, raw_bytes = file_readers.extract_text_from_upload(up)
        meta = {"source": fname, "ingestion_mode": st.session_state.ingestion_mode}
        if any(fname.lower().endswith(ext) for ext in EXCEL_EXTENSIONS):
            st.session_state.excel_file_store[fname] = raw_bytes
            st.caption(f"📊 Stored Excel `{fname}` for smart filtering")
        if st.session_state.ingestion_mode == "temporary":
            meta["session_id"] = st.session_state.session_id
            meta["ingestion_time"] = datetime.now().isoformat()
        count = chroma_store.add_document(collection, text, meta)
        total_chunks += count
        progress_bar.progress((idx + 1) / len(uploads))

    msg = f"Ingested {len(uploads) - skipped} files ({total_chunks} chunks) to {mode_text}."
    if skipped:
        msg += f" {skipped} file(s) skipped (already ingested)."
    st.success(msg)


# ------------------------------------------------------------------
# Chat Interface
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask SyBot about your documents or emails...")

if question:
    if not active_key:
        st.error("Please enter a Groq API Key in the sidebar.")
        st.stop()

    # ── Debug command ──
    if question.strip().lower().startswith("debug excel"):
        for fname, raw_bytes in st.session_state.get("excel_file_store", {}).items():
            df = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=None, keep_default_na=False)
            for sheet, data in df.items():
                st.write(f"**File:** {fname} | **Sheet:** {sheet}")
                st.write(f"**Columns:** {data.columns.tolist()}")
                for col in data.columns:
                    if any(k in col.lower() for k in ["cpu", "flash", "ram", "uart", "freq", "i2c"]):
                        st.write(f"**`{col}`:** {data[col].dropna().unique()[:10].tolist()}")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("SyBot is thinking..."):

            # ── 0. Early pagination check — skip ChromaDB for next/prev/page ──
            # If the user says "next", "prev", "page N" and we have a cached Excel
            # result, handle it immediately without a new similarity search.
            early_page_intent, early_page_arg = detect_pagination(question)
            early_excel_source = next(iter(st.session_state.excel_filtered_df), None)

            if early_page_intent and early_excel_source and                st.session_state.excel_filtered_df.get(early_excel_source) is not None:
                # Synthetic hits so the rest of the flow works unchanged
                hits        = [("", {"source": early_excel_source, "ingestion_mode": "permanent"}, 0.0, "")]
                excel_hits  = [(h[0], h[1], h[2], h[3]) for h in hits]
                raw_bytes   = st.session_state.get("excel_file_store", {}).get(early_excel_source, b"")
                excel_source = early_excel_source
            else:
                # ── 1. Similarity search ──
                # Retrieve more chunks when multiple files are ingested
                # Excel filtering uses DuckDB — ChromaDB only needed for docs
                effective_top_k = max(top_k, 15)
                hits = chroma_store.retrieve(collection, question, top_k=effective_top_k)
                excel_hits  = []
                excel_source = None
                raw_bytes   = b""

            # ── Special: list all ingested documents ──
            LIST_TRIGGERS = ["what documents", "which documents", "what files",
                             "which files", "list documents", "list files",
                             "what do you have", "what have you", "show documents",
                             "show files", "uploaded files", "ingested"]
            if any(t in question.lower() for t in LIST_TRIGGERS):
                try:
                    all_meta    = collection.get(include=["metadatas"])["metadatas"]
                    all_sources = sorted({m.get("source", "unknown") for m in all_meta})
                    excel_files = list(st.session_state.get("excel_file_store", {}).keys())
                    # Add Excel files not yet in ChromaDB
                    for xf in excel_files:
                        if xf not in all_sources:
                            all_sources.append(xf)
                    if all_sources:
                        lines = ["I have the following documents in my knowledge base:\n"]
                        for i, src in enumerate(sorted(all_sources), 1):
                            ext  = src.rsplit(".", 1)[-1].upper() if "." in src else "FILE"
                            icon = {"XLSX":"📊","XLS":"📊","PDF":"📄","DOCX":"📝",
                                    "TXT":"📃","EML":"📧","PPTX":"📑","CSV":"📊"}.get(ext, "📁")
                            lines.append(f"{icon} {i}. {src}")
                        answer = "\n".join(lines)
                    else:
                        answer = "No documents have been ingested yet. Please upload files using the sidebar."
                except Exception as e:
                    answer = f"Could not retrieve document list: {e}"
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.stop()

            # ==============================================================
            # UNIVERSAL PROCESSING ENGINE
            # Strategy:
            #   1. Always search ChromaDB across ALL documents (high top_k)
            #   2. If Excel file exists → always run DuckDB filter in parallel
            #   3. Combine ALL context (Excel rows + doc chunks) into one prompt
            #   4. Let the LLM synthesize the answer from everything available
            #   5. If nothing found anywhere → fall back to general knowledge
            # User never needs to know or specify which file has what.
            # ==============================================================
            answer       = None
            context_text = ""
            system_msg   = ""
            stored_excel = st.session_state.get("excel_file_store", {})

            # ── Collect all context parts ──
            context_parts = []   # list of (label, text) tuples

            # ──────────────────────────────────────────────
            # PART A: Excel DuckDB filter (if any Excel exists)
            # ──────────────────────────────────────────────
            if stored_excel:
                for xls_fname, xls_bytes in stored_excel.items():
                    try:
                        MAX_EXCEL_ROWS = 100_000

                        # Load & cache
                        if xls_fname not in st.session_state.excel_df_cache:
                            with st.spinner(f"📂 Loading `{xls_fname}`..."):
                                dfs        = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=None,
                                                           dtype=str, keep_default_na=False)
                                sname      = list(dfs.keys())[0]
                                sdf        = dfs[sname].fillna("").reset_index(drop=True)
                                if len(sdf) > MAX_EXCEL_ROWS:
                                    sdf = sdf.head(MAX_EXCEL_ROWS)
                                st.session_state.excel_df_cache[xls_fname] = {
                                    "sheet_name": sname, "df": sdf
                                }
                                con         = duckdb.connect()
                                df_duck     = sdf.copy()
                                for col in df_duck.columns:
                                    try:
                                        num = pd.to_numeric(df_duck[col], errors="coerce")
                                        if num.notna().sum() > len(df_duck) * 0.5:
                                            df_duck[col] = num
                                    except Exception:
                                        pass
                                con.register("excel_table", df_duck)
                                st.session_state.excel_duckdb_con[xls_fname] = con
                        else:
                            sdf   = st.session_state.excel_df_cache[xls_fname]["df"]
                            sname = st.session_state.excel_df_cache[xls_fname]["sheet_name"]
                            con   = st.session_state.excel_duckdb_con.get(xls_fname)
                            if con is None:
                                con     = duckdb.connect()
                                df_duck = sdf.copy()
                                for col in df_duck.columns:
                                    try:
                                        num = pd.to_numeric(df_duck[col], errors="coerce")
                                        if num.notna().sum() > len(df_duck) * 0.5:
                                            df_duck[col] = num
                                    except Exception:
                                        pass
                                con.register("excel_table", df_duck)
                                st.session_state.excel_duckdb_con[xls_fname] = con

                        # Pagination or new filter?
                        page_intent, page_arg = detect_pagination(question)
                        cached_df             = st.session_state.excel_filtered_df.get(xls_fname)

                        if page_intent and cached_df is not None:
                            filtered_df   = cached_df
                            total_matched = len(filtered_df)
                            current_off   = st.session_state.excel_page_offset.get(xls_fname, 0)
                            if page_intent == "next":
                                step        = page_arg if page_arg else EXCEL_PAGE_SIZE
                                last_start  = max(0, ((total_matched - 1) // EXCEL_PAGE_SIZE) * EXCEL_PAGE_SIZE)
                                new_off     = min(current_off + step, last_start)
                            elif page_intent == "prev":
                                new_off = max(0, current_off - EXCEL_PAGE_SIZE)
                            elif page_intent == "page":
                                new_off = min((page_arg - 1) * EXCEL_PAGE_SIZE, max(0, total_matched - 1))
                            else:
                                new_off = 0
                            st.session_state.excel_page_offset[xls_fname] = new_off
                        else:
                            # NL → SQL
                            columns     = sdf.columns.tolist()
                            sample_vals = {
                                col: sdf[col].replace("", None).dropna().unique()[:5].tolist()
                                for col in columns
                            }
                            schema_detail = "\n".join(
                                f'  "{c}": e.g. {sample_vals.get(c, [])}' for c in columns
                            )
                            sql_system = f"""You are a DuckDB SQL expert.
Table name: excel_table  (from file: {xls_fname})
Columns and sample values:
{schema_detail}

STRICT RULES:
- Return ONLY valid DuckDB SQL. No explanation, no markdown, no backticks.
- Wrap column names with spaces in double quotes: "Column Name"
- Text filters: LOWER("Col") LIKE '%value%'
- Numeric filters: "Col" > 100  (never use CAST — columns are pre-cast)
- Always use WHERE for specific filters. No LIMIT — pagination handles it.
- If query is general/list all: SELECT * FROM excel_table ORDER BY 1
- If query is unrelated to this file: SELECT * FROM excel_table LIMIT 0

EXAMPLES:
User: IC with cpu arm cortex-m0+ and flash > 500 and ram > 200
SQL: SELECT * FROM excel_table WHERE LOWER("CPU") LIKE '%m0+%' AND "Flash memory (kByte)" > 500 AND "RAM (kByte)" > 200
User: list all products
SQL: SELECT * FROM excel_table ORDER BY 1
User: what is the capital of France
SQL: SELECT * FROM excel_table LIMIT 0
"""
                            with st.spinner(f"🧠 Querying `{xls_fname}`..."):
                                raw_sql   = llm_groq.call_groq(
                                    sql_system, question,
                                    model=selected_model, api_key=active_key
                                )
                                import re as _re
                                sql_query = _re.sub(r"```(?:sql)?\s*", "", raw_sql, flags=_re.IGNORECASE)
                                sql_query = _re.sub(r"```", "", sql_query).strip()
                                st.caption(f"🔍 `{xls_fname}` SQL: `{sql_query}`")

                            try:
                                filtered_df = con.execute(sql_query).df().fillna("").reset_index(drop=True)
                            except Exception as sql_err:
                                st.warning(f"⚠️ SQL failed on `{xls_fname}`: {sql_err}. Using text search.")
                                mask        = sdf.apply(
                                    lambda row: row.astype(str).str.contains(question, case=False, na=False).any(),
                                    axis=1
                                )
                                filtered_df = sdf[mask].reset_index(drop=True)

                            new_off       = 0
                            total_matched = len(filtered_df)
                            st.session_state.excel_filtered_df[xls_fname]  = filtered_df
                            st.session_state.excel_last_query[xls_fname]   = question
                            st.session_state.excel_page_offset[xls_fname]  = 0

                        # Slice page
                        filtered_df   = st.session_state.excel_filtered_df.get(xls_fname, pd.DataFrame())
                        total_matched = len(filtered_df)

                        if total_matched > 0:
                            new_off = st.session_state.excel_page_offset.get(xls_fname, 0)
                            new_off = min(new_off, max(0, total_matched - EXCEL_PAGE_SIZE))
                            st.session_state.excel_page_offset[xls_fname] = new_off
                            page_df = filtered_df.iloc[new_off: new_off + EXCEL_PAGE_SIZE]
                            end_row = min(new_off + EXCEL_PAGE_SIZE, total_matched)

                            st.info(
                                f"📊 `{xls_fname}`: **{total_matched}** rows matched | "
                                f"Showing **{new_off + 1}–{end_row}**" +
                                (f" | Say **'next'** for more" if end_row < total_matched else " | ✅ All shown")
                            )
                            with st.expander(f"📋 `{xls_fname}` — filtered data", expanded=True):
                                st.dataframe(page_df, use_container_width=True)

                            xls_context = df_page_to_context(
                                page_df, sname,
                                offset=new_off, total_matched=total_matched,
                                total_file=len(sdf), page_size=EXCEL_PAGE_SIZE
                            )
                            context_parts.append((f"📊 Excel [{xls_fname}]", xls_context))

                    except Exception as xls_err:
                        st.warning(f"⚠️ Could not process `{xls_fname}`: {xls_err}")

            # ──────────────────────────────────────────────
            # PART B: ChromaDB RAG (ALL non-Excel documents)
            # ──────────────────────────────────────────────
            # Deduplicate: max 2 chunks per source file to spread context across all docs
            _seen_sources  = {}
            non_excel_hits = []
            for doc, meta, dist, _id in hits:
                if is_excel_source(meta) or not doc:
                    continue
                src = meta.get("source", "")
                _seen_sources[src] = _seen_sources.get(src, 0) + 1
                if _seen_sources[src] <= 2:   # max 2 chunks per file
                    non_excel_hits.append((doc, meta, dist, _id))
            if non_excel_hits:
                # No summarize_chunk — that made 1 Groq call per chunk (15+ calls = huge delay).
                # ChromaDB already returns the most relevant chunks via similarity search.
                # Just use the raw chunk text directly — it's already the right content.
                CHAR_BUDGET   = GROQ_MAX_CONTEXT_CHARS // 2   # share budget with Excel context
                sources_block = []
                total_chars   = 0
                for i, (doc, meta, dist, _id) in enumerate(non_excel_hits, start=1):
                    title          = meta.get("source", "Doc")
                    ingestion_mode = meta.get("ingestion_mode", "permanent")
                    mode_indicator = "⏰" if ingestion_mode == "temporary" else "📁"
                    chunk_content  = doc[:3000]   # cap each chunk at 3000 chars
                    if total_chars + len(chunk_content) > CHAR_BUDGET:
                        st.caption(f"ℹ️ Using top {i-1} of {len(non_excel_hits)} document chunks.")
                        break
                    sources_block.append(f"{mode_indicator} [S{i}] {title}\n{chunk_content}")
                    total_chars += len(chunk_content)
                doc_context = "\n\n".join(sources_block)
                context_parts.append(("📄 Documents [PDF/TXT/DOCX/EML/etc.]", doc_context))

            # ──────────────────────────────────────────────
            # PART C: Assemble final context + generate answer
            # ──────────────────────────────────────────────
            if context_parts:
                # Merge all parts with clear section labels
                merged = []
                for label, text in context_parts:
                    merged.append(f"=== {label} ===\n{text}")
                context_text = "\n\n".join(merged)

                if len(context_text) > GROQ_MAX_CONTEXT_CHARS:
                    context_text = context_text[:GROQ_MAX_CONTEXT_CHARS]
                    st.caption("⚠️ Context capped to fit model token limit.")

                system_msg = (
                    "You are a helpful expert assistant with access to multiple data sources. "
                    "The context below may include filtered Excel/tabular data AND text from documents "
                    "(PDFs, manuals, datasheets, notes, emails, etc.). "
                    "Answer the user's question by synthesizing ALL relevant information from ALL sources. "
                    "Clearly reference which source your answer comes from when helpful. "
                    "For Excel data: summarize matching rows — product names, specs, patterns. "
                    "For documents: answer based on the actual text content. "
                    "If 0 Excel rows matched, say so and suggest relaxing the filter criteria. "
                    "Never say data is missing if it is present in the context."
                )
                user_msg = f"Context:\n{context_text}\n\nQuestion: {question}"
                answer   = llm_groq.call_groq(
                    system_msg, user_msg,
                    model=selected_model, api_key=active_key
                )
                st.markdown(answer)

                # Sources expander
                with st.expander("📂 View Sources"):
                    for label, _ in context_parts:
                        st.caption(label)
                    for i, (doc, meta, dist, _) in enumerate(hits, start=1):
                        if not is_excel_source(meta) and doc:
                            ingestion_mode = meta.get("ingestion_mode", "permanent")
                            mode_text      = "(Temp)" if ingestion_mode == "temporary" else "(Perm)"
                            st.caption(f"  📄 {mode_text} {meta.get('source')} (dist: {dist:.3f})")
                            st.text(doc[:150] + "...")

            else:
                # Nothing found anywhere — general knowledge
                st.caption("ℹ️ No relevant data found. Answering from general knowledge.")
                answer = llm_groq.call_groq(
                    "You are a helpful assistant. Answer the user's question using your general knowledge. "
                    "If the question is about a specific private document or data, politely ask them to upload it.",
                    question,
                    model=selected_model, api_key=active_key
                )
                st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer or ""})