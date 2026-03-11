# Config constants
APP_TITLE = "Knowledge Retention & Training Bot"
PERSIST_DIR = ".chroma"
COLLECTION_NAME = "company_knowledge"
DEFAULT_TOP_K = 4

MAX_CHARS_PER_CHUNK = 500000

# ✅ FIX 1: Raised from 40000 → 800000 so full Excel filter results reach the LLM
# Groq's llama-3.3-70b supports ~128k tokens (~500k+ chars). 800k is safe for filtered data.
MAX_TOTAL_CONTEXT_CHARS = 800000  

SUMMARIZE_THRESHOLD = 600

# Embedding / Chat Models
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
EMBED_MODEL = "text-embedding-3-small"
GROQ_MODEL = "llama-3.3-70b-versatile"
