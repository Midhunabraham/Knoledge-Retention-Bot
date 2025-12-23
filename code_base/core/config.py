import os

# App Settings
APP_TITLE = "Knowledge Retention & Training Bot"
PERSIST_DIR = ".chroma"
COLLECTION_NAME = "company_knowledge"

# Model Settings
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434/api"

# Retrieval & Generation Settings
DEFAULT_TOP_K = 4
MAX_CHARS_PER_CHUNK = 1000        # chunk truncation before summarization
MAX_TOTAL_CONTEXT_CHARS = 5000    # max context sent to Ollama (Increased from 500 to 5000 for better answers)
SUMMARIZE_THRESHOLD = 600         # summarize chunks longer than this
OLLAMA_TIMEOUT = 120
# ... existing constants ...

# Groq Settings
GROQ_MODEL = "llama-3.3-70b-versatile" # Very fast & smart
# Other options: "llama-3.1-8b-instant" (super fast), "mixtral-8x7b-32768"