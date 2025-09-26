import os
from dotenv import load_dotenv
load_dotenv()

# Paths
DATA_DIR = os.getenv("DATA_DIR", "./data_html")
INDEX_DIR = os.getenv("INDEX_DIR", "./index_store")

# Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

# --- Internal LLM (for re-ranking, query rewriting, and answering with local context) ---
INTERNAL_OPENAI_API_KEY  = os.getenv("INTERNAL_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
INTERNAL_OPENAI_API_BASE = os.getenv("INTERNAL_OPENAI_API_BASE", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
INTERNAL_OPENAI_MODEL    = os.getenv("INTERNAL_OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

# --- External LLM (for general questions / agent mode) ---
EXTERNAL_OPENAI_API_KEY  = os.getenv("EXTERNAL_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
EXTERNAL_OPENAI_API_BASE = os.getenv("EXTERNAL_OPENAI_API_BASE", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
EXTERNAL_OPENAI_MODEL    = os.getenv("EXTERNAL_OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
BASE_URL_PUBLIC = os.getenv("BASE_URL_PUBLIC", "/public")
BASE_URL_STATIC_HTML = os.getenv("BASE_URL_STATIC_HTML", "/html")
