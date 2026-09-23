"""Central settings. Values come from environment variables (or a .env file)."""
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

# Any OpenAI-compatible chat API works (Groq, OpenRouter, Together, local vLLM/Ollama, ...)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

INDEX_DIR = os.getenv("INDEX_DIR", "index")
ROUTER_DIR = os.getenv("ROUTER_DIR", "router_model")
TOPICS_FILE = os.getenv("TOPICS_FILE", "data/topics.txt")
ARTICLES_CACHE = os.getenv("ARTICLES_CACHE", "data/articles.json")

CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "3"))      # max searches in multi-step mode
MAX_NOTES = int(os.getenv("MAX_NOTES", "12"))     # max passages kept in multi-step context
