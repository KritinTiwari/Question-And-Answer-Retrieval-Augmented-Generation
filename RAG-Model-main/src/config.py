import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Available Gemini models for question answering
AVAILABLE_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
]

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-1.5-flash")

# Embedding options
EMBEDDING_OPTIONS = {
    "Google Embeddings (text-embedding-004) [Recommended]": "google",
    "HuggingFace (all-MiniLM-L6-v2) [Local]": "huggingface",
}

# Default RAG Tuning Parameters for High Accuracy
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
DEFAULT_TEMPERATURE = 0.1  # Low temperature minimizes hallucination and maximizes factual precision


def get_api_key(custom_key: Optional[str] = None) -> Optional[str]:
    """
    Resolves the Google Gemini API key in order of precedence:
    1. User custom key from Streamlit UI
    2. Streamlit secrets (st.secrets["GOOGLE_API_KEY"])
    3. OS environment variable (os.environ["GOOGLE_API_KEY"])
    """
    if custom_key and custom_key.strip():
        return custom_key.strip()

    try:
        import streamlit as st
        if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"]:
            return st.secrets["GOOGLE_API_KEY"].strip()
    except Exception:
        pass

    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    return None
