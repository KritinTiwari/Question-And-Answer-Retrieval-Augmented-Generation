import os
from typing import Optional
from langchain_core.embeddings import Embeddings


def get_embedding_model(
    provider: str = "google",
    api_key: Optional[str] = None
) -> Embeddings:
    """
    Factory to return embedding model based on chosen provider.
    
    Supported providers:
    - 'google': State-of-the-art text-embedding-004 (High precision, zero local download overhead)
    - 'huggingface': Local sentence-transformers/all-MiniLM-L6-v2 (Offline capable)
    """
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Google API Key is required to use Google Embeddings. "
                "Please enter your API Key in the sidebar or provide it via environment variables."
            )
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=key
        )

    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
