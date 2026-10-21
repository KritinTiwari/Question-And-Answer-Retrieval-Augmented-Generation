import os
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Splits documents into overlapping chunks while preserving sentence boundaries and paragraphs.
    Adds chunk index and metadata for traceability.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
        length_function=len,
        is_separator_regex=False
    )

    split_docs = text_splitter.split_documents(documents)

    # Enrich metadata with unique chunk identifier
    for idx, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = idx + 1
        doc.metadata["chunk_length"] = len(doc.page_content)

    return split_docs


def create_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> Tuple[FAISS, int]:
    """
    Chunks documents and indexes them in an in-memory FAISS vector database.
    
    Returns:
        vector_store: FAISS index
        chunk_count: Number of chunks indexed
    """
    if not documents:
        raise ValueError("No documents provided for indexing.")

    chunks = split_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store, len(chunks)


def search_with_relevance(
    vector_store: FAISS,
    query: str,
    k: int = 4,
    min_score_threshold: float = 0.0
) -> List[Tuple[Document, float]]:
    """
    Performs similarity search with relevance scoring.
    Normalizes L2 distance into a confidence percentage (0% to 100%).
    Filters out results below the score threshold for high accuracy.
    """
    results_with_scores = vector_store.similarity_search_with_score(query, k=k)
    scored_docs: List[Tuple[Document, float]] = []

    for doc, distance in results_with_scores:
        # Distance to similarity confidence score conversion
        # When distance is small, confidence approaches 1.0 (100%)
        confidence = 1.0 / (1.0 + float(distance))
        if confidence >= min_score_threshold:
            scored_docs.append((doc, confidence))

    return scored_docs


def save_vector_store(vector_store: FAISS, folder_path: str = "storage/faiss_index"):
    """Persists FAISS vector index to disk."""
    os.makedirs(folder_path, exist_ok=True)
    vector_store.save_local(folder_path)


def load_vector_store(
    folder_path: str,
    embeddings: Embeddings
) -> Optional[FAISS]:
    """Loads a persisted FAISS vector index from disk."""
    if os.path.exists(folder_path):
        return FAISS.load_local(
            folder_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None
