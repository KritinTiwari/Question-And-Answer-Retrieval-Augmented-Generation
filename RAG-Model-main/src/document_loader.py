import io
import re
from typing import List, Tuple
from langchain_core.documents import Document
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """
    Cleans raw extracted text:
    - Replaces null bytes
    - Normalizes consecutive blank lines
    - Removes non-printable control characters while preserving standard punctuation
    """
    if not text:
        return ""
    # Strip null characters
    text = text.replace("\x00", " ")
    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace consecutive spaces with a single space (while keeping single newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into 2 newlines (paragraph boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_documents_from_files(uploaded_files: List) -> Tuple[List[Document], dict]:
    """
    Extracts LangChain Document objects from multiple uploaded files (PDF, TXT, MD, CSV).
    
    Returns:
        documents: List of Document objects with metadata
        stats: Dictionary containing ingestion statistics
    """
    documents: List[Document] = []
    stats = {
        "files_processed": 0,
        "total_pages_or_sections": 0,
        "total_characters": 0,
        "errors": []
    }

    for file in uploaded_files:
        filename = getattr(file, "name", "unknown_document")
        file_ext = filename.lower().split(".")[-1] if "." in filename else ""

        try:
            # Handle PDF Documents
            if file_ext == "pdf":
                file_bytes = io.BytesIO(file.read())
                reader = PdfReader(file_bytes)
                page_count = len(reader.pages)

                for page_num, page in enumerate(reader.pages):
                    raw_text = page.extract_text() or ""
                    cleaned = clean_text(raw_text)

                    if cleaned:
                        doc = Document(
                            page_content=cleaned,
                            metadata={
                                "source": filename,
                                "page": page_num + 1,
                                "total_pages": page_count,
                                "file_type": "pdf"
                            }
                        )
                        documents.append(doc)
                        stats["total_characters"] += len(cleaned)
                        stats["total_pages_or_sections"] += 1

                stats["files_processed"] += 1

            # Handle Plain Text / Markdown Documents
            elif file_ext in ["txt", "md"]:
                content_bytes = file.read()
                try:
                    text = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    text = content_bytes.decode("latin-1", errors="replace")

                cleaned = clean_text(text)
                if cleaned:
                    doc = Document(
                        page_content=cleaned,
                        metadata={
                            "source": filename,
                            "page": 1,
                            "total_pages": 1,
                            "file_type": file_ext
                        }
                    )
                    documents.append(doc)
                    stats["total_characters"] += len(cleaned)
                    stats["total_pages_or_sections"] += 1

                stats["files_processed"] += 1

            # Handle CSV / Tabular Documents
            elif file_ext == "csv":
                content_bytes = file.read()
                try:
                    text = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    text = content_bytes.decode("latin-1", errors="replace")

                cleaned = clean_text(text)
                if cleaned:
                    doc = Document(
                        page_content=cleaned,
                        metadata={
                            "source": filename,
                            "page": 1,
                            "total_pages": 1,
                            "file_type": "csv"
                        }
                    )
                    documents.append(doc)
                    stats["total_characters"] += len(cleaned)
                    stats["total_pages_or_sections"] += 1

                stats["files_processed"] += 1

            else:
                stats["errors"].append(f"Unsupported file format: {filename}")

        except Exception as e:
            stats["errors"].append(f"Error processing {filename}: {str(e)}")

    return documents, stats
