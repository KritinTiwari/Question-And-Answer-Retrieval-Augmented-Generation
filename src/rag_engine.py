from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


SYSTEM_PROMPT = """You are an elite, high-precision Document Analysis & Question-Answering AI.
Your absolute priority is to provide 100% accurate, truthful, strictly grounded answers based ONLY on the provided context excerpts.

### STRICT RULES FOR 100% ACCURACY:
1. STRICT GROUNDING: Answer ONLY based on the facts and data directly stated in the context excerpts below. Never speculate, infer, extrapolate, or inject outside knowledge.
2. CITATION DISCIPLINE: For every key fact, finding, or number mentioned, cite the source and page number in brackets (e.g., "[Source: file.pdf, Page 3]").
3. TRUTHFUL REFUSAL OVER GUESSING: If the context excerpts do not contain the necessary information to fully answer the question, clearly state:
   "The provided documents do not contain sufficient information to answer this question." Do not attempt to guess or hallucinate.
4. FAITHFUL QUOTES: When citing specific definitions, policies, formulas, or rules, use verbatim quotes where possible.
5. CONCISE & CLARIFIED: Organize your answer with clear headers, bullet points, or summary tables when beneficial.

### RETRIEVED CONTEXT EXCERPTS:
{context}
"""


def format_context_excerpts(scored_docs: List[Tuple[Document, float]]) -> str:
    """
    Formats retrieved document chunks into clean, structured context blocks with
    explicit metadata tags to guide LLM grounding.
    """
    if not scored_docs:
        return "No relevant context found in uploaded documents."

    formatted_blocks = []
    for idx, (doc, score) in enumerate(scored_docs, 1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_id", idx)
        confidence_pct = round(score * 100, 1)

        header = f"--- [EXCERPT {idx}] Source: {source} | Page: {page} | Confidence: {confidence_pct}% ---"
        content = doc.page_content.strip()
        formatted_blocks.append(f"{header}\n{content}")

    return "\n\n".join(formatted_blocks)


def get_llm(
    model_name: str = "gemini-1.5-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.1
) -> ChatGoogleGenerativeAI:
    """
    Initializes ChatGoogleGenerativeAI with zero or low temperature
    to minimize variance and guarantee deterministic, factual accuracy.
    """
    if not api_key:
        raise ValueError("Google API key is required to initialize the Gemini model.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_retries=3,
        timeout=60,
    )


def build_rag_chain(llm: ChatGoogleGenerativeAI):
    """
    Constructs a modern LangChain Expression Language (LCEL) chain
    incorporating conversational history and anti-hallucination system prompt.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()


def answer_question(
    query: str,
    scored_docs: List[Tuple[Document, float]],
    chat_history_tuples: Optional[List[Tuple[str, str]]] = None,
    model_name: str = "gemini-1.5-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    Executes grounded question answering against retrieved excerpts.
    
    Returns:
        dict with keys:
            - 'answer': Generated answer string
            - 'citations': List of source metadata dicts
            - 'raw_context': Formatted context string passed to LLM
    """
    if not scored_docs:
        return {
            "answer": "⚠️ No relevant information was found in your uploaded documents. Please ensure your query matches the document contents or try rephrasing.",
            "citations": [],
            "raw_context": ""
        }

    formatted_context = format_context_excerpts(scored_docs)

    # Convert past chat history tuples to LangChain message objects
    messages_history: List[BaseMessage] = []
    if chat_history_tuples:
        # Keep last 6 exchanges (12 messages) to stay focused on relevant context
        for user_msg, assistant_msg in chat_history_tuples[-6:]:
            messages_history.append(HumanMessage(content=user_msg))
            messages_history.append(AIMessage(content=assistant_msg))

    llm = get_llm(model_name=model_name, api_key=api_key, temperature=temperature)
    chain = build_rag_chain(llm)

    response_text = chain.invoke({
        "context": formatted_context,
        "chat_history": messages_history,
        "question": query,
    })

    # Prepare structured citations for UI display
    citations = []
    for doc, score in scored_docs:
        citations.append({
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 1),
            "chunk_id": doc.metadata.get("chunk_id", 0),
            "confidence_score": round(score * 100, 1),
            "snippet": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
            "full_content": doc.page_content,
        })

    return {
        "answer": response_text,
        "citations": citations,
        "raw_context": formatted_context
    }
