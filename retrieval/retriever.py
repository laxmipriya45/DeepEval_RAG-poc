"""Vector similarity retrieval from the Chroma knowledge base."""

from config.model_config import TOP_K
from models.embedding_model import create_embedding
from retrieval.vector_store import get_collection


def retrieve_context(question: str, top_k: int = TOP_K) -> list[str]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    collection = get_collection(create=False)
    available = collection.count()
    if available == 0:
        return []

    query_embedding = create_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, available),
    )
    documents = results.get("documents") or [[]]
    return [str(doc) for doc in (documents[0] or [])]
