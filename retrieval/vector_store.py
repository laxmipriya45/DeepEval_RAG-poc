"""ChromaDB lifecycle and knowledge-base indexing."""

from pathlib import Path

import chromadb
import pandas as pd

from config.model_config import EMBEDDING_MODEL
from models.embedding_model import create_embedding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = PROJECT_ROOT / "vector_store" / "chroma_db"
COLLECTION_NAME = "knowledge_base"
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "datasets" / "knowledge_base.csv"


def get_client():
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_collection(create: bool = False):
    client = get_client()
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"embedding_model": EMBEDDING_MODEL},
        )
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "ChromaDB collection 'knowledge_base' does not exist. "
            "Run 'python scripts/build_vector_store.py' first."
        ) from exc


def build_vector_store(knowledge_base_path=KNOWLEDGE_BASE_PATH):
    """Create/update the Chroma collection from the CSV knowledge base."""
    path = Path(knowledge_base_path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge-base file not found: {path}")

    df = pd.read_csv(path)
    required = {"chunk_id", "content"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Knowledge base is missing required column(s): {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("Knowledge base is empty.")
    if df["chunk_id"].duplicated().any():
        raise ValueError("Knowledge base contains duplicate chunk_id values.")

    collection = get_collection(create=True)
    ids, documents, embeddings = [], [], []
    for _, row in df.iterrows():
        chunk_id = str(row["chunk_id"]).strip()
        content = str(row["content"]).strip()
        if not chunk_id or not content:
            raise ValueError("Knowledge base contains an empty chunk_id or content value.")
        ids.append(chunk_id)
        documents.append(content)
        embeddings.append(create_embedding(content))

    collection.upsert(ids=ids, documents=documents, embeddings=embeddings)
    return collection.count()


def vector_store_exists() -> bool:
    try:
        return get_collection(create=False).count() > 0
    except RuntimeError:
        return False
