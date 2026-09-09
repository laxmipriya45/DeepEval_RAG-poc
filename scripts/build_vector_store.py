"""Build the ChromaDB knowledge-base vector store."""

from retrieval.vector_store import build_vector_store


if __name__ == "__main__":
    count = build_vector_store()
    print(f"Knowledge-base vector store built successfully with {count} chunks.")
