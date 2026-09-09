"""Application LLM client for grounded response generation."""

import ollama

from config.model_config import GENERATION_MODEL


def generate_response(question: str, retrieval_context: list[str]) -> str:
    """Generate an answer using only the retrieved knowledge-base context."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if retrieval_context is None:
        raise ValueError("retrieval_context must not be None")

    context_text = "\n".join(f"- {chunk}" for chunk in retrieval_context)
    if not context_text:
        context_text = "(No relevant knowledge-base context was retrieved.)"

    prompt = (
        "Answer the question using only the information in the provided context.\n"
        "Do not add facts that are not supported by the context.\n"
        "If the context does not contain enough information to answer, say so clearly.\n"
        "Follow the user's requested format or length when it does not conflict with these rules.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}"
    )

    try:
        response = ollama.chat(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate a response with Ollama model '{GENERATION_MODEL}'. "
            "Make sure Ollama is running and the model is available."
        ) from exc

    try:
        return response["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Ollama returned an unexpected response format.") from exc
