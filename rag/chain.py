"""
RAG chain — combines retriever output with LLM generation.

Flow:
    user query
        → retriever (top-k chunks + spatial context)
        → prompt builder
        → Groq API (Llama 3.1 8B)
        → grounded answer

Main entry point:
    from rag.chain import RAGChain
    chain = RAGChain()
    answer = chain.run("What was the humanitarian situation in Hatay?")

Run with:
    python -m rag.chain
"""

from __future__ import annotations

import logging
import os

import requests
from dotenv import load_dotenv

from rag.retriever import Retriever

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are a humanitarian intelligence assistant specialising in "
    "disaster response for the 2023 Turkey-Syria earthquake. "
    "Answer using ONLY the context provided. "
    "If the context does not contain enough information, say so clearly. "
    "Do not make up facts."
)


class RAGChain:

    def __init__(
        self,
        retriever: Retriever | None = None,
        model_id: str = MODEL_ID,
        groq_api_key: str | None = None,
    ) -> None:
        load_dotenv()
        self.retriever = retriever or Retriever()
        self.model_id = model_id
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")

        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY not set — LLM calls will fail")

    def _build_messages(self, question: str, context: str) -> list[dict]:
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _call_llm(self, question: str, context: str, model_id: str | None = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id or self.model_id,
            "messages": self._build_messages(question, context),
            "max_tokens": 512,
            "temperature": 0.2,
        }

        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            logger.warning("LLM unreachable (network): %s", exc)
            return "[LLM unavailable — network error. Retrieval context is shown above.]"
        except requests.exceptions.HTTPError as exc:
            logger.warning("LLM HTTP error: %s", exc)
            return f"[LLM error {response.status_code}: {response.text[:200]}]"

        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def stream(self, question: str, top_k: int | None = None, model_id: str | None = None):
        """
        Generator that yields (event_type, data) tuples.
        First yields ("context", context_string), then ("token", token) per LLM token.
        """
        import json as _json

        context = self.retriever.retrieve(question, top_k=top_k)
        yield "context", context

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id or self.model_id,
            "messages": self._build_messages(question, context),
            "max_tokens": 512,
            "temperature": 0.2,
            "stream": True,
        }

        try:
            with requests.post(GROQ_API_URL, headers=headers, json=payload, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line and line.startswith(b"data: "):
                        data_str = line[6:].decode()
                        if data_str == "[DONE]":
                            return
                        try:
                            data = _json.loads(data_str)
                            content = data["choices"][0]["delta"].get("content", "")
                            if content:
                                yield "token", content
                        except _json.JSONDecodeError:
                            pass
        except requests.exceptions.ConnectionError as exc:
            logger.warning("LLM unreachable (network): %s", exc)
            yield "token", "[LLM unavailable — network error. Retrieval context is shown above.]"
        except requests.exceptions.HTTPError as exc:
            logger.warning("LLM HTTP error: %s", exc)
            yield "token", f"[LLM error {resp.status_code}: {resp.text[:200]}]"

    def run(self, question: str, top_k: int | None = None, model_id: str | None = None) -> dict:
        """
        Run the full RAG chain.
        Returns a dict with the answer and the context used.
        top_k overrides the retriever default when provided.
        model_id overrides self.model_id for this call only.
        """
        effective_model = model_id or self.model_id
        logger.info("Query: %s", question)

        context = self.retriever.retrieve(question, top_k=top_k)
        logger.info("Retrieved context (%d chars)", len(context))

        logger.info("Calling LLM (%s) ...", effective_model)
        answer = self._call_llm(question, context, model_id=model_id)

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "model_id": effective_model,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chain = RAGChain()
    result = chain.run("What was the humanitarian situation in Hatay?")
    print("\n--- Answer ---")
    print(result["answer"])
    print("\n--- Context used ---")
    print(result["context"])
