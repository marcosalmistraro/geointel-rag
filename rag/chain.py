"""
RAG chain — combines retriever output with LLM generation.

Flow:
    user query
        → retriever (top-k chunks + spatial context)
        → prompt builder
        → HF Inference API (Phi-3-mini)
        → grounded answer

The model ID can be swapped to the fine-tuned adapter once available.

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
from pathlib import Path

import requests

from dotenv import load_dotenv

from rag.retriever import Retriever

logger = logging.getLogger(__name__)

# Swap this to your fine-tuned model ID once pushed to HF
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

PROMPT_TEMPLATE = """You are a humanitarian intelligence assistant specialising in
disaster response for the 2023 Turkey-Syria earthquake.

Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.
Do not make up facts.

Context:
{context}

Question: {question}

Answer:"""


class RAGChain:

    def __init__(
        self,
        retriever: Retriever | None = None,
        model_id: str = MODEL_ID,
        hf_token: str | None = None,
    ) -> None:
        load_dotenv()
        self.retriever = retriever or Retriever()
        self.model_id = model_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"

        if not self.hf_token:
            logger.warning("HF_TOKEN not set — API calls will fail")

    def _build_prompt(self, question: str, context: str) -> str:
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def _call_llm(self, prompt: str) -> str:
        """Call the HF Inference API and return the generated text."""
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.2,
                "return_full_text": False,
            },
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            logger.warning("LLM unreachable (network): %s", exc)
            return "[LLM unavailable — network error. Retrieval context is shown above.]"
        except requests.exceptions.HTTPError as exc:
            logger.warning("LLM HTTP error: %s", exc)
            return f"[LLM error {response.status_code}: {response.text[:200]}]"

        result = response.json()

        if isinstance(result, list) and result:
            return result[0].get("generated_text", "").strip()

        return "No response generated."

    def run(self, question: str, top_k: int | None = None) -> dict:
        """
        Run the full RAG chain.
        Returns a dict with the answer and the context used.
        top_k overrides the retriever default when provided.
        """
        logger.info("Query: %s", question)

        # 1. Retrieve
        context = self.retriever.retrieve(question, top_k=top_k)
        logger.info("Retrieved context (%d chars)", len(context))

        # 2. Build prompt
        prompt = self._build_prompt(question, context)

        # 3. Generate
        logger.info("Calling LLM (%s) ...", self.model_id)
        answer = self._call_llm(prompt)

        return {
            "question": question,
            "answer": answer,
            "context": context,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chain = RAGChain()
    result = chain.run("What was the humanitarian situation in Hatay?")
    print("\n--- Answer ---")
    print(result["answer"])
    print("\n--- Context used ---")
    print(result["context"])