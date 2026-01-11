# rag/summarizer.py
from typing import List, Dict
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from rag.config import (
    SUMMARISER_MODEL_NAME,
    MAX_INPUT_TOKENS,
    MAX_SUMMARY_TOKENS,
)
from rag.retriever import NewsRetriever


class NewsSummariser:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(SUMMARISER_MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARISER_MODEL_NAME)
        self.retriever = NewsRetriever()

    def _generate(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            return_tensors="pt",
        )
        summary_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_SUMMARY_TOKENS,
            num_beams=4,
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def summarize_vanilla(self, article_text: str) -> str:
        """
        Baseline: only the article text, no retrieval.
        """
        return self._generate(article_text)

    def _build_rag_input(self, article_text: str, retrieved_docs: List[Dict], max_context_chars: int = 2000) -> str:
        parts = ["Context:\n"]
        used = 0

        for doc in retrieved_docs:
            chunk = f"[Title] {doc.get('title','')}\n{doc.get('text','')}\n\n"
            if used + len(chunk) > max_context_chars:
                break
            parts.append(chunk)
            used += len(chunk)

        parts.append("Article:\n")
        parts.append(article_text)
        parts.append(
    "\n\nTask: Write a concise summary of the ARTICLE only. "
    "Use the CONTEXT above only to clarify entities and background, "
    "but do not invent new events or change the main facts of the article."
)

        return "".join(parts)

    def summarize_rag(self, article_text: str, top_k: int = None, debug: bool = False) -> str:
        retrieved_docs = self.retriever.retrieve(
            article_text,
            top_k=top_k if top_k is not None else 5
        )

        if debug:
            print("\n[DEBUG] Retrieved context titles:")
            for i, doc in enumerate(retrieved_docs, start=1):
                print(f"  {i}. {doc.get('title')}")

        rag_input = self._build_rag_input(article_text, retrieved_docs)
        return self._generate(rag_input)

