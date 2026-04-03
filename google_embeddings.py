"""
google_embeddings.py — Custom Google Embeddings Wrapper

Calls the Google Generative AI embedding API directly via google-genai SDK
using api_version='v1beta' where gemini-embedding-001 is available.

The langchain-google-genai v4.x wrapper has broken model routing. This
wrapper bypasses it and calls the SDK directly with correct configuration.
"""

import os
from typing import List
from langchain_core.embeddings import Embeddings
from google import genai
from google.genai import types


class GoogleV1Embeddings(Embeddings):
    """
    LangChain-compatible embeddings using Google Generative AI (v1 API).

    Directly uses the google-genai SDK with api_version='v1' to avoid
    the v1beta 404 errors from the langchain-google-genai wrapper.

    Args:
        model:          Model name, e.g. 'models/text-embedding-004'
        api_key:        Google AI Studio API key
        task_type:      Embedding task type (default: RETRIEVAL_DOCUMENT)
    """

    def __init__(
        self,
        model: str = "models/gemini-embedding-001",
        api_key: str = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.task_type = task_type
        self._client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1beta"},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        # Batch in groups of 100 (API limit)
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            contents = [types.Content(parts=[types.Part(text=t)]) for t in batch]
            result = self._client.models.embed_content(
                model=self.model,
                contents=contents,
                config=types.EmbedContentConfig(task_type=self.task_type),
            )
            embeddings.extend([e.values for e in result.embeddings])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        result = self._client.models.embed_content(
            model=self.model,
            contents=[types.Content(parts=[types.Part(text=text)])],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values
