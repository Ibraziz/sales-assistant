"""Gemini text, Pydantic output, and normalized embeddings."""

import math

from google import genai
from google.genai import types
from pydantic import BaseModel


class Gemini:
    def __init__(self, settings):
        settings.require_key()
        self.settings = settings
        self.client = genai.Client(
            api_key=settings.gemini_api_key.get_secret_value(),
            http_options=types.HttpOptions(
                timeout=settings.api_timeout_seconds * 1000,
                retry_options=types.HttpRetryOptions(attempts=settings.max_api_retries),
            ),
        )

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self.settings.embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.settings.embedding_dimensions,
            ),
        )
        vectors = []
        # l2 normalize the embeddings
        for embedding in response.embeddings:
            vector = embedding.values
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector])
        return vectors

    def _generate(self, instruction: str, contents: str, schema=None) -> str:
        config = types.GenerateContentConfig(
            system_instruction=instruction,
            temperature=0,
            thinking_config=types.ThinkingConfig(
                thinking_level=self.settings.thinking_level
            ),
        )
        if schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = schema.model_json_schema(mode="serialization")
        response = self.client.models.generate_content(
            model=self.settings.chat_model, contents=contents, config=config
        )
        if not response.text:
            raise ValueError("Gemini returned an empty response.")
        return response.text.strip()

    def answer(self, instruction: str, contents: str) -> str:
        return self._generate(instruction, contents)

    def structured(
        self, instruction: str, contents: str, schema: type[BaseModel]
    ) -> BaseModel:
        return schema.model_validate_json(self._generate(instruction, contents, schema))

    def close(self):
        self.client.close()
