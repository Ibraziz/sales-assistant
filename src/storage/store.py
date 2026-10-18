"""Retrieve indexed evidence and persist conversation turns in Chroma."""

import json
from time import time_ns
from typing import Literal
from uuid import uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..assistant.costs import ServiceReport
from ..models import Source, Turn
from ..tracing import trace_retrieval
from .ingest import PAGE_MARKER, REPORT_DATA


class Store:
    def __init__(self, settings):
        self.settings = settings
        if not (settings.chroma_dir / "chroma.sqlite3").is_file():
            raise ValueError(
                f"Existing Chroma database not found at {settings.chroma_dir}. "
                "Run the ingest command first, or set CHROMA_DIR to a prepared database."
            )
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.product_collection = self.client.get_collection(
            settings.chroma_product_collection, embedding_function=None
        )
        self.service_collection = self.client.get_collection(
            settings.chroma_service_collection, embedding_function=None
        )
        self.memory = self.client.get_or_create_collection(
            settings.chroma_memory_collection, embedding_function=None
        )

    @trace_retrieval
    def search(
        self, embedding: list[float], domain: Literal["product", "service"]
    ) -> list[Source]:
        collection = (
            self.product_collection if domain == "product" else self.service_collection
        )
        count = collection.count()
        if not count:
            return []
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(self.settings.retrieval_k, count),
            include=["documents", "metadatas", "distances"],
        )
        selected, seen = [], set()
        for record_id, text, metadata, distance in zip(
            result["ids"][0],  # first query only, since we only have one embedding
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            if not text:
                continue
            source = Source(
                id=record_id,
                text=text,
                metadata=metadata or {},
                distance=distance,
            )
            key = (
                source.metadata.get("document_id", source.id)
                if domain == "service"
                else source.id
            )
            if key in seen:
                continue
            seen.add(key)
            # service return as a whole document text, product return as a chunked text
            if domain == "service":
                source.text = self._report_text(source)
            selected.append(source)
        return selected

    def _source_path(self, source: Source):
        path = (self.settings.data_dir / source.metadata["relative_path"]).resolve()
        if not path.is_relative_to(self.settings.data_dir):
            raise ValueError("Source path must stay inside DATA_DIR.")
        return path

    def _report_text(self, source: Source) -> str:
        return self._source_path(source).read_text(encoding="utf-8")

    def _report_record(self, source: Source):
        records = self.service_collection.get(
            where={
                "$and": [
                    {"document_id": source.metadata["document_id"]},
                    {"chunk_index": 0},
                ]
            },
            include=["metadatas"],
        )
        if not records["ids"]:
            raise ValueError("The report's first chunk is missing. Run ingest first.")
        return records

    def load_report(self, source: Source, reference: int) -> ServiceReport | None:
        data = self._report_record(source)["metadatas"][0].get(REPORT_DATA)
        if not data:
            return None
        saved = json.loads(data)
        # check if cached text is still the same as the current source text
        if saved["text"] != source.text:
            return None
        return ServiceReport(source=reference, **saved["costs"])

    # store service report data in the first chunk of a service document
    def save_report(self, source: Source, report: ServiceReport) -> None:
        records = self._report_record(source)
        data = json.dumps(
            {
                "text": source.text,
                "costs": report.model_dump(
                    mode="json", exclude={"source"}
                ),  # all extracted information from the report, except the reference number
            },
            ensure_ascii=False,
        )
        self.service_collection.update(
            ids=records["ids"], metadatas=[{REPORT_DATA: data}]
        )

    def expand_page(self, source: Source) -> Source:
        text = self._source_path(source).with_suffix(".md").read_text(encoding="utf-8")
        page = source.metadata.get("page")
        if page is not None:
            parts = PAGE_MARKER.split(text)
            pages = {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}
            text = f"{parts[0].strip()}\n\n## Page {page}\n{pages[page].strip()}"
        return source.model_copy(update={"text": text})

    def history(self, session: str) -> list[Turn]:
        records = self.memory.get(
            where={"session_id": session}, include=["documents", "metadatas"]
        )
        ordered = sorted(
            zip(records["documents"], records["metadatas"], strict=True),
            key=lambda record: record[1]["created_ns"],
        )
        return [
            Turn(**json.loads(text))
            for text, _ in ordered[-self.settings.history_turns :]
        ]

    def save(self, session: str, turn: Turn, embedding: list[float]) -> None:
        self.memory.add(
            ids=[uuid4().hex],
            documents=[json.dumps({"question": turn.question, "answer": turn.answer})],
            metadatas=[{"session_id": session, "created_ns": time_ns()}],
            embeddings=[embedding],
        )
