"""Optional Langfuse tracing, shared by the CLI, graph, and retrieval layer."""

import json
import re
from contextlib import contextmanager
from functools import wraps

_client = None
_public_key = None
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
SECRET_FIELDS = {"api_key", "secret_key", "public_key", "authorization", "password"}
ROUTING_SPANS = {
    "RunnableCallable",
    "after_routing",
    "after_product_selection",
    "after_service_selection",
    "after_loading",
    "after_verification",
}


def _redact(value, secrets):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "report_data":
                continue
            if key.lower() in SECRET_FIELDS:
                cleaned[key] = "[REDACTED]"
            elif key in {"embedding", "alternative_embedding"} and isinstance(
                item, list
            ):
                cleaned[key] = {"dimensions": len(item)}
            else:
                cleaned[key] = _redact(item, secrets)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        value = EMAIL.sub("[REDACTED EMAIL]", value)
        try:
            parsed = json.loads(value)
        except ValueError, RecursionError:
            return value
        if isinstance(parsed, (dict, list)):
            return json.dumps(_redact(parsed, secrets), ensure_ascii=False)
    return value


def configure_tracing(settings):
    """Initialize integrations after settings have loaded credentials from .env."""
    global _client, _public_key
    if not settings.langfuse_tracing_enabled:
        return None
    if not settings.langfuse_public_key and not settings.langfuse_secret_key:
        return None
    if not all(
        (
            settings.langfuse_public_key,
            settings.langfuse_secret_key,
            settings.langfuse_base_url,
        )
    ):
        raise ValueError(
            "Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL."
        )

    from langfuse import Langfuse
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

    secrets = tuple(
        secret.get_secret_value()
        for secret in (
            settings.gemini_api_key,
            settings.langfuse_public_key,
            settings.langfuse_secret_key,
        )
        if secret and secret.get_secret_value()
    )

    def prepare_spans(*, params):
        patches = {}
        for identifier, span in params.spans.items():
            replacements = {}
            deleted = []
            dimensions = []
            for key, value in span.attributes.items():
                if key.startswith("embedding.embeddings.") and key.endswith(".vector"):
                    dimensions.append(len(value))
                    deleted.append(key)
                    continue
                masked = _redact(value, secrets)
                if masked != value:
                    replacements[key] = masked
            # The native integration stores embedding models under OpenInference attributes.
            if model := span.attributes.get("embedding.model_name"):
                replacements["langfuse.observation.model.name"] = model
            if dimensions:
                replacements["langfuse.observation.output"] = json.dumps(
                    {"embedding_count": len(dimensions), "dimensions": dimensions}
                )
            if replacements or deleted:
                patches[identifier] = OtelSpanPatch(
                    set_attributes=replacements, delete_attributes=deleted
                )
        return MaskOtelSpansResult(span_patches=patches)

    _public_key = settings.langfuse_public_key.get_secret_value()
    _client = Langfuse(
        public_key=_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_tracing_environment,
        mask_otel_spans=prepare_spans,
        should_export_span=lambda span: span.name not in ROUTING_SPANS,
    )
    GoogleGenAIInstrumentor().instrument()
    return _client


@contextmanager
def trace_turn(question, session):
    """Use one trace per turn and group turns with the existing chat session ID."""
    if _client is None:
        yield None, {}
        return

    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

    with propagate_attributes(session_id=session, tags=["metso-assistant", "chat"]):
        with _client.start_as_current_observation(
            name="answer-sales-question", as_type="agent", input=question
        ) as span:
            yield (
                span,
                {
                    "callbacks": [CallbackHandler(public_key=_public_key)],
                    "run_name": "run-sales-workflow",
                },
            )


def trace_retrieval(function):
    """Capture a typed lookup without logging its embedding vector as input."""

    @wraps(function)
    def wrapped(self, embedding, domain):
        if _client is None:
            return function(self, embedding, domain)
        with _client.start_as_current_observation(
            name=f"retrieve-{domain}-evidence",
            as_type="retriever",
            input={"domain": domain, "embedding_dimensions": len(embedding)},
            metadata={
                "vector_store": "chroma",
                "retrieval_k": self.settings.retrieval_k,
            },
        ) as span:
            sources = function(self, embedding, domain)
            span.update(output=[source.model_dump(mode="json") for source in sources])
            return sources

    return wrapped


def trace_ingestion(function):
    """Group an indexing run and its native Gemini embedding calls in one trace."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        if _client is None:
            yield from function(*args, **kwargs)
            return

        from langfuse import propagate_attributes

        with propagate_attributes(tags=["metso-assistant", "ingest"]):
            with _client.start_as_current_observation(
                name="index-source-documents",
                as_type="chain",
                input={"source_kinds": ["brochure", "service"]},
            ) as span:
                documents, chunks = 0, 0
                for path, count in function(*args, **kwargs):
                    documents += 1
                    chunks += count
                    yield path, count
                span.update(output={"documents": documents, "chunks": chunks})

    return wrapped


def flush_traces():
    if _client is not None:
        _client.flush()
