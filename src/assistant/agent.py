"""Small graph actions for retrieval, service costs, answers, and verification."""

import json
import logging

from ..models import AgentState, Answer, EvidenceSelection, Route, Turn, Verification
from ..tracing import trace_turn
from .costs import ServiceEstimates, ServiceReports, calculate_cost
from .graph import build_graph
from .prompts import (
    GENERAL_ANSWER_PROMPT,
    PRODUCT_ANSWER_PROMPT,
    PRODUCT_SELECTION_PROMPT,
    REPORT_PROMPT,
    ROUTER_PROMPT,
    SERVICE_ANSWER_PROMPT,
    SERVICE_PROMPT,
    SERVICE_SELECTION_PROMPT,
    VERIFY_PROMPT,
)

logger = logging.getLogger(__name__)


def context(state: AgentState, sources=None, **extra) -> str:
    return json.dumps(
        {
            "question": state.question,
            "conversation": [turn.model_dump() for turn in state.history],
            "evidence": [
                {"reference": i, "source": source.label, "text": source.text}
                for i, source in enumerate(
                    state.sources if sources is None else sources, 1
                )
            ],
            "calculation": state.calculation,
            "estimates": (
                state.estimates.model_dump(mode="json") if state.estimates else None
            ),
            "draft": state.answer,
            "verification": (
                state.verification.model_dump() if state.verification else None
            ),
            **extra,
        },
        ensure_ascii=False,
    )


class Agent:
    def __init__(self, store, gemini):
        self.store = store
        self.gemini = gemini
        self.graph = build_graph(self)

    def route_request(self, state: AgentState) -> dict:
        return {
            "decision": self.gemini.structured(ROUTER_PROMPT, context(state), Route)
        }

    def embed_query(self, state: AgentState) -> dict:
        embedding = self.gemini.embed_query(state.decision.query)
        alternative_query = (state.decision.alternative_query or "").strip()
        alternative_embedding = []
        if (
            state.decision.domain == "product"
            and alternative_query
            and alternative_query != state.decision.query.strip()
        ):
            alternative_embedding = self.gemini.embed_query(alternative_query)
        return {
            "embedding": embedding,
            "alternative_embedding": alternative_embedding,
        }

    def retrieve_product_chunks(self, state: AgentState) -> dict:
        candidates = {}
        for embedding in (state.embedding, state.alternative_embedding):
            if embedding:
                for source in self.store.search(embedding, "product"):
                    candidates.setdefault(source.id, source)
        return {"candidates": list(candidates.values())}

    def retrieve_service_reports(self, state: AgentState) -> dict:
        return {"candidates": self.store.search(state.embedding, "service")}

    def select_product_brochures(self, state: AgentState) -> dict:
        return self._select_documents(state, PRODUCT_SELECTION_PROMPT)

    def select_service_reports(self, state: AgentState) -> dict:
        return self._select_documents(state, SERVICE_SELECTION_PROMPT)

    def _select_documents(self, state: AgentState, prompt: str) -> dict:
        if not state.candidates:
            return {"sources": []}
        selection = self.gemini.structured(
            prompt, context(state, state.candidates), EvidenceSelection
        )
        # Keep one source per product page or service document, preserving rank.
        selected, seen = [], set()
        for reference in selection.references:
            if not 1 <= reference <= len(state.candidates):
                continue
            source = state.candidates[reference - 1]
            document_id = source.metadata.get("document_id", source.id)
            page = (
                source.metadata.get("page")
                if state.decision.domain == "product"
                else None
            )
            key = (document_id, page)
            if key not in seen:
                selected.append(source)
                seen.add(key)
        return {"sources": selected[:5]}

    def expand_product_pages(self, state: AgentState) -> dict:
        return {"sources": [self.store.expand_page(source) for source in state.sources]}

    def load_reports(self, state: AgentState) -> dict:
        # force a re-extraction if any previous review failed
        if state.failed_reviews:
            return {"reports": []}
        reports = [
            self.store.load_report(source, reference)
            for reference, source in enumerate(state.sources, 1)
        ]
        return {"reports": [report for report in reports if report is not None]}

    def extract_reports(self, state: AgentState) -> dict:
        loaded = {report.source for report in state.reports}
        evidence = [
            {"reference": reference, "source": source.label, "text": source.text}
            for reference, source in enumerate(state.sources, 1)
            if reference not in loaded
        ]
        extracted = self.gemini.structured(
            REPORT_PROMPT,
            json.dumps(
                {
                    "evidence": evidence,
                    "issues": state.verification.issues if state.verification else [],
                },
                ensure_ascii=False,
            ),
            ServiceReports,
        )
        return {"reports": [*state.reports, *extracted.reports]}

    def save_reports(self, state: AgentState) -> dict:
        for report in state.reports:
            if not 1 <= report.source <= len(state.sources):
                raise ValueError(
                    "The report extraction refers to an unavailable source."
                )
            self.store.save_report(state.sources[report.source - 1], report)
        return {"reports": state.reports}

    def apply_requirements(self, state: AgentState) -> dict:
        return {
            "estimates": self.gemini.structured(
                SERVICE_PROMPT,
                context(
                    state,
                    sources=[],
                    reports=[
                        report.model_dump(mode="json") for report in state.reports
                    ],
                ),
                ServiceEstimates,
            )
        }

    def order_estimates_by_source(self, state: AgentState) -> dict:
        # The selector already ranked the sources; keep estimates in that order.
        ordered = sorted(
            state.estimates.estimates,
            key=lambda estimate: estimate.source or len(state.sources) + 1,
        )
        return {"estimates": state.estimates.model_copy(update={"estimates": ordered})}

    def calculate_estimates(self, state: AgentState) -> dict:
        for estimate in state.estimates.estimates:
            if estimate.source is not None and estimate.source > len(state.sources):
                raise ValueError("The cost extraction refers to an unavailable source.")
        return {
            "calculation": {
                "reports": [
                    calculate_cost(estimate) for estimate in state.estimates.estimates
                ],
                "clarification": state.estimates.clarification,
            }
        }

    def draft_answer(self, state: AgentState) -> dict:
        prompt = {
            "product": PRODUCT_ANSWER_PROMPT,
            "service": SERVICE_ANSWER_PROMPT,
            "general": GENERAL_ANSWER_PROMPT,
        }[state.decision.domain]
        return {"answer": self.gemini.answer(prompt, context(state))}

    def verify_answer(self, state: AgentState) -> dict:
        review = self.gemini.structured(VERIFY_PROMPT, context(state), Verification)
        if not review.passed:
            logger.warning(
                "Review %s failed (needs_recalculation=%s): %s",
                state.failed_reviews + 1,
                review.needs_recalculation,
                "; ".join(review.issues),
            )
        return {
            "verification": review,
            "failed_reviews": state.failed_reviews + int(not review.passed),
        }

    def verification_failed(self, state: AgentState) -> dict:
        return {
            "answer": (
                "I couldn't verify this answer against the available evidence. "
                "Please try again or provide more details."
            )
        }

    def ask(self, question: str, session: str) -> Answer:
        if not question.strip():
            raise ValueError("Enter a question.")
        with trace_turn(question.strip(), session) as (span, config):
            state = AgentState(
                question=question.strip(), history=self.store.history(session)
            )
            result = AgentState.model_validate(self.graph.invoke(state, config=config))
            self.store.save(
                session,
                Turn(question=result.question, answer=result.answer),
                result.embedding,
            )
            if span is not None:
                span.update(
                    output=result.answer,
                    metadata={
                        "domain": result.decision.domain,
                        "selected_sources": len(result.sources),
                        "failed_reviews": result.failed_reviews,
                        "verification_passed": result.verification.passed,
                    },
                )
            return Answer(text=result.answer, sources=result.sources)
