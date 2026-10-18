from typing import Literal

from pydantic import BaseModel, Field

from .assistant.costs import ServiceEstimates, ServiceReport


# source is a single chunk of text from a document
class Source(BaseModel):
    id: str
    text: str
    metadata: dict = Field(default_factory=dict)
    distance: float = 0

    @property
    def label(self) -> str:
        path = self.metadata.get("relative_path", self.id)
        page = self.metadata.get("page")
        return f"{path}, page {page}" if page is not None else str(path)


class Turn(BaseModel):
    question: str
    answer: str


class Answer(BaseModel):
    text: str
    sources: list[Source] = Field(default_factory=list)


class Route(BaseModel):
    domain: Literal["product", "service", "general"]
    query: str = Field(
        description="A standalone search query incorporating relevant chat context."
    )
    alternative_query: str | None = Field(
        default=None,
        description="One complementary technical product query; null for service or general.",
    )


class EvidenceSelection(BaseModel):
    references: list[int] = Field(default_factory=list)


class Verification(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    needs_recalculation: bool = False


class AgentState(BaseModel):
    question: str
    history: list[Turn] = Field(default_factory=list)
    decision: Route | None = None
    embedding: list[float] = Field(default_factory=list)
    alternative_embedding: list[float] = Field(default_factory=list)
    candidates: list[Source] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    reports: list[ServiceReport] = Field(default_factory=list)
    estimates: ServiceEstimates | None = None
    calculation: dict | None = None
    answer: str = ""
    failed_reviews: int = 0
    verification: Verification | None = None
