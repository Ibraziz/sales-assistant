"""Extracted service costs."""

from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, Field


class CostItem(BaseModel):
    description: str
    quantity: Decimal = Field(
        default=Decimal("1"),
        gt=0,
        description="Numeric quantity only; put units such as hours or lots in description.",
        json_schema_extra={"type": "number", "default": 1},
    )
    unit_cost: Decimal = Field(
        description="Numeric price per unit, without currency symbols or unit labels.",
        json_schema_extra={"type": "number"},
    )
    evidence: str


class ReportCosts(BaseModel):
    currency: str | None
    items: list[CostItem]
    notes: list[str] = Field(default_factory=list)
    unresolved: list[str]
    stated_total: Decimal | None = Field(
        json_schema_extra={"anyOf": [{"type": "number"}, {"type": "null"}]},
    )
    stated_total_evidence: str | None


class ServiceReport(ReportCosts):
    source: int = Field(ge=1)
    details: list[str] = Field(default_factory=list)


class ServiceReports(BaseModel):
    reports: list[ServiceReport]


class ServiceEstimate(ReportCosts):
    source: int | None = Field(default=None, ge=1)
    exclusions: list[str] = Field(default_factory=list)
    same_scope: bool = False


class ServiceEstimates(BaseModel):
    estimates: list[ServiceEstimate] = Field(default_factory=list)
    clarification: str | None = None


def calculate_cost(estimate: ServiceEstimate) -> dict:
    """Sum supported charges and compare a stated total only for matching scope."""
    result = {
        **estimate.model_dump(mode="json"),
        "items": [],
        "subtotal": None,
        "subtotal_label": (
            "reconstructed subtotal" if estimate.stated_total is None else "calculated subtotal"
        ),
        "difference": None,
    }
    if not estimate.items or estimate.source is None:
        return result

    total = Decimal("0")
    for item in estimate.items:
        amount = (item.quantity * item.unit_cost).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        result["items"].append(
            {
                **item.model_dump(mode="json"),
                "amount": str(amount),
            }
        )
        total += amount

    result["subtotal"] = str(total)
    if estimate.same_scope and estimate.stated_total is not None:
        result["difference"] = str(
            (total - estimate.stated_total).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )
    return result
