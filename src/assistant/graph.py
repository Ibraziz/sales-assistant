"""Connect small actions and make each workflow branch explicit."""

from langgraph.graph import END, START, StateGraph

from ..models import AgentState

MAX_REVIEWS = 2


def after_routing(state: AgentState):
    return "draft_answer" if state.decision.domain == "general" else "embed_query"


def after_product_selection(state: AgentState):
    return "expand_product_pages" if state.sources else "draft_answer"


def after_service_selection(state: AgentState):
    return "load_reports" if state.sources else "draft_answer"


def after_loading(state: AgentState):
    if len(state.reports) < len(state.sources):
        return "extract_reports"
    return "apply_requirements"


def after_verification(state: AgentState):
    if state.verification.passed:
        return END
    if state.failed_reviews >= MAX_REVIEWS:
        return "verification_failed"
    if state.verification.needs_recalculation and state.estimates is not None:
        return "load_reports"
    return "draft_answer"


def build_graph(agent):
    graph = StateGraph(AgentState)
    graph.add_node("route_request", agent.route_request)
    graph.add_node("embed_query", agent.embed_query)
    graph.add_node("retrieve_product_chunks", agent.retrieve_product_chunks)
    graph.add_node("select_product_brochures", agent.select_product_brochures)
    graph.add_node("expand_product_pages", agent.expand_product_pages)
    graph.add_node("retrieve_service_reports", agent.retrieve_service_reports)
    graph.add_node("select_service_reports", agent.select_service_reports)
    graph.add_node("load_reports", agent.load_reports)
    graph.add_node("extract_reports", agent.extract_reports)
    graph.add_node("save_reports", agent.save_reports)
    graph.add_node("apply_requirements", agent.apply_requirements)
    graph.add_node("order_estimates_by_source", agent.order_estimates_by_source)
    graph.add_node("calculate_estimates", agent.calculate_estimates)
    graph.add_node("draft_answer", agent.draft_answer)
    graph.add_node("verify_answer", agent.verify_answer)
    graph.add_node("verification_failed", agent.verification_failed)

    graph.add_edge(START, "route_request")
    graph.add_conditional_edges(
        "route_request", after_routing, ["embed_query", "draft_answer"]
    )
    graph.add_conditional_edges(
        "embed_query",
        lambda state: state.decision.domain,
        {
            "product": "retrieve_product_chunks",
            "service": "retrieve_service_reports",
        },
    )
    graph.add_edge("retrieve_product_chunks", "select_product_brochures")
    graph.add_conditional_edges(
        "select_product_brochures",
        after_product_selection,
        ["expand_product_pages", "draft_answer"],
    )
    graph.add_edge("expand_product_pages", "draft_answer")
    graph.add_edge("retrieve_service_reports", "select_service_reports")
    graph.add_conditional_edges(
        "select_service_reports", after_service_selection, ["load_reports", "draft_answer"]
    )
    graph.add_conditional_edges(
        "load_reports", after_loading, ["extract_reports", "apply_requirements"]
    )
    graph.add_edge("extract_reports", "save_reports")
    graph.add_edge("save_reports", "apply_requirements")
    graph.add_edge("apply_requirements", "order_estimates_by_source")
    graph.add_edge("order_estimates_by_source", "calculate_estimates")
    graph.add_edge("calculate_estimates", "draft_answer")
    graph.add_edge("draft_answer", "verify_answer")
    graph.add_conditional_edges(
        "verify_answer",
        after_verification,
        [END, "load_reports", "draft_answer", "verification_failed"],
    )
    graph.add_edge("verification_failed", END)
    return graph.compile()
