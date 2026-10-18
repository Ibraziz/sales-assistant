"""Export the agent graph: python -m src.assistant.draw_graph [output.png]."""

import argparse
from pathlib import Path

from .agent import Agent

# uv run --frozen python -m src.assistant.draw_graph


def main():
    parser = argparse.ArgumentParser(
        description="Save the agent graph as PNG and Mermaid."
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("docs/agent_graph.png"),
        help="PNG output path (default: docs/agent_graph.png)",
    )
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("The output path must end in .png.")

    # Building the graph only binds node methods; it does not run them.
    graph = Agent(store=None, gemini=None).graph.get_graph()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mermaid_path = args.output.with_suffix(".mmd")
    mermaid_path.write_text(graph.draw_mermaid(), encoding="utf-8")
    print(f"Mermaid: {mermaid_path}")
    # LangGraph's default renderer uses the public Mermaid.INK API.
    graph.draw_mermaid_png(output_file_path=str(args.output))
    print(f"PNG: {args.output}")


if __name__ == "__main__":
    main()
