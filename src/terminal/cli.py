"""Parse commands and connect terminal interaction to the assistant and storage."""

import argparse
from pathlib import Path
from uuid import uuid4

from chromadb.errors import ChromaError
from google.genai.errors import APIError
from rich.console import Console
from rich.text import Text

from ..assistant.agent import Agent
from ..assistant.llm import Gemini
from ..config import load_settings
from ..storage.ingest import ingest_documents
from ..storage.store import Store
from ..tracing import configure_tracing, flush_traces
from .console import Terminal

ERRORS = (ValueError, OSError, APIError, ChromaError)


def parser():
    result = argparse.ArgumentParser(
        description="Product advice and service estimates using LangGraph and Chroma."
    )
    result.add_argument("command", choices=["ingest", "chat"])
    return result


def interact(agent, terminal):
    session = uuid4().hex
    terminal.message("Type /exit or press Ctrl+C to leave the conversation.")
    while True:
        try:
            message = terminal.read().strip()
        except EOFError, KeyboardInterrupt:
            return
        if not message:
            continue
        if message == "/exit":
            return
        try:
            with terminal.console.status("Thinking...", spinner="dots"):
                answer = agent.ask(message, session)
            terminal.answer(answer)
        except ERRORS as exc:
            terminal.message(f"Could not answer: {exc}")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        settings = load_settings(Path.cwd())
        configure_tracing(settings)
        terminal = Terminal(settings)
        gemini = Gemini(settings)
        try:
            if args.command == "ingest":
                for path, count in ingest_documents(settings, gemini):
                    terminal.message(f"{path}: {count} chunks stored")
            else:
                interact(Agent(Store(settings), gemini), terminal)
        finally:
            gemini.close()
        return 0
    except ERRORS as exc:
        Console(stderr=True).print(Text(str(exc)))
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        flush_traces()


if __name__ == "__main__":
    raise SystemExit(main())
