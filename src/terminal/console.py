"""Render messages, answers, and clickable source references."""

from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text


class Terminal:
    def __init__(self, settings):
        self.settings = settings
        self.console = Console(highlight=False)

    def message(self, text):
        self.console.print(Text(str(text)))

    def read(self):
        return self.console.input(Text("Human: ", style="yellow"))

    def answer(self, answer):
        self.console.print(Text("Agent:", style="red"))
        self.console.print(Markdown(answer.text))
        self.references(answer.sources)

    def references(self, sources):
        for index, source in enumerate(sources, 1):
            label = Text(f"[{index}] {source.label}")
            relative = source.metadata.get("relative_path")
            if relative:
                path = (self.settings.data_dir / Path(relative)).resolve()
                if path.is_relative_to(self.settings.data_dir) and path.is_file():
                    page = source.metadata.get("page")
                    uri = path.as_uri() + (f"#page={page}" if page is not None else "")
                    label.stylize(f"blue link {uri}")
            self.console.print(label)
