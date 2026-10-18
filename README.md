# Metso Assistant

A terminal app with two features:

- **Product recommendations:** suggests products using the supplied brochures.
- **Historical service costs:** finds comparable service reports and calculates cost estimates from their recorded charges.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run this command from the project folder in your terminal. uv installs Python 3.14 if needed.

```text
uv sync --frozen --python 3.14
```

Copy `.env.example` to a new file named `.env`, then set `GEMINI_API_KEY` to your Gemini API key.
Keep the supplied files in `data/brochures/` and `data/HistoricalServices/`. Data, API keys, and the local database are excluded from Git.

## Ingest

Index the data before starting chat. 

```text
uv run --frozen metso-assistant ingest
```

## Chat

```text
uv run --frozen metso-assistant chat
```

Type your questions in the terminal. Use `/exit` or Ctrl+C to quit.
