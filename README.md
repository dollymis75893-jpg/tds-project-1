# Data-Analyst Telegram Bot

An intelligent LLM agent that answers data-analysis questions sent via Telegram. Built for IIT Madras Tools in Data Science, Project 1.

## What it does

Send a data-analysis question to the bot via Telegram (inline data or a link to a public dataset like MOSPI). The agent figures out the answer—fetching data and running pandas/numpy code in a sandboxed `run_python` tool when needed—and replies with exactly one JSON object:

```json
{"answer": {"state": "Assam"}, "log_url": "https://<host>/run.jsonl"}
```

- `answer` is structured exactly as the question asks
- `log_url` is a publicly accessible JSONL log showing every agent step (questions, tool calls, tool outputs, final answers)—one JSON object per line

Multi-turn conversations are supported: per-chat history is maintained, and the agent answers the latest message in context.

## Architecture

- `bot.py` — everything in one file:
  - FastAPI app serving `/health` and `/run.jsonl` (the public agent log)
  - Background thread long-polling the Telegram Bot API (`getUpdates`)
  - Agentic loop over an OpenAI-compatible chat API with a `run_python` tool (pandas, numpy, requests, BeautifulSoup, openpyxl available; network access enabled)
  - Self-ping keep-warm mechanism so the free host never idles out

## Run

```bash
pip install -r requirements.txt
export BOT_TOKEN=...        # from @BotFather
export AIPIPE_TOKEN=...     # OpenAI-compatible API token
export BASE_URL=https://your-host   # public URL of this service
uvicorn bot:app --host 0.0.0.0 --port 8000
```
