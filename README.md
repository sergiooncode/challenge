# Ticket Reply Evaluator

Python script that evaluates AI-generated customer support replies using GPT-4o.
For each ticket/reply pair, it scores **content** (relevance, correctness, completeness)
and **format** (clarity, structure, grammar) on a 1–5 scale with brief explanations.

## Setup

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

## Run with Docker (recommended)

```bash
docker build -t evaluate .
docker run --env-file .env -v "$(pwd)":/app evaluate
```

The output file `tickets_evaluated.csv` will appear in the project directory.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python evaluate.py
```

## Run tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Configuration

| Variable          | Default | Description                        |
|-------------------|---------|------------------------------------|
| `OPENAI_API_KEY`  | —       | Required. Your OpenAI API key.     |
| `MAX_CONCURRENCY` | `10`    | Max parallel API calls.            |

## Assumptions

- Each ticket/reply pair is independent and can be evaluated in isolation
- Rows with empty ticket or reply fields are skipped (written with blank scores)
- Transient API errors are retried up to 3 times with exponential backoff; persistent failures produce blank scores rather than crashing the run

## Tradeoffs

- **Single script over multiple modules**: keeps the solution simple and easy to review for a small, focused task
- **Async concurrency over sequential calls**: scales to large CSVs without overcomplicating the code
- **Structured output via Pydantic**: guarantees valid JSON and score ranges without manual parsing, at the cost of coupling to OpenAI's structured output feature
- **stdlib csv over pandas**: fewer dependencies for a straightforward read/write task
