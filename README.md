# Ticket Reply Evaluator

Python script that evaluates AI-generated customer support replies using GPT-4o.
For each ticket/reply pair, it scores **content** (relevance, correctness, completeness)
and **format** (clarity, structure, grammar) on a 1–5 scale with brief explanations.

## Setup

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

## Run (Docker)

```bash
make run      # build image and evaluate tickets → output/tickets_evaluated.csv
make test     # run tests inside container
make clean    # remove output files and caches
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

- **Small package over monolithic script**: separates concerns (schemas, API client, CSV I/O) for readability and testability without over-abstracting
- **Async concurrency over sequential calls**: scales to large CSVs without overcomplicating the code
- **Structured output via Pydantic**: guarantees valid JSON and score ranges without manual parsing, at the cost of coupling to OpenAI's structured output feature
- **stdlib csv over pandas**: fewer dependencies for a straightforward read/write task

## What I'd do next with more time

- Configurable scoring rubrics per client
- Score consistency checks across runs to measure LLM evaluation reliability
  - in a quick 10-run experiment, most scores showed zero variance; only 2 out of 5 tickets had content score fluctuations (stdev < 0.52), suggesting GPT-4o is reliable for this task but warrants monitoring at scale
- Async rate limiting for high-traffic clients
- Evaluation caching for similar ticket patterns to reduce redundant API calls
- CI pipeline with the test suite
