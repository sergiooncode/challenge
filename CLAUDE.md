# Challenge

Read resources/tickets.csv (columns: ticket, reply). For each row, call OpenAI GPT-4o
to evaluate the reply on content (relevance, correctness, completeness) and
format (clarity, structure, grammar). Output tickets_evaluated.csv with
original columns plus: content_score, content_explanation, format_score,
format_explanation. Scores 1-5, explanations 1-2 sentences.

## Input file

resources/tickets.csv

## Deliverables

- Python script (.py) or Jupyter notebook (.ipynb)
- tickets_evaluated.csv with the 6 columns
- README with setup and run instructions
- Optional: tests with pytest

# My read

- Deliverable: Python script, output CSV, README, optional tests
- They're evaluating: prompt engineering, code quality, error handling
- Use Pydantic for structured output (they use it internally)
- Use OpenAI SDK (they use it internally)
- Handle edge cases: empty replies, empty tickets, API failures

# Rules

- Do not invent requirements
- Do not over-engineer
- Use environment variables for API key, never commit keys
- PEP-8, type hints, docstrings
- Handle errors: missing data, API issues, malformed responses
- List dependencies in requirements.txt

# Decisions

- Python 3.13, GPT-4o, evaluator package (schemas, client, csv_io) + evaluate.py entry point
- Pydantic structured output via OpenAI SDK (no manual JSON parsing)
- asyncio + semaphore for concurrent API calls (scales to large CSVs)
- stdlib csv (no pandas), python-dotenv for env loading
- Docker as primary run method for reviewer convenience
- pytest with mocked OpenAI client for tests