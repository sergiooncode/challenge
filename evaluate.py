"""Evaluate customer support ticket replies using GPT-4o.

Reads ticket/reply pairs from a CSV file, sends each pair to GPT-4o for
evaluation on content (relevance, correctness, completeness) and format
(clarity, structure, grammar), and writes results to a new CSV.
"""

import asyncio
import csv
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert evaluator of customer support replies.

Given a customer support ticket and an AI-generated reply, evaluate the reply on two dimensions:

1. **Content** (relevance, correctness, completeness):
   - Does the reply address the customer's issue?
   - Is the information provided accurate?
   - Does it cover everything the customer needs to know?

2. **Format** (clarity, structure, grammar/spelling):
   - Is the reply clearly written and easy to understand?
   - Is it well-structured and appropriately concise?
   - Is it free of grammar and spelling errors?

Score each dimension from 1 to 5.

Provide a brief explanation (1-2 sentences) for each score.\
"""

MAX_RETRIES = 3
BASE_DELAY = 1.0
DEFAULT_CONCURRENCY = 10
RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIError)


class Evaluation(BaseModel):
    """Structured evaluation result from GPT-4o."""

    content_score: int = Field(ge=1, le=5, description="Content score from 1 to 5")
    content_explanation: str = Field(description="Brief explanation of the content score")
    format_score: int = Field(ge=1, le=5, description="Format score from 1 to 5")
    format_explanation: str = Field(description="Brief explanation of the format score")


def read_tickets(input_path: Path) -> list[dict[str, str]]:
    """Read ticket/reply pairs from a CSV file.

    Args:
        input_path: Path to the input CSV file.

    Returns:
        List of dicts with 'ticket' and 'reply' keys.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If required columns are missing.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"ticket", "reply"}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain 'ticket' and 'reply' columns")
        return list(reader)


def write_results(
    output_path: Path,
    rows: list[dict[str, str]],
    evaluations: list[Evaluation | None],
) -> None:
    """Write evaluated results to a CSV file.

    Args:
        output_path: Path to the output CSV file.
        rows: Original ticket/reply rows.
        evaluations: Evaluation results aligned with rows (None for failures).
    """
    fieldnames = [
        "ticket",
        "reply",
        "content_score",
        "content_explanation",
        "format_score",
        "format_explanation",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, evaluation in zip(rows, evaluations):
            out = {"ticket": row.get("ticket", ""), "reply": row.get("reply", "")}
            if evaluation:
                out.update(evaluation.model_dump())
            else:
                out.update(
                    content_score="",
                    content_explanation="",
                    format_score="",
                    format_explanation="",
                )
            writer.writerow(out)


async def evaluate_reply(
    client: AsyncOpenAI,
    ticket: str,
    reply: str,
    semaphore: asyncio.Semaphore,
) -> Evaluation | None:
    """Evaluate a single ticket/reply pair using GPT-4o.

    Retries on transient API errors with exponential backoff.

    Args:
        client: AsyncOpenAI client instance.
        ticket: Customer support ticket text.
        reply: AI-generated reply text.
        semaphore: Semaphore to limit concurrent API calls.

    Returns:
        Evaluation result, or None if all retries failed.
    """
    if not ticket.strip() or not reply.strip():
        logger.warning("Skipping row with empty ticket or reply")
        return None

    user_prompt = f"Ticket: {ticket}\n\nReply: {reply}"

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.responses.parse(
                    model="gpt-4o",
                    instructions=SYSTEM_PROMPT,
                    input=user_prompt,
                    text_format=Evaluation,
                )
                return response.output_parsed
            except RETRYABLE_ERRORS as e:
                if attempt == MAX_RETRIES:
                    logger.error("Failed after %d retries: %s", MAX_RETRIES, e)
                    return None
                delay = BASE_DELAY * (2 ** (attempt - 1))
                logger.warning("Attempt %d failed (%s), retrying in %.1fs", attempt, e, delay)
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error("Unexpected error: %s", e)
                return None

    return None  # unreachable, but satisfies type checker


async def process_csv(
    input_path: Path,
    output_path: Path,
    max_concurrency: int = DEFAULT_CONCURRENCY,
) -> None:
    """Read tickets, evaluate them concurrently, and write results.

    Args:
        input_path: Path to the input CSV file.
        output_path: Path to the output CSV file.
        max_concurrency: Maximum number of concurrent API calls.
    """
    rows = read_tickets(input_path)
    logger.info("Read %d rows from %s", len(rows), input_path)

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        evaluate_reply(client, row.get("ticket", ""), row.get("reply", ""), semaphore)
        for row in rows
    ]
    evaluations = await asyncio.gather(*tasks)

    write_results(output_path, rows, list(evaluations))
    logger.info("Wrote results to %s", output_path)


def main() -> None:
    """Entry point: load config, run evaluation pipeline."""
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    input_path = Path("resources/tickets.csv")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"tickets_evaluated_{timestamp}.csv"
    max_concurrency = int(os.environ.get("MAX_CONCURRENCY", DEFAULT_CONCURRENCY))

    asyncio.run(process_csv(input_path, output_path, max_concurrency))


if __name__ == "__main__":
    main()
