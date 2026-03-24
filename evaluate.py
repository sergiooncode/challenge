"""Entry point for ticket reply evaluation.

Reads ticket/reply pairs from a CSV file, sends each pair to GPT-4o for
evaluation on content and format, and writes results to a new CSV.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from evaluator.client import evaluate_reply
from evaluator.csv_io import read_tickets, write_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 10


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

    client = AsyncOpenAI(max_retries=0)
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
    output_path = Path("tickets_evaluated.csv")
    max_concurrency = int(os.environ.get("MAX_CONCURRENCY", DEFAULT_CONCURRENCY))

    asyncio.run(process_csv(input_path, output_path, max_concurrency))


if __name__ == "__main__":
    main()
