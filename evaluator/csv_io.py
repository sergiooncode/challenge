"""CSV reading and writing utilities."""

import csv
from pathlib import Path

from evaluator.schemas import Evaluation


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
