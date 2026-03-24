"""Tests for the ticket reply evaluation script."""

import csv
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from evaluator.schemas import Evaluation
from evaluator.client import evaluate_reply
from evaluator.csv_io import read_tickets, write_results


# --- Evaluation model tests ---


class TestEvaluationModel:
    """Tests for the Pydantic Evaluation model."""

    def test_valid_evaluation(self):
        e = Evaluation(
            content_score=4,
            content_explanation="Relevant and complete.",
            format_score=5,
            format_explanation="Clear and well-structured.",
        )
        assert e.content_score == 4
        assert e.format_score == 5

    def test_score_boundaries(self):
        for score in (1, 5):
            e = Evaluation(
                content_score=score,
                content_explanation="ok",
                format_score=score,
                format_explanation="ok",
            )
            assert e.content_score == score

    def test_score_out_of_range_low(self):
        with pytest.raises(ValidationError):
            Evaluation(
                content_score=0,
                content_explanation="ok",
                format_score=3,
                format_explanation="ok",
            )

    def test_score_out_of_range_high(self):
        with pytest.raises(ValidationError):
            Evaluation(
                content_score=6,
                content_explanation="ok",
                format_score=3,
                format_explanation="ok",
            )


# --- CSV reading tests ---


class TestReadTickets:
    """Tests for CSV reading."""

    def test_read_valid_csv(self, tmp_path: Path):
        csv_file = tmp_path / "tickets.csv"
        csv_file.write_text("ticket,reply\nHello,Hi there\n", encoding="utf-8")
        rows = read_tickets(csv_file)
        assert len(rows) == 1
        assert rows[0]["ticket"] == "Hello"
        assert rows[0]["reply"] == "Hi there"

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_tickets(tmp_path / "nonexistent.csv")

    def test_missing_columns(self, tmp_path: Path):
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("foo,bar\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="ticket.*reply"):
            read_tickets(csv_file)

    def test_empty_csv_with_headers(self, tmp_path: Path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("ticket,reply\n", encoding="utf-8")
        rows = read_tickets(csv_file)
        assert rows == []


# --- CSV writing tests ---


class TestWriteResults:
    """Tests for CSV writing."""

    def test_write_with_evaluations(self, tmp_path: Path):
        output = tmp_path / "out.csv"
        rows = [{"ticket": "Q", "reply": "A"}]
        evals = [
            Evaluation(
                content_score=4,
                content_explanation="Good.",
                format_score=5,
                format_explanation="Great.",
            )
        ]
        write_results(output, rows, evals)

        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            result = list(reader)

        assert len(result) == 1
        assert result[0]["content_score"] == "4"
        assert result[0]["format_explanation"] == "Great."

    def test_write_with_failed_evaluation(self, tmp_path: Path):
        output = tmp_path / "out.csv"
        rows = [{"ticket": "Q", "reply": "A"}]
        evals = [None]
        write_results(output, rows, evals)

        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            result = list(reader)

        assert result[0]["content_score"] == ""
        assert result[0]["format_score"] == ""

    def test_roundtrip_preserves_data(self, tmp_path: Path):
        """Write and read back to verify all columns are present."""
        output = tmp_path / "out.csv"
        rows = [{"ticket": "Help me", "reply": "Sure thing"}]
        evals = [
            Evaluation(
                content_score=3,
                content_explanation="Adequate.",
                format_score=4,
                format_explanation="Clean.",
            )
        ]
        write_results(output, rows, evals)

        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {
                "ticket",
                "reply",
                "content_score",
                "content_explanation",
                "format_score",
                "format_explanation",
            }


# --- evaluate_reply tests ---


class TestEvaluateReply:
    """Tests for the evaluate_reply function with mocked API."""

    @pytest.fixture
    def semaphore(self):
        return asyncio.Semaphore(1)

    @pytest.fixture
    def mock_evaluation(self):
        return Evaluation(
            content_score=4,
            content_explanation="Relevant response.",
            format_score=5,
            format_explanation="Well-written.",
        )

    def test_successful_evaluation(self, semaphore, mock_evaluation):
        client = AsyncMock()
        response = MagicMock()
        response.output_parsed = mock_evaluation
        client.responses.parse = AsyncMock(return_value=response)

        result = asyncio.run(evaluate_reply(client, "ticket text", "reply text", semaphore))
        assert result is not None
        assert result.content_score == 4
        assert result.format_score == 5

    def test_empty_ticket_returns_none(self, semaphore):
        client = AsyncMock()
        result = asyncio.run(evaluate_reply(client, "", "reply text", semaphore))
        assert result is None

    def test_empty_reply_returns_none(self, semaphore):
        client = AsyncMock()
        result = asyncio.run(evaluate_reply(client, "ticket text", "  ", semaphore))
        assert result is None

    def test_retryable_error_retries_then_fails(self, semaphore):
        from openai import RateLimitError

        client = AsyncMock()
        client.responses.parse = AsyncMock(
            side_effect=RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            )
        )

        with patch("evaluator.client.BASE_DELAY", 0.01):
            result = asyncio.run(
                evaluate_reply(client, "ticket text", "reply text", semaphore)
            )
        assert result is None
        assert client.responses.parse.call_count == 3  # MAX_RETRIES

    def test_auth_error_fails_immediately(self, semaphore):
        from openai import AuthenticationError

        client = AsyncMock()
        client.responses.parse = AsyncMock(
            side_effect=AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401, headers={}),
                body=None,
            )
        )

        result = asyncio.run(
            evaluate_reply(client, "ticket text", "reply text", semaphore)
        )
        assert result is None
        assert client.responses.parse.call_count == 1  # no retries
