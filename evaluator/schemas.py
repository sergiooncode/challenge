"""Pydantic models and prompt definitions for ticket reply evaluation."""

from pydantic import BaseModel, Field

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


class Evaluation(BaseModel):
    """Structured evaluation result from GPT-4o."""

    content_score: int = Field(ge=1, le=5, description="Content score from 1 to 5")
    content_explanation: str = Field(description="Brief explanation of the content score")
    format_score: int = Field(ge=1, le=5, description="Format score from 1 to 5")
    format_explanation: str = Field(description="Brief explanation of the format score")
