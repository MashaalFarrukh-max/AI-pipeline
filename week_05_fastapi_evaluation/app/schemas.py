
"""Define Pydantic request and response schemas for the RAG API."""

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Validate questions submitted to the RAG API."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question to ask the RAG system"
    )


class QuestionResponse(BaseModel):
    """Define the structured response returned by the RAG API."""

    question: str
    answer: str
    sources: list[str]
    retrieval_time: float
    generation_time: float
    total_time: float
