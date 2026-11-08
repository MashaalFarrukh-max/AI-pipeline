
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question to ask the RAG system"
    )


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    retrieval_time: float
    generation_time: float
    total_time: float
