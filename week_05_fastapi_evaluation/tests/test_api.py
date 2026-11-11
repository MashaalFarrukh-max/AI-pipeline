
"""Tests for the FastAPI endpoints and their error handling."""

import pytest
from fastapi.testclient import TestClient

from week_05_fastapi_evaluation.app import main


class FakeRAGPipeline:
    """Provide predictable RAG responses without calling the real pipeline."""

    def answer(self, question: str) -> dict:
        """Return a simulated answer or raise an error for test cases."""
        if question == "trigger value error":
            raise ValueError("Invalid question")

        if question == "trigger server error":
            raise RuntimeError("Simulated server error")

        return {
            "answer": "ISO 10007 provides guidance for configuration management.",
            "sources": ["Source document 1", "Source document 2"],
            "retrieval_seconds": 0.15,
            "generation_seconds": 0.80,
            "total_seconds": 0.95,
        }


@pytest.fixture
def client(monkeypatch):
    """Create a test client using the simulated RAG pipeline."""
    monkeypatch.setattr(main, "RAGPipeline", FakeRAGPipeline)

    with TestClient(main.app) as test_client:
        yield test_client


def test_home_endpoint(client):
    """Check that the home endpoint returns a successful response."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "AI Pipeline RAG API is running"


def test_health_endpoint(client):
    """Check that the health endpoint reports a healthy status."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ask_endpoint(client):
    """Check that the ask endpoint returns an answer and timing details."""
    response = client.post(
        "/ask",
        json={"question": "What is ISO 10007?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What is ISO 10007?"
    assert data["answer"]
    assert len(data["sources"]) == 2
    assert data["retrieval_time"] == 0.15
    assert data["generation_time"] == 0.80


def test_ask_rejects_short_question(client):
    """Check that questions below the minimum length are rejected."""
    response = client.post(
        "/ask",
        json={"question": "Hi"},
    )

    assert response.status_code == 422


def test_ask_rejects_missing_question(client):
    """Check that a missing question produces a validation error."""
    response = client.post("/ask", json={})

    assert response.status_code == 422


def test_ask_returns_400_for_value_error(client):
    """Check that invalid questions produce a 400 response."""
    response = client.post(
        "/ask",
        json={"question": "trigger value error"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid question"


def test_ask_returns_500_for_unexpected_error(client):
    """Check that unexpected errors produce a 500 response."""
    response = client.post(
        "/ask",
        json={"question": "trigger server error"},
    )

    assert response.status_code == 500
    assert "internal error occurred" in response.json()["detail"]
