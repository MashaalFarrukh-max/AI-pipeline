
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from week_05_fastapi_evaluation.app.logging_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK3_PATH = PROJECT_ROOT / "week_03_rag_llm_integration"

sys.path.insert(0, str(WEEK3_PATH))

from src.rag_pipeline import RAGPipeline

from week_05_fastapi_evaluation.app.schemas import (
    QuestionRequest,
    QuestionResponse,
)

from week_05_fastapi_evaluation.app.logging_config import setup_logging


logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading RAG pipeline")
        app.state.rag_pipeline = RAGPipeline()
        logger.info("RAG pipeline loaded successfully")
        yield
    except Exception:
        logger.exception("Failed to initialize RAG pipeline")
        raise
    finally:
        logger.info("Shutting down FastAPI application")


app = FastAPI(
    title="AI Pipeline RAG API",
    description="A FastAPI service for question answering using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        duration_ms = round(
            (time.perf_counter() - start_time) * 1000, 2
        )

        logger.exception(
            "HTTP request failed",
            extra={
                "request_method": request.method,
                "request_path": request.url.path,
                "status_code": 500,
                "duration_ms": duration_ms,
            },
        )
        raise

    duration_ms = round(
        (time.perf_counter() - start_time) * 1000, 2
    )

    logger.info(
        "HTTP request completed",
        extra={
            "request_method": request.method,
            "request_path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    return response


@app.get("/")
def home():
    return {
        "message": "AI Pipeline RAG API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask", response_model=QuestionResponse)
def ask_question(
    request: QuestionRequest,
    http_request: Request,
):
    pipeline = http_request.app.state.rag_pipeline

    try:
        result = pipeline.answer(request.question)

        sources = result.get("sources", [])

        return QuestionResponse(
            question=request.question,
            answer=result["answer"],
            sources=sources,
            retrieval_time=result["retrieval_seconds"],
            generation_time=result["generation_seconds"],
            total_time=result["total_seconds"],
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:
        logger.exception("RAG question processing failed")

        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred "
                "while processing your question."
            ),
        )
