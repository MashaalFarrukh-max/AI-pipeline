# Week 1: Environment, Data Ingestion & Preprocessing

## 1. Project Overview

This project implements the first stage of an AI text-processing pipeline. It sets up the Python environment, collects a real-world text dataset from Wikipedia, cleans the text, and tests the preprocessing functions.

The objective is to prepare a clean text corpus that can be used for future Natural Language Processing (NLP) and machine learning tasks.

## 2. Technologies Used

- Python 3.11
- pandas
- Sentence Transformers
- ChromaDB
- spaCy
- PyTorch
- BeautifulSoup
- langdetect
- pytest
- Git

## 3. Project Structure

```text
week_01_environment_data_preprocessing/
├── data/
│   ├── raw/
│   │   └── wikipedia_articles.jsonl
│   └── processed/
│       └── clean_corpus.jsonl
├── scripts/
│   ├── verify_env.py
│   ├── collect_data.py
│   └── process_data.py
├── src/
│   └── preprocessing.py
├── tests/
│   └── test_preprocessing.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 4. Environment Setup

Python 3.11 was used for this project. A Python virtual environment was created to keep the project's dependencies separate from other Python installations.

Activate the environment from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the environment verification script:

```powershell
python week_01_environment_data_preprocessing/scripts/verify_env.py
```

The script checks the required libraries and reports whether CUDA/GPU support is available. CPU execution is possible when CUDA is unavailable.

## 5. Dataset Collection

The dataset was collected from Wikipedia using its API.

- Source: Wikipedia
- Raw documents collected: 5,000 unique documents
- File format: JSON Lines (`.jsonl`)
- Raw dataset path: `data/raw/wikipedia_articles.jsonl`

Each record contains an article ID, title, text, source, and URL.

Run the collection script from the project root:

```powershell
python week_01_environment_data_preprocessing/scripts/collect_data.py
```

The raw dataset is kept separate from the processed dataset so that the original text remains available.

## 6. Text Preprocessing

The preprocessing functions are implemented in `src/preprocessing.py`.

The pipeline includes:

1. HTML stripping: removes HTML tags from text.
2. Unicode normalization: standardizes Unicode characters using NFKC normalization.
3. Whitespace cleanup: removes unnecessary spaces and line breaks.
4. Language filtering: identifies English-language documents.
5. Text validation: excludes documents that are too short and removes duplicate text.

The `clean_text()` function combines the text-cleaning operations.

Run the preprocessing script:

```powershell
python week_01_environment_data_preprocessing/scripts/process_data.py
```

The processed dataset is saved to:

`data/processed/clean_corpus.jsonl`

## 7. Preprocessing Results

The preprocessing script reported the following results:

| Measurement | Result |
|---|---:|
| Raw documents read | 5,000 |
| Clean documents saved | 4,987 |
| Documents excluded | 13 |

Documents may be excluded if they are too short, are not identified as English, or duplicate text already exists.

## 8. Unit Testing

Unit tests are located in `tests/test_preprocessing.py`.

The tests cover:

- HTML stripping
- Unicode normalization
- Whitespace cleanup
- English-language detection
- Combined text cleaning

Run the tests from the project root:

```powershell
python -m pytest week_01_environment_data_preprocessing/tests/test_preprocessing.py -v
```

All five preprocessing tests passed during verification.

## 9. Data and Git Management

Raw and processed datasets should not be committed to Git because they are generated data files. The `.gitignore` configuration excludes data files, Python cache files, virtual environments, and environment-variable files such as `.env`.

The source code, tests, requirements file, and README should remain tracked by Git.

## 10. Conclusion

Week 1 established the Python environment, collected 5,000 unique Wikipedia documents, implemented modular text-cleaning functions, generated a processed corpus containing 4,987 documents, and verified the preprocessing functions with five passing unit tests.

The resulting corpus is prepared for subsequent NLP pipeline development.









## Week 2: Chunking, Embeddings, and Retrieval

### Overview
This week, a text retrieval pipeline was implemented using the cleaned Wikipedia corpus generated in Week 1. The pipeline splits documents into overlapping chunks, generates embeddings using Sentence Transformers, stores them in ChromaDB, and retrieves relevant text using semantic similarity.

### Technologies
- Python
- Sentence Transformers (`all-MiniLM-L6-v2`)
- ChromaDB
- NumPy
- Pytest

### Implementation
- Implemented character-based text chunking with a chunk size of 500 characters and an overlap of 50 characters.
- Generated embeddings for 13,500 text chunks.
- Stored the chunks and embeddings in a persistent ChromaDB vector database using cosine distance.
- Implemented interactive semantic search to retrieve the five most relevant chunks for a query.
- Created a retrieval benchmark script to measure query retrieval latency.
- Added unit tests for chunking and vector database operations.

### Performance Results
- Total chunks stored: 13,500
- Embedding generation time: 383.26 seconds
- Number of benchmark queries: 5
- Average retrieval latency: 90.727 milliseconds
- Unit tests: 10 passed

### Scripts
- `scripts/ingest_data.py` — chunks the cleaned corpus, generates embeddings, and populates ChromaDB.
- `scripts/semantic_search.py` — performs interactive semantic search.
- `scripts/benchmark_retrieval.py` — measures retrieval latency and saves results to CSV.

### Source Code
- `src/chunking.py` — text chunking.
- `src/embeddings.py` — embedding model and performance measurement.
- `src/vector_store.py` — ChromaDB storage and retrieval.
- `tests/` — unit tests for chunking and vector storage.

### Generated Outputs
- `ingestion_log.txt` — records ingestion and embedding performance.
- `retrieval_benchmark.csv` — contains retrieval latency results.
- `chroma_db/` — local persistent vector database, regenerated by running the ingestion script.

### How to Run
From the repository root, with the virtual environment activated:

1. Install dependencies using `pip install -r requirements.txt`.
2. Run `python .\week_02_chunking_embeddings_retrieval\scripts\ingest_data.py`.
3. Run `python .\week_02_chunking_embeddings_retrieval\scripts\semantic_search.py`.
4. Run `python .\week_02_chunking_embeddings_retrieval\scripts\benchmark_retrieval.py`.
5. Run tests using `python -m pytest .\week_02_chunking_embeddings_retrieval\tests -v`.

The cleaned Week 1 corpus must be available locally before ingestion. The model is downloaded automatically on the first run.











# Week 3: RAG and LLM Integration

## Overview

This week integrates a Large Language Model (LLM) into the retrieval pipeline developed in Week 2. Groq API is used to generate answers based on relevant information retrieved from the Wikipedia document collection stored in ChromaDB.

## Objectives

- Integrate the Groq API with the existing retrieval pipeline.
- Retrieve relevant document chunks using sentence-transformer embeddings and ChromaDB.
- Generate context-grounded answers using prompt engineering.
- Reduce hallucinations by rejecting questions when relevant context is unavailable.
- Handle empty questions and API failures.
- Record retrieval, generation, and total request latency.
- Write automated unit tests for the RAG pipeline.

## Technologies Used

- Python 3.11
- Groq API
- `openai/gpt-oss-20b`
- Sentence Transformers: `all-MiniLM-L6-v2`
- ChromaDB
- python-dotenv
- pytest

## Project Structure

- `src/rag_pipeline.py`: RAG pipeline, context filtering, prompt construction, answer generation, and latency logging.
- `scripts/test_groq.py`: Groq API connection test.
- `tests/test_rag_pipeline.py`: Automated tests for the RAG pipeline.
- `rag.log`: Runtime logs generated by the pipeline.

## How It Works

1. The user submits a question.
2. Sentence Transformers converts the question into an embedding.
3. ChromaDB retrieves semantically similar document chunks.
4. A distance threshold filters out insufficiently relevant chunks.
5. Relevant context is sent to Groq with instructions to answer only from the supplied documents.
6. If relevant context is unavailable, the pipeline returns a fallback response without calling the LLM.
7. Retrieval, generation, and total execution times are logged.

## Configuration

Create a `.env` file in the repository root and add your own Groq API key:

`GROQ_API_KEY=your_groq_api_key_here`

Do not commit the real API key to GitHub. The root `.gitignore` excludes environment files.

## Installation

Activate the project's virtual environment and install dependencies from the repository root:

`pip install -r requirements.txt`

## Running the Pipeline

From the repository root, run:

`python week_03_rag_llm_integration/src/rag_pipeline.py`

Enter a question when prompted. Type `exit` to quit.

## Running Tests

Run the complete internship test suite from the repository root:

`pytest -v`

Latest verification: 20 tests passed across Weeks 1, 2, and 3.

## Error Handling and Limitations

- Empty questions are rejected.
- Missing API configuration raises an error.
- Groq API exceptions are logged and raised to the caller.
- Unsupported questions receive a fallback response when retrieved context does not meet the relevance threshold.
- Retrieval quality depends on the documents available in the collection and the configured distance threshold.
- The project currently uses the CPU for embedding generation.

## Week 3 Outcome

A working retrieval-augmented generation pipeline was implemented and tested. The system retrieves Wikipedia chunks, generates context-grounded responses through Groq, avoids answering when retrieved context is insufficient, and records request latency.