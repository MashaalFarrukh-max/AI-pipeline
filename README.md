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

At the end of Week 3, 20 tests passed across Weeks 1, 2, and 3. The latest complete internship test suite passes all 29 tests across Weeks 1–4.

## Error Handling and Limitations

- Empty questions are rejected.
- Missing API configuration raises an error.
- Groq API exceptions are logged and raised to the caller.
- Unsupported questions receive a fallback response when retrieved context does not meet the relevance threshold.
- Retrieval quality depends on the documents available in the collection and the configured distance threshold.
- The project currently uses the CPU for embedding generation.

## Week 3 Outcome

A working retrieval-augmented generation pipeline was implemented and tested. The system retrieves Wikipedia chunks, generates context-grounded responses through Groq, avoids answering when retrieved context is insufficient, and records request latency.













 

# Week 4: Topic Modeling, Named Entity Recognition, and NLP Metadata

## Overview

This week extends the existing Wikipedia retrieval pipeline by adding Natural Language Processing (NLP) metadata. Latent Dirichlet Allocation (LDA) was used for topic modeling, while spaCy was used for Named Entity Recognition (NER).

The extracted metadata was integrated into the existing ChromaDB vector database to support topic-filtered semantic retrieval.

## Objectives

- Apply topic modeling to the cleaned Wikipedia corpus.
- Discover and label major topics using LDA.
- Extract named entities using spaCy.
- Evaluate NER against a manually labeled sample.
- Enrich existing ChromaDB chunks with topic and entity metadata.
- Implement topic-filtered semantic retrieval.
- Add automated tests for the new modules.

## Technologies Used

- Python 3.11
- scikit-learn
- Latent Dirichlet Allocation (LDA)
- spaCy (`en_core_web_sm`)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- ChromaDB
- pandas
- pytest

## Topic Modeling

LDA was trained on 4,987 cleaned Wikipedia articles.

Eight topics were generated and assigned descriptive labels:

| Topic ID | Topic Label |
|---|---|
| 0 | Music |
| 1 | Geography and Population |
| 2 | Film and Entertainment |
| 3 | Technology and General Concepts |
| 4 | Military and History |
| 5 | Science, Nature and Culture |
| 6 | People, Education and Sports |
| 7 | Places and Urban Geography |

The model generated topic keywords and assigned each article its most probable topic.

## Named Entity Recognition

spaCy's `en_core_web_sm` model was used to extract named entities from the Wikipedia articles.

Detected entity categories include:

- PERSON
- ORG
- GPE
- DATE
- CARDINAL
- EVENT
- WORK_OF_ART

A total of 138,684 entity detections were recorded across 4,987 articles.

## NER Evaluation

A manually labeled sample of 10 articles was used for an initial evaluation.

| Metric | Result |
|---|---:|
| Precision | 18.24% |
| Recall | 41.43% |
| F1-score | 25.33% |

**Evaluation limitation:** The manual annotations may not include every valid entity in the selected articles. Therefore, unmatched model predictions can be counted as false positives even when they are valid entities. These scores should be treated as preliminary agreement measurements, not definitive model accuracy.

## ChromaDB Metadata Enrichment

The existing ChromaDB collection was enriched without regenerating embeddings.

A total of 13,500 chunks were successfully updated.

The following metadata fields were added:

- `topic_id`
- `topic_label`
- `topic_probability`
- `topic_keywords`
- `entities`
- `entity_types`

Existing metadata fields, including `document_id`, `source`, and `title`, were preserved.

## Filtered Semantic Retrieval

A new script, `scripts/filtered_retrieval.py`, was implemented.

The retrieval process:

1. Accepts a search query and selected topic ID.
2. Converts the query into an embedding using Sentence Transformers.
3. Applies a ChromaDB topic filter.
4. Retrieves the five most semantically similar chunks within that topic.
5. Displays the article title, topic information, distance, and extracted entities.

Two test queries were performed:

- `ISO standards` — Technology and General Concepts.
- `music albums` — Music.

Both searches successfully returned results belonging to their selected topics.

## Scripts and Source Code

- `src/topic_modeling.py` — text cleaning, LDA training, and topic assignment.
- `src/ner.py` — named entity extraction using spaCy.
- `scripts/run_topic_modeling.py` — runs LDA and saves topic results.
- `scripts/run_ner.py` — extracts entities from the corpus.
- `scripts/create_labeled_sample.py` — creates a manually labeled sample.
- `scripts/evaluate_ner.py` — evaluates NER predictions.
- `scripts/enrich_chroma_metadata.py` — adds NLP metadata to existing ChromaDB chunks.
- `scripts/filtered_retrieval.py` — performs topic-filtered semantic retrieval.
- `tests/test_topic_modeling.py` — topic modeling tests.
- `tests/test_ner.py` — NER tests.

## Running Week 4

From the repository root, activate the virtual environment and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the spaCy English model if it is not already installed:

```powershell
python -m spacy download en_core_web_sm
```

Run topic modeling:

```powershell
python week_04_nlp_metadata/scripts/run_topic_modeling.py
```

Run NER:

```powershell
python week_04_nlp_metadata/scripts/run_ner.py
```

Evaluate NER:

```powershell
python week_04_nlp_metadata/scripts/evaluate_ner.py
```

Enrich ChromaDB metadata:

```powershell
python week_04_nlp_metadata/scripts/enrich_chroma_metadata.py
```

Run filtered retrieval:

```powershell
python week_04_nlp_metadata/scripts/filtered_retrieval.py
```

Run Week 4 tests:

```powershell
python -m pytest week_04_nlp_metadata/tests -v
```

## Testing Results

Week 4 automated tests:

- Topic modeling tests: 5 passed.
- NER tests: 4 passed.
- Total Week 4 tests: 9 passed.

The complete internship test suite contains **29 passing tests** across all four weeks.

## Week 4 Outcome

LDA topic modeling and spaCy-based NER were integrated into the existing retrieval pipeline. The 13,500 ChromaDB chunks were enriched with NLP metadata, and topic-filtered semantic retrieval was successfully tested.

This extends the original semantic retrieval system with topic-aware filtering and named entity information.
