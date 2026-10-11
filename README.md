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