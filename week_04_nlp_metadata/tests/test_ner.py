import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK4_DIR = PROJECT_ROOT / "week_04_nlp_metadata"

sys.path.insert(0, str(WEEK4_DIR))

from src.ner import NamedEntityExtractor


def test_extractor_loads_model():
    extractor = NamedEntityExtractor()
    assert extractor.nlp is not None


def test_extract_entities_returns_entities():
    extractor = NamedEntityExtractor()

    text = "Microsoft was founded by Bill Gates in the United States."

    entities = extractor.extract_entities(text)

    assert len(entities) > 0
    assert all("text" in entity for entity in entities)
    assert all("label" in entity for entity in entities)


def test_extract_entities_handles_empty_text():
    extractor = NamedEntityExtractor()

    assert extractor.extract_entities("") == []
    assert extractor.extract_entities(None) == []


def test_extract_batch_returns_one_result_per_document():
    extractor = NamedEntityExtractor()

    documents = [
        {"text": "Microsoft is based in the United States."},
        {"text": "Apple was founded by Steve Jobs."}
    ]

    results = extractor.extract_batch(documents)

    assert len(results) == 2
    assert all(isinstance(entities, list) for entities in results)