
"""Extract named entities from individual documents and document batches."""

import spacy
from spacy.language import Language


class NamedEntityExtractor:
    """Extract named entities using a spaCy language model."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """Load the specified spaCy model.

        Args:
            model_name: Name of the installed spaCy language model.
        """
        self.nlp: Language = spacy.load(model_name)

    def extract_entities(self, text: str) -> list[dict[str, str]]:
        """Extract named entities from one document.

        Args:
            text: Input document text.

        Returns:
            A list of dictionaries containing entity text and label.
            Returns an empty list for invalid or empty input.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        doc = self.nlp(text[:100000])

        entities: list[dict[str, str]] = []

        for entity in doc.ents:
            entities.append({
                "text": entity.text,
                "label": entity.label_
            })

        return entities

    def extract_batch(
        self,
        documents: list[dict[str, str]],
        batch_size: int = 32
    ) -> list[list[dict[str, str]]]:
        """Extract named entities from multiple documents.

        Args:
            documents: Documents containing a text field.
            batch_size: Number of documents processed together.

        Returns:
            A list of entity lists, with one list per input document.
        """
        texts = [
            document.get("text", "")[:100000]
            for document in documents
        ]

        results: list[list[dict[str, str]]] = []

        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            entities = [
                {
                    "text": entity.text,
                    "label": entity.label_
                }
                for entity in doc.ents
            ]
            results.append(entities)

        return results
