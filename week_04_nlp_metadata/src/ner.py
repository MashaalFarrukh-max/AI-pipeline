
import spacy


class NamedEntityExtractor:
    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def extract_entities(self, text):
        """Extract named entities from one document."""
        if not isinstance(text, str) or not text.strip():
            return []

        doc = self.nlp(text[:100000])

        entities = []

        for entity in doc.ents:
            entities.append({
                "text": entity.text,
                "label": entity.label_
            })

        return entities

    def extract_batch(self, documents, batch_size=32):
        """Extract entities from multiple documents."""
        texts = [
            document.get("text", "")[:100000]
            for document in documents
        ]

        results = []

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
