# GAR-RAG: Enhanced Query Processing System

A hybrid query processing system combining Generative AI Retrieval (GAR) with Retrieval-Augmented Generation (RAG) for improved information retrieval.

## Overview

This project implements an advanced query processing pipeline that enhances traditional information retrieval with modern AI techniques. The system processes user queries through multiple stages of analysis and enhancement before retrieving and refining results.

## System Architecture

```
[User Query] → [Query Processor]
                     ↓
[Query Enhancement (GAR)] ← FLAN-T5/Sentence Transformers
                     ↓
[Document Retrieval] ← FAISS + BM25
                     ↓
[Result Refinement (RAG)]
                     ↓
[Result Ranking] → [Enhanced Results]
```

## Key Features

- **Advanced Query Processing**
  - Multi-level tokenization
  - Entity recognition
  - Intent classification
  - Query difficulty estimation

- **Query Enhancement**
  - FLAN-T5 integration for query rewriting
  - Sentence transformers for semantic understanding
  - Intelligent caching mechanism

- **Hybrid Retrieval**
  - FAISS vector store integration
  - BM25 scoring implementation
  - Combined retrieval strategies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gar-rag.git
cd gar-rag
```

2. Install dependencies:
```bash
pip install -e .
```

3. Download required NLTK data:
```python
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

4. Install spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

Basic usage example:
```python
from src.query_processor.preprocessor import QueryProcessor

# Initialize processor
processor = QueryProcessor()

# Process a query
result = processor.process_query("How does Tesla impact climate change?")

# Access processed information
print(f"Difficulty: {result['difficulty']['difficulty_level']}")
print(f"Primary Intent: {result['intent']['primary'][0].value}")
print(f"Entities: {result['entities']}")
```

## Evaluation

The system includes comprehensive evaluation using the BEIR benchmark:
```python
from src.evaluation.beir_evaluator import BEIREvaluator

evaluator = BEIREvaluator(dataset_name="nfcorpus")
stats = evaluator.evaluate_queries()
```

## Project Structure

```
gar-rag/
├── src/
│   ├── query_processor/
│   │   ├── preprocessor.py
│   │   ├── tokenizer.py
│   │   ├── entity_recognizer.py
│   │   ├── intent_classifier.py
│   │   └── difficulty_estimator.py
│   ├── evaluation/
│   │   └── beir_evaluator.py
│   └── utils/
│       └── logger.py
├── tests/
│   └── query_processor/
│       ├── test_processor.py
│       ├── test_tokenizer.py
│       └── test_difficulty_estimator.py
└── setup.py
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Dependencies

- Python 3.8+
- NLTK >= 3.8.1
- spaCy >= 3.7.2
- BEIR (for evaluation)
- Pandas (for analysis)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Authors

- Adamay Mann
- Dhruv Pandoh

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BEIR benchmark for evaluation datasets
- Hugging Face for transformer models
- Facebook Research for FAISS