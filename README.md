## Overview

This project implements a RAG pipeline to process and answer multiple-choice questions using a combination of retrieval mechanisms and language models.

## Getting Started

1. Clone this repository
2. Install dependencies
3. Configure your environment variables
4. Run the application:
   - Start the API server:
     ```bash
     uvicorn src.api.main:app --reload --port 8000
     ```
   - Index your embeddings:
     ```bash
     python src/embeddings/indexer.py
     ```
   - Validate your model:
     ```bash
     python src/validator/validate.py
     ```

## Usage
- Process your question dataset
- Configure retrieval parameters
- Generate and evaluate answers

## Benchmark Results

Benchmarked different models for this system:
- **Claude Sonnet 3.5** achieved **89% accuracy**.
- **GPT-3.5 Turbo** achieved **87% accuracy**.
- **Gemini 1.5** achieved **64% accuracy**.

## Best Practices

For optimal results, Use Claude Sonnet 3.5 as the language model.