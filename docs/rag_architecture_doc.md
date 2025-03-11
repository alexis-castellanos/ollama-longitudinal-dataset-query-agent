# RAG Architecture in LongitudinalLLM

This document explains how Retrieval-Augmented Generation (RAG) is implemented in LongitudinalLLM to provide natural language understanding of dataset schemas.

## Overview

LongitudinalLLM uses a RAG architecture to bridge the gap between natural language queries and structured dataset schemas. This approach allows users to ask questions about their data without needing to know exact column names or dataset structures.

![RAG Architecture Diagram](docs/images/rag_architecture.png)

## Key Components

### 1. Vector Database with ChromaDB

The system uses [ChromaDB](https://www.trychroma.com/) as a vector database to store embeddings of dataset schema information:

```python
# Initialize Chroma DB with schema descriptions
self.vector_db = Chroma.from_texts(
    texts=[item["text"] for item in schema_descriptions],
    metadatas=[item["metadata"] for item in schema_descriptions],
    embedding=self.embeddings,
    persist_directory=self.persist_directory
)
```

Schema descriptions are stored with metadata that links them back to their source dataset and column:

```python
{
    "text": "patient identifier - unique ID for each patient",
    "metadata": {
        "dataset": "patient_demographics",
        "column": "patient_id"
    }
}
```

### 2. Embedding Generation

The system uses [Ollama](https://ollama.com/) with the `nomic-embed-text` model to generate embeddings for schema descriptions:

```python
self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
```

These embeddings capture the semantic meaning of each column, enabling natural language understanding.

### 3. Query Understanding with LLMs

Natural language queries are processed using Ollama's LLMs (default: `llama3.2:latest`):

```python
self.llm = Ollama(model=model_name)
```

The LLM parses the query to identify:
- Query type (data retrieval or verification)
- Requested datasets
- Required columns
- Filtering conditions
- Aggregation operations

### 4. Semantic Schema Mapping

When processing a query, the system maps natural language column descriptions to actual dataset schema using vector similarity search:

```python
# Find matching columns using vector similarity
results = self.vector_db.similarity_search(column_desc, k=1)
```

This allows users to use phrases like "patient age" instead of needing to know the exact column name (e.g., "year_of_birth").

## The RAG Workflow

1. **Initialization Phase**
   - Load all datasets and catalog their schemas
   - Generate natural language descriptions for each column
   - Create embeddings for all descriptions
   - Store embeddings and metadata in ChromaDB

2. **Query Processing Phase**
   - Parse natural language query with LLM
   - Extract key components (datasets, columns, filters, etc.)
   - For each column description, perform semantic search to find matching columns
   - Generate a structured query plan
   - Execute the query against the datasets
   - Use LLM to explain results and transformations

## Benefits of the RAG Approach

- **Natural Language Understanding**: Users can describe data in everyday terms
- **Semantic Matching**: System understands synonyms and related concepts
- **Transparent Explanations**: LLM can explain data lineage and transformations
- **Privacy-Preserving**: All processing happens locally with Ollama

## Configuration and Customization

You can customize the RAG components by:

1. **Using Different Embedding Models**:
   ```python
   query_processor = QueryProcessor(embed_model="your-preferred-model:latest")
   ```

2. **Adding Custom Column Descriptions**:
   - Extend the `setup_vector_db` method in `QueryProcessor` class
   - Add additional descriptions for domain-specific terminology

3. **Adjusting Similarity Thresholds**:
   - Modify the `k` parameter in `similarity_search` calls
   - Add confidence scoring to filter out low-quality matches

## Performance Considerations

- ChromaDB indexes are stored in a temporary directory by default
- For production use, consider:
  - Setting a persistent directory for ChromaDB
  - Pre-computing embeddings for large datasets
  - Using a more powerful embedding model for complex domains

## How It Works in Practice

When a user asks "Show me the average recovery score by gender":

1. The LLM identifies this as a data query requesting:
   - The "recovery_score" column
   - The "gender" column (for grouping)
   - An average aggregation

2. The RAG system:
   - Maps "recovery score" → "recovery_score" in "patient_outcomes"
   - Maps "gender" → "gender" in "patient_demographics_v2"
   - Identifies a need to join these datasets
   - Creates an execution plan that selects, joins, and aggregates

3. The query is executed and results are returned with an explanation.

This seamless process handles complex schema mapping without requiring users to know technical details of the underlying data structure.
