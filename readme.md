# LongitudinalLLM

A natural language interface for querying longitudinal datasets with built-in verification and transformation tracking capabilities.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

LongitudinalLLM is a tool that enables natural language querying of longitudinal datasets with complete transparency about how data has been transformed over time. The system uses local LLMs via Ollama to interpret natural language queries, map them to appropriate dataset schemas, and provide explanations about data lineage and transformations.

![Screenshot of LongitudinalLLM interface](docs/images/interface_screenshot.png)

## Key Features

- 🔍 **Natural Language Querying**: Ask questions about your data in plain English
- 📊 **Longitudinal Data Support**: Track changes in datasets and variables over time
- 🔄 **Transformation Tracking**: Explain how data was modified, combined, and why
- 🔐 **Local LLM Integration**: Uses Ollama for privacy-preserving local inference
- 📝 **Detailed Explanations**: Get clear information about data and transformations

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/download) installed and running

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/longitudinal-llm.git
cd longitudinal-llm
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Pull the required Ollama models:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser at http://localhost:8501

3. Enter natural language queries in the text box:
   - **Data queries**: "Show me all patients with mobility scores above 7"
   - **Verification queries**: "How has the pain variable changed over time?"

## Example Queries

### Data Queries
- "Show me all patient recovery scores for 2018-2020"
- "What's the average mobility score by gender?"
- "Find patients with high recovery scores but low mobility"

### Verification Queries
- "How have columns in the patient demographics dataset changed over time?"
- "Explain how the overall health score was calculated"
- "What transformations were applied to the pain level variable?"

## Architecture

LongitudinalLLM uses a modular architecture:

1. **Query Understanding**: LLM-powered natural language parsing
2. **Schema Mapping**: Vector similarity to map terms to columns
3. **Query Execution**: Dynamic query planning and execution
4. **Verification Tracking**: Dataset lineage and transformation history
5. **Explanation Generation**: LLM-generated natural language explanations

## RAG Architecture

LongitudinalLLM uses Retrieval-Augmented Generation (RAG) to bridge the gap between natural language and data structures:

- 🧠 **Semantic Understanding**: Maps natural language descriptions to actual dataset columns
- 📊 **Schema Vectorization**: Embeds dataset schemas using ChromaDB and Ollama embeddings
- 🔎 **Vector Similarity Search**: Finds the most relevant columns based on semantic similarity
- 🤖 **LLM-Powered Workflow**: Combines retrieval with LLM generation for accurate results

The system embeds descriptions of each column in your datasets, creating a semantic index that enables users to ask questions using everyday language without knowing exact column names.

[Learn more about our RAG architecture](docs/rag_architecture.md)

## Project Structure

```
longitudinal-llm/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_models.py      # Pydantic models for data structures
│   ├── query_processor.py  # LLM query processing logic
│   ├── data_manager.py     # Dataset handling and transformations
│   └── utils.py            # Helper functions
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
│   └── images/             # Screenshots and diagrams
└── README.md               # Project documentation
```

## Customizing for Your Data

To use with your own longitudinal datasets:

1. Modify `src/data_manager.py` to load your data sources
2. Update schema descriptions in `src/query_processor.py`
3. Add custom transformation tracking in `src/data_models.py`

See the [documentation](docs/custom_data.md) for detailed instructions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses [Ollama](https://ollama.com/) for local LLM inference
- Built with [Streamlit](https://streamlit.io/), [LangChain](https://langchain.com/), and [ChromaDB](https://www.trychroma.com/)