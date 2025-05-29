# Longitudinal Data Assistant

A comprehensive AI-powered assistant for exploring and analyzing longitudinal survey data using natural language queries. Built with Streamlit, ChromaDB, and Ollama for seamless interaction with complex survey datasets.

## 🌟 Features

### Core Functionality
- **Natural Language Queries**: Ask questions about survey data in plain English
- **Intelligent Search**: Hybrid search combining vector similarity and keyword matching
- **Interactive Chat Interface**: Conversational exploration of survey data
- **Data Explorer**: Browse, filter, and visualize survey questions
- **Variable Analysis**: Detailed explanations and comparisons across waves
- **Smart Caching**: Optimized performance with intelligent cache management

### Technical Highlights
- **Vector Embeddings**: ChromaDB integration for semantic search
- **LLM Integration**: Ollama-powered query processing and response generation
- **Cache Management**: Automatic cache clearing while preserving embeddings
- **Responsive UI**: Modern Streamlit interface with pagination and filtering
- **Data Visualization**: Interactive charts and response distributions

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai)
3. **Required Ollama Models**:
   ```bash
   ollama pull granite3-dense:8b
   ollama pull nomic-embed-text:latest
   ```

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Nishit-Gopani08/Data-Harmonization-LLM.git
   cd Data-Harmonization-LLM
   git checkout wave-function
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**:
   - Place your survey data JSON file in the `data/` directory
   - Update `DATA_PATH` in `app.py` if needed (default: `data/hrs_data_leave_behind.json`)

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Initialize the system**:
   - Click "Initialize System" in the sidebar
   - Wait for data loading and embedding (first-time setup takes longer)

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── data_extraction.py # Survey data scraping utilities
│   ├── data_manager.py    # Data loading and vector database management
│   ├── data_models.py     # Pydantic models for data structures
│   ├── query_processor.py # Query processing and LLM interactions
│   └── utils.py          # Utility functions
├── data/                 # Survey data files (JSON format)
├── cache/               # Query cache (auto-cleared on initialization)
├── chroma_db/           # Vector embeddings (preserved across restarts)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 Configuration

### Model Configuration
Edit the constants in `app.py`:

```python
# Primary LLM model
MODEL_NAME = "granite3-dense:8b"
# Alternative models (uncomment to use):
# MODEL_NAME = "deepseek-r1:8b"
# MODEL_NAME = "llama3-chatqa:latest"
# MODEL_NAME = "mistral-nemo:latest"

# Embedding model
EMBEDDING_MODEL = "nomic-embed-text:latest"

# Data file path
DATA_PATH = "data/hrs_data_leave_behind.json"
```

### Cache Settings
Cache management is configured in `query_processor.py`:
- Cache expiration: 7 days (configurable)
- Cache directory: `./cache`
- Automatic cleanup on initialization

## 💡 Usage Examples

### Chat Interface
- **General questions**: "What variables measure life satisfaction?"
- **Specific variables**: "Explain variable KLB023D"
- **Wave comparisons**: "How did responses change between 2004 and 2016?"
- **Content searches**: "Find questions about retirement planning"

### Data Explorer
- **Browse Questions**: Filter by wave, section, or search terms
- **Variable Explorer**: Deep dive into specific variables
- **Compare Waves**: Analyze changes across time periods

## 🔍 Data Format

The system expects survey data in JSON format with the following structure:

```json
[
  {
    "id": "uuid",
    "variableName": "KLB023D",
    "description": "Variable description",
    "Section": "Section name",
    "Level": "Respondent",
    "Type": "Numeric",
    "Width": "1",
    "Decimals": "0",
    "CAI Reference": "Reference info",
    "question": "Survey question text",
    "response": {
      "1. STRONGLY AGREE": 1234,
      "2. SOMEWHAT AGREE": 5678,
      "...": "..."
    },
    "wave": "2016 Core"
  }
]
```

## 🛠️ Advanced Features

### Smart Cache Management
- **Query caching**: Speeds up repeated searches
- **Embedding preservation**: Keeps vector embeddings across restarts
- **Automatic cleanup**: Clears stale cache while preserving embeddings

### Hybrid Search
- **Vector similarity**: Semantic understanding of queries
- **Keyword matching**: Direct text matching for precision
- **Relevance scoring**: Intelligent ranking of results

### Intent Analysis
- **Query understanding**: Automatically detects user intent
- **Filter extraction**: Identifies relevant constraints
- **Context awareness**: Maintains conversation context

## 🚨 Troubleshooting

### Common Issues

1. **Ollama not running**:
   ```bash
   # Start Ollama service
   ollama serve
   ```

2. **Models not found**:
   ```bash
   # Pull required models
   ollama pull granite3-dense:8b
   ollama pull nomic-embed-text:latest
   ```

3. **Slow initialization**:
   - First-time embedding takes time (normal)
   - Subsequent starts are much faster
   - Use "Initialize System" to clear cache only

4. **Memory issues**:
   - Reduce batch size in `data_manager.py`
   - Use smaller embedding models if needed
   - Increase system memory allocation

### Performance Optimization

1. **Embedding optimization**:
   - Embeddings are generated once and cached
   - Cache clearing preserves embeddings
   - Use balanced sampling for large datasets

2. **Query optimization**:
   - Results are cached for repeated queries
   - Hybrid search provides fast, relevant results
   - Pagination reduces memory usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Requirements

### Python Dependencies
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
chromadb>=0.4.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-ollama>=0.1.0
pydantic>=2.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
```

### System Requirements
- **RAM**: 8GB minimum (16GB recommended for large datasets)
- **Storage**: 2GB+ for embeddings and cache
- **CPU**: Multi-core recommended for embedding generation
- **GPU**: Optional (can accelerate Ollama models)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for local LLM hosting
- **ChromaDB** for vector database capabilities
- **Streamlit** for the web interface
- **LangChain** for LLM orchestration
- **HRS Survey Data** for providing comprehensive longitudinal datasets

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the configuration options

---

**Note**: This is a research tool designed for longitudinal survey data analysis. Ensure compliance with data usage policies and privacy requirements when working with sensitive survey data.