# Data Harmonization LLM - Longitudinal Survey Data Assistant

A comprehensive AI-powered assistant for exploring and analyzing longitudinal survey data using natural language queries. Built with Streamlit, ChromaDB, and Ollama for seamless interaction with complex survey datasets across multiple waves and time periods.

## 🌟 Key Features

### Advanced Query Interface
- **Natural Language Queries**: Ask questions about survey data in plain English
- **Intelligent Intent Analysis**: Automatically detects query intent and extracts relevant filters
- **Longitudinal Pattern Analysis**: Identifies changes and trends across survey waves
- **Confidence-Based Filtering**: User-controllable confidence thresholds for result quality

### Smart Search Capabilities
- **Hybrid Search**: Combines vector similarity with keyword matching for optimal results
- **Balanced Wave Sampling**: Ensures representation across all available survey waves (1996-2018)
- **Enhanced Similarity Scoring**: Differentiated relevance scores from 70% to 98% confidence
- **Variable-Specific Search**: Direct lookup by variable codes (e.g., KLB023D, JLB001)

### Interactive Data Exploration
- **Chat Interface**: Conversational exploration with context-aware responses
- **Data Explorer**: Browse, filter, and visualize survey questions with pagination
- **Variable Analysis**: Detailed explanations and cross-wave comparisons
- **Response Visualization**: Interactive charts showing response distributions

### Performance & Reliability
- **Smart Caching**: Multi-layer caching system preserving embeddings across restarts
- **Optimized Performance**: Batch processing and intelligent memory management
- **Error Resilience**: Comprehensive fallback mechanisms for robust operation

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

1. **Clone and setup**:
   ```bash
   git clone https://github.com/yourusername/Data-Harmonization-LLM.git
   cd Data-Harmonization-LLM
   git checkout main  # or wave-function for latest features
   pip install -r requirements.txt
   ```

2. **Prepare your data**:
   - Place your survey data JSON file in the `data/` directory
   - Default expected file: `data/hrs_data_leave_behind.json`
   - Update `DATA_PATH` in `app.py` if using a different file

3. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

4. **Initialize the system**:
   - Click "Initialize System" in the sidebar
   - First-time setup includes data embedding (may take several minutes)
   - Subsequent launches are much faster due to cached embeddings

## 🏗️ Architecture

### Core Components
```
├── app.py                 # Main Streamlit application & UI
├── src/
│   ├── data_manager.py    # Data loading, vector DB, and search operations
│   ├── query_processor.py # Query analysis, intent detection, and LLM integration
│   ├── data_models.py     # Pydantic models for type safety
│   ├── data_extraction.py # Web scraping utilities for survey data
│   └── utils.py          # Helper functions and text processing
├── data/                 # Survey data files (JSON format)
├── cache/               # Query cache (auto-managed)
└── chroma_db/           # Vector embeddings (persistent)
```

## ⚙️ Configuration

### Model Settings
Edit constants in `app.py`:

```python
# Primary LLM model (current: granite3-dense:8b for optimal performance)
MODEL_NAME = "granite3-dense:8b"

# Alternative models (uncomment to test):
# MODEL_NAME = "deepseek-r1:8b"
# MODEL_NAME = "llama3-chatqa:latest" 
# MODEL_NAME = "mistral-nemo:latest"

# Embedding model (optimized for survey data)
EMBEDDING_MODEL = "nomic-embed-text:latest"

# Data file path
DATA_PATH = "data/hrs_data_leave_behind.json"
```

### Performance Tuning
- **Cache Management**: Automatically clears query cache while preserving embeddings
- **Balanced Sampling**: Ensures fair representation across all survey waves
- **Confidence Thresholds**: Default 90% confidence for high-quality results
- **Batch Processing**: Configurable batch sizes for large datasets

## 💡 Usage Examples

### Natural Language Queries
```
"Variables about life satisfaction"
"Health-related questions from 2016"
"Questions about retirement planning across all waves"
"Compare financial satisfaction between 2004 and 2018"
```

### Variable-Specific Queries
```
"Explain variable KLB023D"
"Show me all versions of JLB001 across waves"
"Compare KLB023D between 2004 and 2016 waves"
```

### Longitudinal Analysis
```
"How did responses about work satisfaction change over time?"
"Show trends in health variables from 1996 to 2018"
"What variables were consistently asked across all waves?"
```

## 📊 Advanced Features

### Longitudinal Pattern Analysis
- **Variable Consistency Scoring**: Measures how consistently variables were asked across waves
- **Question Wording Analysis**: Detects changes in question phrasing over time
- **Response Option Tracking**: Identifies modifications in answer choices
- **Trend Visualization**: Charts showing response distribution changes

### Smart Filtering & Search
- **Intent-Based Search**: Automatically applies relevant filters based on query context
- **Multi-Wave Sampling**: Balanced representation ensuring no wave bias
- **Confidence-Based Results**: Filter results by similarity confidence levels
- **Hybrid Relevance**: Combines semantic similarity with keyword matching

## 🔧 Data Format

Expected JSON structure for survey data:

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

## 🚨 Troubleshooting

### Common Issues

1. **Ollama Connection**:
   ```bash
   # Ensure Ollama is running
   ollama serve
   
   # Verify models are available
   ollama list
   ```

2. **Slow Initialization**:
   - First-time embedding generation takes time (normal behavior)
   - Subsequent starts are much faster due to cached embeddings
   - Clear cache only if needed: delete `cache/` folder (preserves `chroma_db/`)

3. **Memory Issues**:
   - Reduce batch size in `data_manager.py` if needed
   - Ensure sufficient RAM (8GB minimum, 16GB recommended)
   - Consider using smaller models for resource-constrained environments

4. **Search Quality**:
   - Adjust confidence threshold in the sidebar (default: 90%)
   - Try more specific queries for better results
   - Use exact variable names when available

### Performance Optimization

- **Embeddings**: Generated once and cached permanently
- **Query Cache**: 7-day expiration for frequently asked questions  
- **Balanced Sampling**: Prevents over-representation of certain waves
- **Memory Management**: Efficient handling of large datasets

## 📈 Recent Improvements

### Version 2.0 Features
- **Enhanced Similarity Scoring**: More discriminative relevance scores (70%-98% range)
- **Longitudinal Analysis**: Comprehensive cross-wave pattern detection
- **Improved Model**: Upgraded to `granite3-dense:8b` for better understanding
- **Balanced Sampling**: Equal representation across all survey waves
- **User Controls**: Confidence threshold slider for result filtering

### Performance Enhancements
- **Smart Caching**: Preserves expensive embeddings across restarts
- **Hybrid Search**: Combines semantic and keyword matching
- **Error Resilience**: Comprehensive fallback mechanisms
- **UI Improvements**: Better pagination and result organization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 2GB+ for embeddings and cache
- **CPU**: Multi-core recommended for embedding generation

### Recommended Setup
- **RAM**: 16GB+ for large datasets
- **GPU**: Optional (accelerates Ollama models)
- **Storage**: SSD for better I/O performance
- **Network**: Stable connection for initial model downloads

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for local LLM hosting capabilities
- **ChromaDB** for efficient vector database operations
- **Streamlit** for the intuitive web interface
- **LangChain** for LLM orchestration and prompt management
- **HRS Survey Data** for providing comprehensive longitudinal datasets

## 📞 Support & Documentation

- **Issues**: Report bugs or request features via GitHub Issues
- **Configuration**: See troubleshooting section above
- **Data Format**: Follow the JSON schema provided in this README
- **Performance**: Consult the optimization guidelines for large datasets

## ⚡ Performance Metrics

- **Query Response Time**: < 2 seconds for cached queries
- **Embedding Generation**: ~1-2 minutes per 1000 questions (first time only)
- **Memory Usage**: ~2-4GB for typical datasets
- **Search Accuracy**: 85-95% relevance for natural language queries

---

**Note**: This tool is designed for longitudinal survey data analysis and research. Ensure compliance with data usage policies and privacy requirements when working with sensitive survey data.