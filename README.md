# Data-Harmonization-LLM
A system for data harmonization utilizing Large Language Models to transform, normalize, and standardize data from multiple sources.

Overview
This project leverages LLMs to automate the complex task of data harmonization across longitudinal datasets. The system provides:

Natural Language Interface: Ask questions about data in plain English
Intelligent Data Discovery: Automatic mapping of disparate data schemas
Interactive Visualization: Explore and compare data across different waves and sources
Context-Aware Processing: LLM-powered understanding of data semantics and relationships
Features
Data Integration: Harmonize data from multiple sources with different schemas
Query Processing: Natural language query understanding and response generation
Vector Search: Semantic search through survey questions and variables
Wave Comparison: Compare data across different time periods
Data Exploration: Browse, filter, and analyze survey questions and responses
Caching System: Efficient retrieval with multi-level caching for performance
Architecture
The system consists of the following components:

Data Manager: Handles data loading, processing, and vector database operations
Query Processor: Analyzes user queries and generates appropriate responses
Vector Database: Stores embedded representations of survey questions for semantic search
Web Interface: Streamlit-based UI for interactive data exploration
Getting Started
Prerequisites
Python 3.8+
Ollama for local LLM inference
Installation
Clone the repository:

git clone https://github.com/Nishit-Gopani08/Data-Harmonization-LLM.git
cd Data-Harmonization-LLM
Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

pip install -r requirements.txt
Make sure Ollama is running with the required models:

ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
Usage
Run the Streamlit application:

streamlit run app.py
Initialize the system through the sidebar

You can now:

Ask questions using natural language in the chat interface
Browse survey questions in the Data Explorer
Compare variables across different waves
Search for specific variables
Project Structure
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── src/
│   ├── data_extraction.py # Extracts data from external sources
│   ├── data_manager.py    # Manages data loading and vector database
│   ├── data_models.py     # Pydantic models for data structures
│   ├── query_processor.py # Processes natural language queries
│   └── utils.py           # Utility functions
├── data/                  # Data storage (not included in repo)
└── cache/                 # Cache storage for query results
Example Queries
"What questions in the survey ask about depression?"
"How did responses to retirement planning questions change between 2016 and 2020?"
"Find all variables related to healthcare access"
"Show me details about variable KLB023D"
"Compare smoking habits across different waves"
Contributing
Contributions to improve the system are welcome. Please feel free to submit pull requests or open issues for any bugs or feature requests.

License
[License information]

Acknowledgments
This project uses data from Health and Retirement Study (HRS)
Powered by Llama 3 and Nomic embedding models
