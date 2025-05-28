import json
import uuid
import hashlib
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Updated imports using newer package structure
import chromadb
import pandas as pd
from chromadb.config import Settings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic import ValidationError

# Local imports
from src.data_models import SurveyData, SurveyQuestion, EmbeddingRecord, QueryResult
from src.utils import clean_text, format_response_summary, normalize_query



class DataManager:
    """
    Handles all data operations including loading, processing,
    embedding, and vector database management.
    """

    def __init__(self,
                 data_path: str,
                 embeddings_model: str = "nomic-embed-text",
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "survey_data"):
        """
        Initialize the DataManager.

        Args:
            data_path: Path to the JSON data file
            embeddings_model: Name of the embeddings model to use with Ollama
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
        """
        self.data_path = data_path
        self.embeddings_model = embeddings_model
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name

        # Initialize empty containers
        self.raw_data = []
        self.survey_data = None
        self.df = None

        # Will be initialized later
        self.embedding_client = None
        self.vector_db = None
        self.chroma_client = None
        self.collection = None

        # Cache for vector searches
        self._vector_search_cache = {}

        # Cache for keyword searches
        self._keyword_search_cache = {}

        # Cache for hybrid searches
        self._hybrid_search_cache = {}

        # Cache manager will be assigned externally if needed
        self.cache_manager = None

    def load_data(self) -> SurveyData:
        """Load survey data from JSON file."""
        try:
            with open(self.data_path, 'r') as f:
                self.raw_data = json.load(f)

            # Parse data with Pydantic model
            self.survey_data = SurveyData.from_json(self.raw_data)
            print(f"Successfully loaded {len(self.survey_data.questions)} survey questions")

            # Also create a DataFrame for easier filtering and analysis
            self.df = self._create_dataframe()

            return self.survey_data

        except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
            print(f"Error loading data: {str(e)}")
            raise

    def _create_dataframe(self) -> pd.DataFrame:
        """Convert survey data to pandas DataFrame for analysis."""
        records = []

        for q in self.survey_data.questions:
            # Create a base record with metadata
            base_record = {
                "id": str(q.id),
                "variable_name": q.variable_name,
                "description": q.description,
                "section": q.section,
                "level": q.level,
                "type": q.type,
                "wave": q.wave,
                "question": q.question
            }

            # Add response statistics
            response_counts = {}
            for resp in q.response_items:
                if resp.count is not None:
                    # Clean the response option text
                    option_key = resp.option.replace("Blank.  INAP (Inapplicable); Partial Interview", "INAP")
                    response_counts[option_key] = resp.count

            # Merge the dictionaries
            record = {**base_record, **response_counts}
            records.append(record)

        return pd.DataFrame(records)
    
    def initialize_vector_db(self):
        """Initialize ChromaDB and embedding model."""
        # Initialize embeddings
        try:
            # Set fixed random seed for deterministic behavior
            np.random.seed(42)

            # Initialize embeddings using updated import path
            self.embedding_client = OllamaEmbeddings(
                model=self.embeddings_model,
                base_url="http://localhost:11434"  # Explicitly set base URL
            )

            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

            # Check if collection exists, if not create it
            try:
                # Try to get the collection if it exists
                try:
                    self.collection = self.chroma_client.get_collection(name=self.collection_name)
                    print(f"Found existing collection '{self.collection_name}' with {self.collection.count()} records")
                except Exception as get_error:
                    print(f"Could not get collection: {str(get_error)}")
                    # If getting fails, try creating it
                    try:
                        # If we get a UniqueConstraintError, the collection exists but something is wrong
                        # Try deleting it first and then recreating it
                        try:
                            self.chroma_client.delete_collection(name=self.collection_name)
                            print(f"Deleted existing problematic collection '{self.collection_name}'")
                        except Exception as delete_error:
                            # If delete fails, it probably doesn't exist
                            print(f"Note: Could not delete collection: {str(delete_error)}")

                        # Now try to create it
                        self.collection = self.chroma_client.create_collection(
                            name=self.collection_name,
                            metadata={"description": "Survey question data with embeddings"}
                        )
                        print(f"Created new collection '{self.collection_name}'")
                    except Exception as create_error:
                        print(f"Failed to create collection: {str(create_error)}")
                        # One last attempt - try to get it again
                        self.collection = self.chroma_client.get_collection(name=self.collection_name)
                        print(f"Retrieved collection on final attempt")
            except Exception as e:
                print(f"Fatal error with ChromaDB: {str(e)}")
                raise

            # Initialize LangChain's Chroma wrapper with updated import
            self.vector_db = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_client
            )

            return self.vector_db

        except Exception as e:
            print(f"Error initializing vector database: {str(e)}")
            raise

    # The rest of the class remains the same...

    def prepare_embedding_records(self) -> List[EmbeddingRecord]:
        """Prepare records for embedding."""
        if self.survey_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        embedding_records = []

        for question in self.survey_data.questions:
            # Format a summary of responses
            response_summary = format_response_summary(question.response_items)

            # Create an embedding record
            record = EmbeddingRecord(
                id=str(uuid.uuid4()),  # Generate a new ID for the embedding record
                question_id=str(question.id),
                variable_name=question.variable_name,
                description=question.description,
                question_text=clean_text(question.question),
                response_summary=response_summary,
                wave=question.wave,
                section=question.section,
                level=question.level,
                metadata={
                    "variable_name": question.variable_name,
                    "description": question.description,
                    "wave": question.wave,
                    "section": question.section,
                    "level": question.level
                }
            )
            embedding_records.append(record)

        return embedding_records

    def embed_and_store(self):
        """Embed survey questions and store in ChromaDB."""
        if self.vector_db is None:
            raise ValueError("Vector DB not initialized. Call initialize_vector_db() first.")

        # Use fixed random seed for embedding consistency
        np.random.seed(42)

        # Prepare records
        embedding_records = self.prepare_embedding_records()

        # Prepare data for ChromaDB
        ids = [record.id for record in embedding_records]

        # Create document texts that combine question and response data
        # Include wave information prominently for better wave-specific searches
        documents = [
            f"Wave: {record.wave}\nVariable: {record.variable_name}\nDescription: {record.description}\n"
            f"Question: {record.question_text}\nResponses: {record.response_summary}"
            for record in embedding_records
        ]

        # Prepare metadata
        metadatas = [record.metadata for record in embedding_records]

        # Add to ChromaDB collection
        batch_size = 100  # Process in batches to avoid memory issues
        total_records = len(embedding_records)

        for i in range(0, total_records, batch_size):
            end_idx = min(i + batch_size, total_records)
            print(f"Adding records {i} to {end_idx}...")

            batch_ids = ids[i:end_idx]
            batch_docs = documents[i:end_idx]
            batch_metadata = metadatas[i:end_idx]

            # Add documents to ChromaDB via LangChain's Chroma wrapper
            self.vector_db.add_texts(
                texts=batch_docs,
                metadatas=batch_metadata,
                ids=batch_ids
            )

        print(f"Successfully embedded and stored {total_records} records")

    def _get_cache_key(self, prefix, query_text, filters=None, limit=5):
        """Create a consistent cache key for vector searches."""
        # Normalize the query for consistent caching
        normalized_query = normalize_query(query_text)
        # Include filters and limit in the key
        if filters:
            filters_str = str(sorted([(k, v) for k, v in filters.items()]))
        else:
            filters_str = "None"

        # Create the full key string
        key_str = f"{normalized_query}_{filters_str}_{limit}"
        # Hash it for shorter keys
        return f"{prefix}_{hashlib.md5(key_str.encode()).hexdigest()}"

 

    def query_similar(self,
                    query_text: str,
                    filters: Dict[str, Any] = None,
                    limit: int = 5) -> List[QueryResult]:
        """
        Query the vector database for similar questions.
        """
        if self.vector_db is None:
            raise ValueError("Vector DB not initialized. Call initialize_vector_db() first.")

        # Ensure filters is never None
        if filters is None:
            filters = {}

        # Only pass filter to Chroma if it's not empty
        chroma_filter = filters if filters else None

        # Create a cache key
        cache_key = self._get_cache_key("vector_search", query_text, filters, limit)

        # Check cache
        if cache_key in self._vector_search_cache:
            print(f"Using memory cache for vector search: {query_text}")
            return self._vector_search_cache[cache_key]

        if self.cache_manager:
            cached_results = self.cache_manager.get(cache_key)
            if cached_results:
                print(f"Using disk cache for vector search: {query_text}")
                self._vector_search_cache[cache_key] = cached_results
                return cached_results

        # Set fixed random seed
        np.random.seed(42)

        # Perform the similarity search
        results = self.vector_db.similarity_search_with_score(
            query=query_text,
            k=limit,
            filter=filters
        )

        # Process results
        query_results = []
        for doc, score in results:
            # Extract the question ID from metadata
            metadata = doc.metadata
            variable_name = metadata.get("variable_name")

            # Find the corresponding survey question
            question = next(
                (q for q in self.survey_data.questions if q.variable_name == variable_name),
                None
            )

            if question:
                # Convert similarity score (lower is better) to 0-1 scale (higher is better)
                normalized_score = 1.0 - min(score, 1.0)

                result = QueryResult(
                    question=question,
                    similarity_score=normalized_score,
                    relevance_explanation=f"This question about '{question.description}' is similar to your query."
                )
                query_results.append(result)

        # Cache results in memory
        self._vector_search_cache[cache_key] = query_results

            # Cache to disk if available
        if self.cache_manager:
            self.cache_manager.set(cache_key, query_results)

            return query_results
    
        except Exception as e:
            print(f"Error in vector search: {e}")
            # Fallback to a simpler approach - no filtering
            try:
                print("Trying fallback search without filters")
                results = self.vector_db.similarity_search_with_score(
                    query=query_text,
                    k=limit
                )
                
                # Process results (same as above)
                query_results = []
                for doc, score in results:
                    metadata = doc.metadata
                    variable_name = metadata.get("variable_name")
                    question = next(
                        (q for q in self.survey_data.questions if q.variable_name == variable_name),
                        None
                    )
                    if question:
                        normalized_score = 1.0 - min(score, 1.0)
                        result = QueryResult(
                            question=question,
                            similarity_score=normalized_score,
                            relevance_explanation=f"This question about '{question.description}' is similar to your query."
                        )
                        query_results.append(result)
                        
                return query_results
                
            except Exception as e2:
                print(f"Error in fallback search: {e2}")
                return []
    

    def hybrid_search(self,
                      query_text: str,
                      filters: Dict[str, Any] = None,
                      limit: int = 5) -> List[QueryResult]:
        """
        Perform a hybrid search combining vector similarity and keyword matching.
        """
        # Ensure filters is never None
        if filters is None:
            filters = {}

        # Create a cache key
        cache_key = self._get_cache_key("hybrid_search", query_text, filters, limit)

        # Check in-memory cache first
        if cache_key in self._hybrid_search_cache:
            print(f"Using memory cache for hybrid search: {query_text}")
            return self._hybrid_search_cache[cache_key]

        # Check disk cache if available
        if self.cache_manager:
            cached_results = self.cache_manager.get(cache_key)
            if cached_results:
                print(f"Using disk cache for hybrid search: {query_text}")
                self._hybrid_search_cache[cache_key] = cached_results
                return cached_results

        # Set a higher limit for individual searches to ensure good coverage
        search_limit = min(limit * 2, 20)

        try:
            # Get results from both search methods with proper error handling
            try:
                vector_results = self.query_similar(query_text, filters, search_limit)
            except Exception as e:
                print(f"Vector search error in hybrid search: {e}")
                vector_results = []

            try:
                keyword_results = self.keyword_search(query_text, filters, search_limit)
            except Exception as e:
                print(f"Keyword search error in hybrid search: {e}")
                keyword_results = []

            # If both methods failed, return empty list
            if not vector_results and not keyword_results:
                print("Both search methods failed in hybrid search")
                return []

            # Combine results with deduplication
            seen_variables = set()
            hybrid_results = []

            # Helper function to add results with deduplication
            def add_results(results, hybrid_results, seen_variables, is_keyword=False):
                for result in results:
                    var_name = result.question.variable_name
                    if var_name not in seen_variables:
                        if is_keyword:
                            result.similarity_score = min(1.0, result.similarity_score * 1.05)
                        hybrid_results.append(result)
                        seen_variables.add(var_name)

            # First add exact variable name matches
            exact_var_matches = [r for r in keyword_results if r.similarity_score >= 0.95]
            add_results(exact_var_matches, hybrid_results, seen_variables, True)

            # Then interleave remaining results
            keyword_idx = 0
            vector_idx = 0
            remaining_keyword = [r for r in keyword_results if r.similarity_score < 0.95]

            while len(hybrid_results) < limit:
                # Add a keyword result if available
                if keyword_idx < len(remaining_keyword):
                    add_results([remaining_keyword[keyword_idx]], hybrid_results, seen_variables, True)
                    keyword_idx += 1

                # Break if we've reached the limit
                if len(hybrid_results) >= limit:
                    break

                # Add a vector result if available
                if vector_idx < len(vector_results):
                    add_results([vector_results[vector_idx]], hybrid_results, seen_variables)
                    vector_idx += 1

                # Break if both lists are exhausted
                if keyword_idx >= len(remaining_keyword) and vector_idx >= len(vector_results):
                    break

            # Sort and limit results
            hybrid_results.sort(key=lambda x: x.similarity_score, reverse=True)
            final_results = hybrid_results[:limit]

            # Cache results
            self._hybrid_search_cache[cache_key] = final_results
            if self.cache_manager:
                self.cache_manager.set(cache_key, final_results)

            return final_results

        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []

    def filter_questions(self,
                        filters: Dict[str, Any] = None,
                        limit: int = 100,
                        balanced: bool = True) -> List[SurveyQuestion]:
        """
        Filter questions based on metadata criteria.

        Args:
            filters: Dictionary of filter criteria
            limit: Maximum number of results
            balanced: If True, retrieve balanced samples from each wave

        Returns:
            List of SurveyQuestion objects
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Start with full dataset
        filtered_df = self.df

        # Apply filters if provided
        if filters:
            for key, value in filters.items():
                if key in filtered_df.columns:
                    if isinstance(value, list):
                        # For wave filtering, ensure proper format with " Core" suffix
                        if key == "wave":
                            # Make sure values have " Core" suffix if needed
                            formatted_values = []
                            for item in value:
                                if isinstance(item, str) and " Core" not in item:
                                    formatted_values.append(f"{item} Core")
                                else:
                                    formatted_values.append(item)
                            filtered_df = filtered_df[filtered_df[key].isin(formatted_values)]
                        else:
                            filtered_df = filtered_df[filtered_df[key].isin(value)]
                    else:
                        # For single values, also handle formatting for wave
                        if key == "wave" and isinstance(value, str) and " Core" not in value:
                            value = f"{value} Core"
                        filtered_df = filtered_df[filtered_df[key] == value]

        # Get balanced results if requested
        if balanced and "wave" in filtered_df.columns and limit is not None:
            # Group by wave
            wave_groups = filtered_df.groupby("wave")
            
            # Calculate how many samples per wave
            num_waves = len(wave_groups)
            if num_waves == 0:
                return []
                
            samples_per_wave = max(1, limit // num_waves)
            print(f"Balanced sampling: {samples_per_wave} samples per wave from {num_waves} waves")
            
            # Sample from each wave
            result_df = pd.DataFrame()
            for wave, group in wave_groups:
                if len(group) > 0:
                    # If group is smaller than samples_per_wave, take all; otherwise sample
                    if len(group) <= samples_per_wave:
                        wave_sample = group
                    else:
                        # Use random seed for deterministic sampling
                        np.random.seed(42)
                        wave_sample = group.sample(samples_per_wave)
                    
                    print(f"Sampled {len(wave_sample)} questions from wave {wave}")
                    result_df = pd.concat([result_df, wave_sample])
                
            # If we have fewer than limit, add more samples randomly
            if len(result_df) < limit and len(filtered_df) > len(result_df):
                remaining = limit - len(result_df)
                # Exclude rows already in result_df
                remaining_df = filtered_df[~filtered_df.index.isin(result_df.index)]
                if len(remaining_df) > 0:
                    np.random.seed(42)  # For deterministic sampling
                    additional = remaining_df.sample(min(remaining, len(remaining_df)))
                    result_df = pd.concat([result_df, additional])
                    print(f"Added {len(additional)} additional samples to reach limit")
                    
            filtered_df = result_df
        elif limit is not None:
            # Apply standard limit if not using balanced sampling
            filtered_df = filtered_df.head(limit)

        # Convert back to SurveyQuestion objects
        result_questions = []
        for _, row in filtered_df.iterrows():
            question_id = row["id"]
            # Find the corresponding survey question
            question = next(
                (q for q in self.survey_data.questions if str(q.id) == question_id),
                None
            )
            if question:
                result_questions.append(question)

        return result_questions

    def get_question_by_variable(self, variable_name: str) -> Optional[SurveyQuestion]:
        """Get a question by its variable name."""
        if self.survey_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return next(
            (q for q in self.survey_data.questions if q.variable_name == variable_name),
            None
        )

    def get_unique_values(self, field: str) -> List[str]:
        """Get unique values for a specific field."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if field in self.df.columns:
            return self.df[field].unique().tolist()
        return []

    def get_waves_summary(self) -> Dict[str, int]:
        """Get a summary of questions by wave."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return self.df["wave"].value_counts().to_dict()

    def get_sections_summary(self) -> Dict[str, int]:
        """Get a summary of questions by section."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return self.df["section"].value_counts().to_dict()