import json
import re
import random
import numpy as np
import hashlib
import os
import pickle
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field

from src.data_manager import DataManager
# Local imports
from src.data_models import QueryResult, UserQuery, SurveyQuestion
from src.utils import create_structured_context, format_filters_description, normalize_query

# Set fixed random seeds for deterministic behavior
random.seed(42)
np.random.seed(42)


class CacheManager:
    """Manages persistent caching to disk with expiration functionality."""

    def __init__(self, cache_dir="./cache", expiration_days=7):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory to store cache files
            expiration_days: Number of days after which cache entries expire (0 for no expiration)
        """
        self.cache_dir = cache_dir
        self.expiration_days = expiration_days
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, key):
        """Get the file path for a cache key."""
        return os.path.join(self.cache_dir, f"{key}.pickle")

    def get(self, key):
        """Retrieve a value from cache if it exists and isn't expired."""
        path = self.get_cache_path(key)
        if not os.path.exists(path):
            return None

        # Check for expiration
        if self.expiration_days > 0:
            modified_time = datetime.fromtimestamp(os.path.getmtime(path))
            if datetime.now() - modified_time > timedelta(days=self.expiration_days):
                os.remove(path)  # Remove expired cache
                return None

        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Cache retrieval error: {e}")
            return None

    def set(self, key, value):
        """Store a value in the cache."""
        path = self.get_cache_path(key)
        try:
            with open(path, 'wb') as f:
                pickle.dump(value, f)
            return True
        except Exception as e:
            print(f"Cache storage error: {e}")
            return False


# Intent analysis models
class QueryIntent(BaseModel):
    """Model for representing the detected intent of a user query."""
    primary_intent: str = Field(
        ...,
        description="The primary intent of the query (search, filter, compare, analyze, explain, summarize)"
    )
    secondary_intent: Optional[str] = Field(
        None,
        description="Optional secondary intent of the query"
    )
    filter_criteria: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted filter criteria (wave, section, variable name, etc.)"
    )
    target_variables: List[str] = Field(
        default_factory=list,
        description="List of specifically mentioned variable names"
    )
    time_periods: List[str] = Field(
        default_factory=list,
        description="List of time periods or waves mentioned"
    )
    analysis_type: Optional[str] = Field(
        None,
        description="Type of analysis requested (trend, correlation, significance, etc.)"
    )
    confidence: float = Field(
        ...,
        description="Confidence score (0-1) for the intent detection"
    )


class ProcessedQuery(BaseModel):
    """Complete processed query with intent and results."""
    original_query: str
    intent: QueryIntent
    results: List[QueryResult] = []
    answer: str = ""
    context_used: List[str] = []


class LongitudinalAnalyzer:
    """Analyzes changes in variables across waves for longitudinal studies."""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        
    def find_longitudinal_variables(self, query_results: List[QueryResult]) -> Dict[str, List[SurveyQuestion]]:
        """
        Group similar variables across waves to identify longitudinal series.
        
        Args:
            query_results: Results from a search query
            
        Returns:
            Dictionary mapping concept names to lists of questions across waves
        """
        # Extract unique concepts by analyzing variable descriptions and questions
        concept_groups = {}
        
        for result in query_results:
            question = result.question
            
            # Create a concept key by cleaning the description
            concept_key = self._extract_concept_key(question)
            
            if concept_key not in concept_groups:
                concept_groups[concept_key] = []
            
            concept_groups[concept_key].append(question)
        
        # Sort each group by wave year
        for concept_key in concept_groups:
            concept_groups[concept_key].sort(key=lambda q: self._extract_wave_year(q.wave))
            
        return concept_groups
    
    def _extract_concept_key(self, question: SurveyQuestion) -> str:
        """Extract a standardized concept key from question description."""
        # Remove common prefixes and clean description
        desc = question.description.lower()
        
        # Remove common survey prefixes
        desc = re.sub(r'^(q\d+[a-z]?\.|how much|whether|if)', '', desc).strip()
        
        # Extract core concept (e.g., "satisfied with life")
        core_concept = desc.split('.')[0].strip()
        
        return core_concept
    
    def _extract_wave_year(self, wave_str: str) -> int:
        """Extract year from wave string."""
        match = re.search(r'(\d{4})', wave_str)
        return int(match.group(1)) if match else 0
    
    def analyze_longitudinal_changes(self, concept_groups: Dict[str, List[SurveyQuestion]]) -> Dict[str, Dict]:
        """
        Analyze what changed across waves for each longitudinal concept.
        
        Returns:
            Dictionary with analysis results for each concept
        """
        analysis_results = {}
        
        for concept_key, questions in concept_groups.items():
            if len(questions) < 2:
                continue
                
            analysis = {
                'concept': concept_key,
                'waves_covered': [q.wave for q in questions],
                'variable_names': [q.variable_name for q in questions],
                'changes': self._detect_changes(questions),
                'consistency_score': self._calculate_consistency_score(questions),
                'recommendations': self._generate_recommendations(questions)
            }
            
            analysis_results[concept_key] = analysis
            
        return analysis_results
    
    def _detect_changes(self, questions: List[SurveyQuestion]) -> Dict[str, Any]:
        """Detect specific changes across waves."""
        changes = {
            'variable_naming': self._analyze_variable_naming_changes(questions),
            'question_wording': self._analyze_question_wording_changes(questions),
            'response_options': self._analyze_response_option_changes(questions),
            'metadata_changes': self._analyze_metadata_changes(questions)
        }
        
        return changes
    
    def _analyze_variable_naming_changes(self, questions: List[SurveyQuestion]) -> Dict:
        """Analyze how variable names changed across waves."""
        var_names = [q.variable_name for q in questions]
        
        # Extract patterns
        prefixes = [re.match(r'^([A-Z]+)', name).group(1) if re.match(r'^([A-Z]+)', name) else '' for name in var_names]
        numbers = [re.search(r'(\d+)', name).group(1) if re.search(r'(\d+)', name) else '' for name in var_names]
        suffixes = [re.search(r'([A-Z])$', name).group(1) if re.search(r'([A-Z])$', name) else '' for name in var_names]
        
        return {
            'variable_names': var_names,
            'prefix_pattern': prefixes,
            'number_pattern': numbers,
            'suffix_pattern': suffixes,
            'naming_consistent': len(set(prefixes)) == 1 and len(set(numbers)) == 1,
            'pattern_description': self._describe_naming_pattern(prefixes, numbers, suffixes)
        }
    
    def _analyze_question_wording_changes(self, questions: List[SurveyQuestion]) -> Dict:
        """Analyze changes in question wording."""
        question_texts = [q.question for q in questions]
        
        # Calculate similarity between consecutive waves
        similarities = []
        for i in range(1, len(question_texts)):
            similarity = self._calculate_text_similarity(question_texts[i-1], question_texts[i])
            similarities.append(similarity)
        
        # Detect specific changes
        changes_detected = []
        reference_text = question_texts[0]
        
        for i, text in enumerate(question_texts[1:], 1):
            if text != reference_text:
                changes_detected.append({
                    'wave': questions[i].wave,
                    'change_type': 'wording_modification',
                    'details': self._identify_text_differences(reference_text, text)
                })
        
        return {
            'question_texts': question_texts,
            'similarities': similarities,
            'average_similarity': np.mean(similarities) if similarities else 1.0,
            'changes_detected': changes_detected,
            'wording_stable': len(set(question_texts)) == 1
        }
    
    def _analyze_response_option_changes(self, questions: List[SurveyQuestion]) -> Dict:
        """Analyze changes in response options."""
        response_analyses = []
        
        # Compare response structures
        for i, question in enumerate(questions):
            options = [resp.option for resp in question.response_items if resp.count is not None]
            
            response_analyses.append({
                'wave': question.wave,
                'variable': question.variable_name,
                'num_options': len(options),
                'options': options,
                'response_counts': {resp.option: resp.count for resp in question.response_items if resp.count is not None}
            })
        
        # Detect changes
        changes = []
        if len(response_analyses) > 1:
            reference = response_analyses[0]
            
            for analysis in response_analyses[1:]:
                if analysis['options'] != reference['options']:
                    changes.append({
                        'wave': analysis['wave'],
                        'change_type': 'response_options_modified',
                        'added_options': set(analysis['options']) - set(reference['options']),
                        'removed_options': set(reference['options']) - set(analysis['options'])
                    })
        
        return {
            'response_analyses': response_analyses,
            'changes_detected': changes,
            'options_stable': len(changes) == 0
        }
    
    def _analyze_metadata_changes(self, questions: List[SurveyQuestion]) -> Dict:
        """Analyze changes in metadata (section, level, type, etc.)."""
        metadata_fields = ['section', 'level', 'type', 'width', 'decimals']
        metadata_analysis = {}
        
        for field in metadata_fields:
            values = [getattr(q, field, '') for q in questions]
            metadata_analysis[field] = {
                'values': values,
                'stable': len(set(values)) == 1,
                'changes': [(i, val) for i, val in enumerate(values) if i > 0 and val != values[0]]
            }
        
        return metadata_analysis
    
    def _calculate_consistency_score(self, questions: List[SurveyQuestion]) -> float:
        """Calculate overall consistency score across waves."""
        scores = []
        
        # Question wording consistency (40% weight)
        question_texts = [q.question for q in questions]
        if len(set(question_texts)) == 1:
            scores.append(1.0 * 0.4)
        else:
            avg_similarity = np.mean([
                self._calculate_text_similarity(question_texts[i], question_texts[i+1])
                for i in range(len(question_texts)-1)
            ])
            scores.append(avg_similarity * 0.4)
        
        # Response options consistency (30% weight)
        response_options = [[resp.option for resp in q.response_items] for q in questions]
        if all(opts == response_options[0] for opts in response_options):
            scores.append(1.0 * 0.3)
        else:
            scores.append(0.5 * 0.3)  # Partial credit for some consistency
        
        # Metadata consistency (30% weight)
        metadata_scores = []
        for field in ['section', 'level', 'type']:
            values = [getattr(q, field, '') for q in questions]
            if len(set(values)) == 1:
                metadata_scores.append(1.0)
            else:
                metadata_scores.append(0.0)
        
        scores.append(np.mean(metadata_scores) * 0.3)
        
        return sum(scores)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if text1 == text2:
            return 1.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _identify_text_differences(self, text1: str, text2: str) -> List[str]:
        """Identify specific differences between two texts."""
        differences = []
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        added_words = words2 - words1
        removed_words = words1 - words2
        
        if added_words:
            differences.append(f"Added words: {', '.join(added_words)}")
        if removed_words:
            differences.append(f"Removed words: {', '.join(removed_words)}")
        
        return differences
    
    def _describe_naming_pattern(self, prefixes: List[str], numbers: List[str], suffixes: List[str]) -> str:
        """Describe the variable naming pattern."""
        if len(set(prefixes)) == 1 and len(set(numbers)) == 1:
            return f"Consistent pattern: {prefixes[0]}{numbers[0]}X where X changes by wave"
        elif len(set(numbers)) == 1:
            return f"Number consistent ({numbers[0]}), prefix varies by wave"
        else:
            return "Variable naming pattern changes across waves"
    
    def _generate_recommendations(self, questions: List[SurveyQuestion]) -> List[str]:
        """Generate recommendations for longitudinal analysis."""
        recommendations = []
        
        # Check for potential issues
        var_names = [q.variable_name for q in questions]
        question_texts = [q.question for q in questions]
        
        if len(set(question_texts)) > 1:
            recommendations.append("⚠️ Question wording changed across waves - consider harmonization when analyzing trends")
        
        if len(set([len(q.response_items) for q in questions])) > 1:
            recommendations.append("⚠️ Number of response options changed - verify comparability")
        
        # Positive recommendations
        if len(set(question_texts)) == 1:
            recommendations.append("✅ Question wording is consistent - good for trend analysis")
        
        recommendations.append(f"📊 {len(questions)} waves available for longitudinal analysis")
        recommendations.append("🔍 Consider examining response distributions for each wave")
        
        return recommendations


class QueryProcessor:
    """
    Processes user queries using a combination of vector search,
    intent analysis, and LLM-based response generation.
    """

    def __init__(self, data_manager: DataManager, model_name: str = "llama3.2:latest"):
        """
        Initialize the query processor.

        Args:
            data_manager: DataManager instance with loaded data
            model_name: Name of the Ollama model to use
        """
        self.data_manager = data_manager
        self.model_name = model_name

        # Initialize Ollama LLM
        self.llm = OllamaLLM(model=model_name)

        # Initialize cache manager
        self.cache_manager = CacheManager()

        # Initialize memory caches
        self._memory_cache = {
            'intent': {},
            'query': {},
            'answer': {}
        }

        # Initialize intent analysis chain
        self.intent_chain = self._create_intent_chain()

        # Initialize answer generation chain
        self.answer_chain = self._create_answer_chain()

    def _create_intent_chain(self):
        """Create the chain for query intent analysis."""
        # Create the parser
        parser = PydanticOutputParser(pydantic_object=QueryIntent)

        # Intent analysis prompt with improved variable detection
        intent_template = """
        You are an AI assistant analyzing user queries about longitudinal survey data.

        Given a user query, determine the primary intent, extract relevant filters,
        and identify any specific variables or time periods mentioned.

        IMPORTANT RULES FOR VARIABLE EXTRACTION:
        - Only extract variable names if they are EXPLICITLY mentioned in formats like "KLB023D", "JLB001", or "variable ABC123"
        - Do NOT extract variable names from natural language descriptions
        - For queries like "satisfied with life" - this is asking about CONTENT, not a specific variable
        - For queries like "show me JLB001" or "explain variable KLB023D" - these ARE asking for specific variables

        Available waves: {available_waves}
        Available sections: {available_sections}
        Sample variables: {sample_variables}

        USER QUERY: {query}

        Analyze the above query and ONLY output a valid JSON object.
        The JSON must be in the following format:

        {format_instructions}

        """

        # Create the prompt template
        prompt = PromptTemplate(
            template=intent_template,
            input_variables=["query", "available_waves", "available_sections", "sample_variables"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # Create and return the chain (using LCEL pattern)
        return prompt | self.llm | parser

    def _create_answer_chain(self):
        """Create the chain for generating answers to queries."""
        answer_template = """
        You are an AI assistant helping users explore and understand longitudinal survey data.

        USER QUERY: {query}

        QUERY INTENT: {intent}

        RELEVANT SURVEY QUESTIONS:
        {context}

        Based on the query intent and the provided survey questions, provide a comprehensive and accurate answer.
        Focus on directly addressing the user's intent and providing relevant insights from the data.

        When referring to survey questions, mention their variable names (e.g., KLB023D) and waves.
        If the user is asking about trends over time, make sure to highlight differences across waves.
        If multiple questions are relevant, synthesize information across them.

        If statistical information is requested, provide it clearly with proper context.

        YOUR ANSWER:
        """

        # Create the prompt template
        prompt = PromptTemplate(
            template=answer_template,
            input_variables=["query", "intent", "context"]
        )

        # Create and return the chain (using LCEL pattern)
        return prompt | self.llm

    def _get_cache_key(self, prefix: str, query: str, extra_data: Any = None) -> str:
        """
        Generate a consistent cache key using the query and optional extra data.

        Args:
            prefix: Cache key prefix
            query: The user query
            extra_data: Optional additional data to include in the key

        Returns:
            A unique cache key as string
        """
        # Create a normalized string from the query and extra data
        if extra_data:
            key_data = f"{normalize_query(query)}_{str(extra_data)}"
        else:
            key_data = normalize_query(query)

        # Create MD5 hash
        return f"{prefix}_{hashlib.md5(key_data.encode()).hexdigest()}"

    def analyze_intent(self, query: str) -> QueryIntent:
        """
        Analyze the intent of a user query with improved error handling and caching.

        Args:
            query: The user query string

        Returns:
            QueryIntent object with detected intent information
        """
        # Normalize the query for consistent caching
        normalized_query = normalize_query(query)

        # Generate cache key
        cache_key = self._get_cache_key('intent', normalized_query)

        # Check memory cache first (fastest)
        if cache_key in self._memory_cache['intent']:
            print(f"Using memory cache for intent: {query}")
            return self._memory_cache['intent'][cache_key]

        # Then check disk cache
        cached_intent = self.cache_manager.get(cache_key)
        if cached_intent:
            print(f"Using disk cache for intent: {query}")
            # Also store in memory for faster access next time
            self._memory_cache['intent'][cache_key] = cached_intent
            return cached_intent

        try:
            # Get available metadata for context
            available_waves = self.data_manager.get_unique_values("wave")
            available_sections = self.data_manager.get_unique_values("section")

            # Use deterministic sampling instead of random sampling
            all_variables = self.data_manager.df["variable_name"].tolist()
            # Sort to ensure consistent ordering
            all_variables.sort()
            # Take the first 20 (or fewer) elements predictably
            sample_variables = all_variables[:min(20, len(all_variables))]

            # Create a more direct prompt to get just JSON (bypassing the chain)
            direct_template = """
            Analyze this query about survey data: "{query}"

            Available waves: {waves}
            Available sections: {sections}
            Available variables: {variables}

            CRITICAL RULES:
            - Only extract variable names if EXPLICITLY mentioned (e.g., "KLB023D", "show me JLB001")
            - Do NOT extract variables from natural language (e.g., "satisfied with life" should NOT extract any variables)
            - Natural language queries are about CONTENT, not specific variable codes

            ONLY return a valid JSON object in this exact format with no additional wrapper or nested properties.
            DO NOT use comments in the JSON - they will break parsing:
            {{
                "primary_intent": "search",
                "secondary_intent": null,
                "filter_criteria": {{}},
                "target_variables": [],
                "time_periods": [],
                "analysis_type": null,
                "confidence": 0.7
            }}
            """

            prompt = direct_template.format(
                query=query,
                waves=", ".join(available_waves),
                sections=", ".join(available_sections),
                variables=", ".join(sample_variables)
            )

            # Get the raw response from LLM
            raw_result = self.llm.invoke(prompt)

            # Clean up any comments in the JSON
            raw_result = re.sub(r'//.*', '', raw_result)

            # Extract JSON using regex
            json_match = re.search(r'\{[\s\S]*\}', raw_result)
            if json_match:
                json_str = json_match.group(0)
                try:
                    # Parse the JSON into a dict
                    data = json.loads(json_str)

                    # Check for nested "properties" structure and extract if needed
                    if "properties" in data:
                        print("Found nested properties structure, extracting...")
                        data = data["properties"]

                    # Ensure required fields are present
                    if "primary_intent" not in data:
                        data["primary_intent"] = "search"
                    if "confidence" not in data:
                        data["confidence"] = 0.7

                    # Additional validation: only keep variables that actually exist
                    if "target_variables" in data and isinstance(data["target_variables"], list):
                        validated_variables = []
                        for var in data["target_variables"]:
                            if isinstance(var, dict) and "name" in var:
                                var_name = var["name"]
                            else:
                                var_name = str(var)
                            
                            # Check if variable actually exists in dataset
                            if self.data_manager.get_question_by_variable(var_name):
                                validated_variables.append(var_name)
                            else:
                                print(f"Removed non-existent variable: {var_name}")
                        
                        data["target_variables"] = validated_variables
                    
                    # IMPORTANT FIX: Handle time_periods in dictionary format
                    if "time_periods" in data and isinstance(data["time_periods"], list):
                        processed_time_periods = []
                        for period in data["time_periods"]:
                            if isinstance(period, dict):
                                # Handle dictionary format (e.g., {"start_date": 2004, "end_date": 2008})
                                if "start_date" in period and "end_date" in period:
                                    start_year = period["start_date"]
                                    end_year = period["end_date"]
                                    
                                    # Generate individual year strings for each year in the range
                                    for year in range(start_year, end_year + 1, 2):  # Assuming years increment by 2
                                        if year <= 2018:  # Ensure we don't go beyond available data
                                            processed_time_periods.append(f"{year} Core")
                                elif "year" in period:
                                    # Handle single year dictionaries
                                    processed_time_periods.append(f"{period['year']} Core")
                            elif isinstance(period, str):
                                # Handle already correctly formatted strings
                                if " Core" not in period and period.isdigit():
                                    processed_time_periods.append(f"{period} Core")
                                else:
                                    processed_time_periods.append(period)
                            elif isinstance(period, int):
                                # Handle plain year numbers
                                processed_time_periods.append(f"{period} Core")
                        
                        # Replace the time_periods with our processed list
                        data["time_periods"] = processed_time_periods
                        print(f"Processed time periods: {processed_time_periods}")

                    # Create the QueryIntent object
                    intent = QueryIntent(**data)

                    # Cache the intent in memory
                    self._memory_cache['intent'][cache_key] = intent

                    # Cache to disk as well
                    self.cache_manager.set(cache_key, intent)

                    return intent

                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}, json_str: {json_str}")
                    # Continue to fallback

        except Exception as e:
            print(f"Error in intent analysis: {e}")

        # Fallback - extract key information directly from the query text
        primary_intent = "search"  # Default intent

        # Check for specific words that might indicate intent
        if re.search(r'\b(compare|comparison|differences?|changes?)\b', query.lower()):
            primary_intent = "compare"
        elif re.search(r'\b(explain|meaning|definition|what\s+is|tell\s+me\s+about)\b', query.lower()):
            primary_intent = "explain"
        elif re.search(r'\b(analysis|analyze|trend|correlation|relationship)\b', query.lower()):
            primary_intent = "analyze"
        elif re.search(r'\b(summarize|summary|overview)\b', query.lower()):
            primary_intent = "summarize"

        # Try to extract any variable names ONLY if explicitly mentioned
        variable_pattern = r'\b([A-Z]{2,3}\d{3}[A-Z]?)\b'
        potential_variables = re.findall(variable_pattern, query)
        
        # Validate that these variables exist
        target_variables = []
        for var in potential_variables:
            if self.data_manager.get_question_by_variable(var):
                target_variables.append(var)

        # Extract any mentioned time periods (if waves are numeric)
        time_periods = []
        wave_pattern = r'\b(\d{4})\b'  # Look for 4-digit years
        wave_matches = re.findall(wave_pattern, query)
        
        # Format wave matches with " Core" suffix
        for year in wave_matches:
            time_periods.append(f"{year} Core")
        
        # Also check for "wave X" mentions
        wave_mention_pattern = r'\bwave\s+(\d+|[ivxlcdm]+)\b'
        wave_mention_matches = re.findall(wave_mention_pattern, query.lower())
        
        # Try to map these to years if possible
        for wave_num in wave_mention_matches:
            # For now, just add as is (better handling could be added)
            if wave_num not in time_periods:
                time_periods.append(wave_num)

        # Create fallback intent
        intent = QueryIntent(
            primary_intent=primary_intent,
            secondary_intent=None,
            filter_criteria={},
            target_variables=target_variables,
            time_periods=time_periods,
            analysis_type=None,
            confidence=0.6
        )

        # Cache the fallback intent
        self._memory_cache['intent'][cache_key] = intent
        self.cache_manager.set(cache_key, intent)

        return intent


    def find_relevant_questions(self, query: str, intent: QueryIntent, limit: int = 20) -> List[QueryResult]:
        """
        Find relevant survey questions based on query and intent using enhanced relevance scoring.
        """
        # Create a cache key
        cache_key = self._get_cache_key(
            'query',
            query,
            (intent.primary_intent, tuple(intent.target_variables), tuple(intent.time_periods), limit)
        )

        # Check memory cache
        if cache_key in self._memory_cache['query']:
            print(f"Using memory cache for query results: {query}")
            return self._memory_cache['query'][cache_key]

        # Check disk cache
        cached_results = self.cache_manager.get(cache_key)
        if cached_results:
            print(f"Using disk cache for query results: {query}")
            self._memory_cache['query'][cache_key] = cached_results
            return cached_results

        # Start with an empty list
        results = []

        # Handle exact variable matches with improved relevance checking
        if intent.target_variables:
            for var_name in intent.target_variables:
                question = self.data_manager.get_question_by_variable(var_name)
                if question:
                    # Check content relevance for exact variable matches
                    combined_text = f"{question.description} {question.question}".lower()
                    query_lower = query.lower()
                    
                    # Simple relevance check
                    query_terms = query_lower.split()
                    matching_terms = sum(1 for term in query_terms if term in combined_text)
                    relevance_ratio = matching_terms / len(query_terms) if query_terms else 0
                    
                    # Only give high scores if content is actually relevant
                    if relevance_ratio > 0.3:  # At least 30% of query terms match
                        similarity_score = 0.95 + (relevance_ratio * 0.05)  # 0.95-1.0 range
                        explanation = f"Exact variable match with high content relevance"
                    else:
                        similarity_score = 0.6  # Lower score for irrelevant exact matches
                        explanation = f"Exact variable match but low content relevance"
                    
                    results.append(
                        QueryResult(
                            question=question,
                            similarity_score=similarity_score,
                            relevance_explanation=explanation
                        )
                    )

        # Apply filters from intent
        filters = {}

        # Get a larger set of questions to search through
        try:
            all_questions = self.data_manager.filter_questions(limit=500)

            query_lower = query.lower()
            query_terms = query_lower.split()
            cleaned_query = query_lower.strip()

            # Enhanced categorization with more granular scoring
            exact_matches = []
            high_relevance = []    # High relevance matches
            good_matches = []
            medium_matches = []    # Medium relevance matches  
            partial_matches = []

            for question in all_questions:
                description_lower = question.description.lower() if question.description else ""
                question_text_lower = question.question.lower() if question.question else ""
                combined_text = f"{description_lower} {question_text_lower}"

                # Check for PERFECT matches first
                description_clean = description_lower.strip()
                question_clean = question_text_lower.strip()
                
                perfect_match = False
                
                # Perfect match conditions for exact conceptual matches
                if cleaned_query in description_clean:
                    if (cleaned_query == description_clean or 
                        f" {cleaned_query} " in f" {description_clean} " or
                        description_clean.startswith(f"{cleaned_query} ") or
                        description_clean.endswith(f" {cleaned_query}")):
                        perfect_match = True
                
                if cleaned_query in question_clean:
                    if (cleaned_query == question_clean or 
                        f" {cleaned_query} " in f" {question_clean} " or
                        question_clean.startswith(f"{cleaned_query} ") or
                        question_clean.endswith(f" {cleaned_query}")):
                        perfect_match = True
                
                # Special handling for common research concepts
                if "satisfied with life" in cleaned_query and "satisfied with life" in description_clean:
                    perfect_match = True
                
                if perfect_match:
                    exact_matches.append(
                        QueryResult(
                            question=question,
                            similarity_score=1.00,
                            relevance_explanation="Perfect match - query exactly matches variable concept"
                        )
                    )
                    continue

                # ENHANCED RELEVANCE SCORING for non-perfect matches
                score, explanation = self._calculate_enhanced_relevance_score(
                    query_lower, query_terms, cleaned_query, 
                    description_lower, question_text_lower, combined_text
                )
                
                # Only include results above minimum threshold
                if score >= 0.60:
                    # Categorize based on enhanced scores
                    if score >= 0.95:
                        exact_matches.append(QueryResult(question=question, similarity_score=score, relevance_explanation=explanation))
                    elif score >= 0.90:
                        high_relevance.append(QueryResult(question=question, similarity_score=score, relevance_explanation=explanation))
                    elif score >= 0.80:
                        good_matches.append(QueryResult(question=question, similarity_score=score, relevance_explanation=explanation))
                    elif score >= 0.70:
                        medium_matches.append(QueryResult(question=question, similarity_score=score, relevance_explanation=explanation))
                    else:  # 0.60-0.69
                        partial_matches.append(QueryResult(question=question, similarity_score=score, relevance_explanation=explanation))

            # Add matches in priority order, avoiding duplicates
            existing_vars = {result.question.variable_name for result in results}

            def add_matches(matches, existing_vars, results):
                for match in matches:
                    if match.question.variable_name not in existing_vars:
                        results.append(match)
                        existing_vars.add(match.question.variable_name)

            # Add matches in enhanced priority order
            add_matches(exact_matches, existing_vars, results)
            add_matches(high_relevance, existing_vars, results)
            add_matches(good_matches, existing_vars, results)
            add_matches(medium_matches, existing_vars, results)
            add_matches(partial_matches, existing_vars, results)

        except Exception as e:
            print(f"Error in direct text matching: {e}")

        # If we haven't reached the limit yet, perform semantic search
        if len(results) < limit:
            remaining = limit - len(results)
            existing_vars = {result.question.variable_name for result in results}

            try:
                # Use semantic search for remaining slots
                semantic_results = self.data_manager.query_similar(
                    query_text=query,
                    filters=filters,
                    limit=remaining * 2  # Get more to account for duplicates
                )

                # Add non-duplicate semantic results
                for result in semantic_results:
                    if result.question.variable_name not in existing_vars and len(results) < limit:
                        results.append(result)
                        existing_vars.add(result.question.variable_name)

            except Exception as e:
                print(f"Error in semantic search: {e}")

        # Sort results by similarity score to ensure most relevant are first
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        # Limit the results
        final_results = results[:limit]

        # Cache the results
        self._memory_cache['query'][cache_key] = final_results
        self.cache_manager.set(cache_key, final_results)

        return final_results


    def _calculate_enhanced_relevance_score(self, query_lower: str, query_terms: list, cleaned_query: str, 
                                        description_lower: str, question_text_lower: str, combined_text: str) -> tuple:
        """
        Calculate enhanced relevance score based on multiple factors.
        Returns (score, explanation) tuple.
        """
        
        # Initialize base score
        base_score = 0.0
        score_factors = []
        
        # Factor 1: Exact phrase matching (high weight)
        exact_phrase_match = cleaned_query in combined_text
        if exact_phrase_match:
            base_score += 0.40
            score_factors.append("exact phrase match")
        
        # Factor 2: Term matching ratio - This is the foundation score
        matching_terms = sum(1 for term in query_terms if term in combined_text)
        match_ratio = matching_terms / len(query_terms) if query_terms else 0
        
        # Give base score based on match ratio
        if match_ratio == 1.0:
            base_score += 0.35  # All terms match
            score_factors.append("all terms match")
        elif match_ratio >= 0.8:
            base_score += 0.30  # Most terms match
            score_factors.append("most terms match")
        elif match_ratio >= 0.6:
            base_score += 0.25  # Many terms match
            score_factors.append("many terms match")
        elif match_ratio >= 0.4:
            base_score += 0.20  # Some terms match
            score_factors.append("some terms match")
        elif match_ratio > 0:
            base_score += 0.15  # Few terms match
            score_factors.append("few terms match")
        
        # Factor 3: High-value term bonuses (more sophisticated)
        high_value_bonus = 0
        
        # For "satisfied with life" queries, prioritize these concepts
        if "satisfied" in cleaned_query:
            if "satisfied" in combined_text:
                high_value_bonus += 0.15
                score_factors.append("key term: satisfied")
            if "satisfaction" in combined_text:
                high_value_bonus += 0.10
                score_factors.append("key term: satisfaction")
        
        if "life" in cleaned_query:
            if "life" in combined_text:
                high_value_bonus += 0.15
                score_factors.append("key term: life")
            # Bonus for life-quality concepts
            life_quality_terms = ["ideal", "close to", "whole", "personal", "family", "important"]
            life_quality_matches = sum(1 for term in life_quality_terms if term in combined_text)
            if life_quality_matches > 0:
                high_value_bonus += min(0.10, life_quality_matches * 0.03)
                score_factors.append(f"life-quality concepts ({life_quality_matches})")
        
        base_score += high_value_bonus
        
        # Factor 4: Question vs Description placement bonuses
        if any(term in description_lower for term in query_terms):
            base_score += 0.08
            score_factors.append("matches description")
        elif any(term in question_text_lower for term in query_terms):
            base_score += 0.04
            score_factors.append("matches question text")
        
        # Factor 5: Psychological/subjective measure indicators
        if "satisfied" in cleaned_query or "life" in cleaned_query:
            psych_indicators = ["agree", "disagree", "statements", "much you", "think about", "feel", "close to"]
            psych_matches = sum(1 for indicator in psych_indicators if indicator in combined_text)
            if psych_matches > 0:
                psych_bonus = min(0.08, psych_matches * 0.02)
                base_score += psych_bonus
                score_factors.append("psychological measure")
        
        # Factor 6: Domain relevance penalties/bonuses
        domain_adjustment = 0
        
        # Bonus for well-being/life satisfaction domain
        wellbeing_terms = ["happiness", "well-being", "quality", "ideal", "control", "important"]
        wellbeing_matches = sum(1 for term in wellbeing_terms if term in combined_text)
        if wellbeing_matches > 0:
            domain_adjustment += min(0.06, wellbeing_matches * 0.02)
            score_factors.append("well-being domain")
        
        # Small penalty for work/job context when asking about life satisfaction
        if "satisfied" in cleaned_query and not "job" in cleaned_query and not "work" in cleaned_query:
            work_terms = ["job", "work", "employment", "career"]
            work_matches = sum(1 for term in work_terms if term in combined_text)
            if work_matches > 0:
                domain_adjustment -= 0.03
                score_factors.append("work context penalty")
        
        base_score += domain_adjustment
        
        # Factor 7: Contextual relevance boost
        # If description contains the query concept, boost significantly
        core_concepts = ["satisfied with life", "life satisfaction", "happiness", "well-being"]
        for concept in core_concepts:
            if concept in cleaned_query and concept in combined_text:
                base_score += 0.12
                score_factors.append(f"core concept: {concept}")
                break
        
        # Ensure score stays within reasonable bounds
        final_score = max(0.0, min(0.98, base_score))
        
        # Create explanation
        if score_factors:
            explanation = "Relevance: " + ", ".join(score_factors[:4])  # Limit to first 4 factors for readability
        else:
            explanation = "Basic term matching"
        
        return final_score, explanation



    def generate_answer(self, query: str, intent: QueryIntent, results: List[QueryResult]) -> str:
        """
        Generate an answer based on query intent and relevant results with improved structure
        for handling larger result sets.

        Args:
            query: The original user query
            intent: The analyzed query intent
            results: List of relevant survey questions

        Returns:
            Generated answer text
        """
        # Create a unique key for this specific combination of query, intent, and results
        answer_key = self._get_cache_key(
            'answer',
            query,
            (
                intent.primary_intent,
                tuple(intent.target_variables),
                tuple(r.question.variable_name for r in results)
            )
        )

        # Check memory cache first
        if answer_key in self._memory_cache['answer']:
            print(f"Using memory cache for answer: {query}")
            return self._memory_cache['answer'][answer_key]

        # Then check disk cache
        cached_answer = self.cache_manager.get(answer_key)
        if cached_answer:
            print(f"Using disk cache for answer: {query}")
            # Store in memory cache too
            self._memory_cache['answer'][answer_key] = cached_answer
            return cached_answer

        try:
            # Check if we have results
            if not results:
                answer = "I couldn't find any survey questions related to your query. Please try rephrasing or being more specific about what you're looking for."

                # Cache the answer
                self._memory_cache['answer'][answer_key] = answer
                self.cache_manager.set(answer_key, answer)

                return answer

            # Organize results into multiple tiers by relevance - ADJUSTED THRESHOLDS
            exact_matches = []  # 0.95-1.0: Perfect or near-perfect matches (increased from 0.9)
            high_relevance = []  # 0.85-0.94: Very relevant (adjusted range)
            medium_relevance = []  # 0.75-0.84: Somewhat relevant (adjusted range)
            low_relevance = []  # Below 0.75: Limited relevance (adjusted threshold)

            for result in results:
                if result.similarity_score >= 0.95:  # Increased from 0.9
                    exact_matches.append(result)
                elif result.similarity_score >= 0.85:  # Increased from 0.8
                    high_relevance.append(result)
                elif result.similarity_score >= 0.75:  # Increased from 0.7
                    medium_relevance.append(result)
                else:
                    low_relevance.append(result)

            # Create a more structured response for many results
            response_parts = []

            # Add an introduction
            total_count = len(results)
            search_term = query.replace("give me all", "").replace("variables related to", "").strip()

            if total_count == 1:
                intro = f"I found 1 survey question related to '{search_term}'."
            else:
                intro = f"I found {total_count} survey questions related to '{search_term}'."

            response_parts.append(intro)

            # Add main variables section for exact matches
            if exact_matches:
                response_parts.append("\n## Exact Matches\n")
                for result in exact_matches:
                    q = result.question
                    var_info = f"**{q.variable_name}**: {q.description} (Wave: {q.wave})\n"
                    var_info += f"*Question:* {q.question}\n"

                    # Add response summary if available
                    if q.response_items and len(q.response_items) > 0:
                        var_info += "*Response options include:* "
                        options = [r.option for r in q.response_items[:3]]
                        if len(q.response_items) > 3:
                            options.append("...")
                        var_info += ", ".join(options) + "\n"

                    response_parts.append(var_info)

            # Add highly relevant variables
            if high_relevance:
                if exact_matches:  # If we already have exact matches, label these differently
                    response_parts.append("\n## Highly Relevant Variables\n")
                else:  # Otherwise these are our main variables
                    response_parts.append("\n## Main Variables\n")

                for result in high_relevance:
                    q = result.question
                    var_info = f"**{q.variable_name}**: {q.description} (Wave: {q.wave})\n"
                    var_info += f"*Question:* {q.question}\n"

                    # Add response summary if available (more concise if we have many variables)
                    if q.response_items and len(q.response_items) > 0:
                        var_info += "*Response options include:* "
                        options = [r.option for r in q.response_items[:3]]
                        if len(q.response_items) > 3:
                            options.append("...")
                        var_info += ", ".join(options) + "\n"

                    response_parts.append(var_info)

            # Add medium relevance section
            if medium_relevance:
                response_parts.append("\n## Related Variables\n")

                # If we have many variables, use more compact format
                if len(medium_relevance) > 5:
                    # More compact format for many variables
                    for i, result in enumerate(medium_relevance):
                        q = result.question
                        var_info = f"**{q.variable_name}**: {q.description} (Wave: {q.wave})\n"
                        response_parts.append(var_info)
                else:
                    # Regular format for fewer variables
                    for i, result in enumerate(medium_relevance):
                        q = result.question
                        var_info = f"**{q.variable_name}**: {q.description} (Wave: {q.wave})\n"
                        var_info += f"*Question:* {q.question}\n"
                        response_parts.append(var_info)

            # Add low relevance section if needed
            if low_relevance and (exact_matches or high_relevance or medium_relevance):
                response_parts.append("\n## Other Potentially Related Variables\n")
                # Very compact format for low relevance matches
                var_list = [f"**{r.question.variable_name}** ({r.question.description})" for r in low_relevance]
                response_parts.append(", ".join(var_list))

            # Add usage suggestions
            if total_count > 0:
                response_parts.append("\n## Research Suggestions\n")
                response_parts.append("Based on these variables, you could explore:")

                # Generate research suggestions based on the found variables
                if exact_matches:
                    main_var = exact_matches[0].question.variable_name
                elif high_relevance:
                    main_var = high_relevance[0].question.variable_name
                elif medium_relevance:
                    main_var = medium_relevance[0].question.variable_name
                else:
                    main_var = low_relevance[0].question.variable_name

                suggestions = [
                    f"- How responses to {main_var} vary across different demographic groups",
                    f"- Relationships between {main_var} and other related variables",
                    "- Differences in responses across different waves of the survey"
                ]

                response_parts.append("\n".join(suggestions))

            # Join all parts
            answer = "\n".join(response_parts)

            # Cache the answer
            self._memory_cache['answer'][answer_key] = answer
            self.cache_manager.set(answer_key, answer)

            return answer

        except Exception as e:
            print(f"Error in structured answer generation: {e}")
            fallback_answer = f"I found {len(results)} variables related to your query about '{query}'. The most relevant ones include: " + \
                              ", ".join([f"{r.question.variable_name} ({r.question.description})" for r in results[:3]])

            # Cache the fallback answer
            self._memory_cache['answer'][answer_key] = fallback_answer
            self.cache_manager.set(answer_key, fallback_answer)

            return fallback_answer

    def process_query(self, user_query: UserQuery) -> ProcessedQuery:
        """
        Process a user query end-to-end with better error handling.

        Args:
            user_query: UserQuery object with query text and parameters

        Returns:
            ProcessedQuery object with intent, results, and answer
        """
        try:
            # Analyze intent
            intent = self.analyze_intent(user_query.query_text)

            print(f"Intent analyzed: {intent}")

            # Find relevant questions with balanced sampling
            results = self.find_relevant_questions(
                query=user_query.query_text,
                intent=intent,
                limit=user_query.limit
            )

            # If no results found through intent analysis, try direct semantic search
            if not results:
                try:
                    # Get all available waves for complete search
                    available_waves = self.data_manager.get_unique_values("wave")
                    wave_filter = {"wave": available_waves}
                    
                    # Create ChromaDB compatible filter
                    chroma_filter = {"$or": []}
                    for wave in available_waves:
                        chroma_filter["$or"].append({"wave": {"$eq": wave}})
                    
                    # Try semantic search with wave filter
                    results = self.data_manager.query_similar(
                        query_text=user_query.query_text,
                        filters=chroma_filter,  # Use ChromaDB compatible filter
                        limit=user_query.limit
                    )
                    
                    print(f"Fallback semantic search found {len(results)} results")
                except Exception as e:
                    print(f"Error in fallback semantic search: {e}")

            # Generate answer
            if results:
                answer = self.generate_answer(
                    query=user_query.query_text,
                    intent=intent,
                    results=results
                )
            else:
                answer = "I couldn't find any relevant survey questions about that topic in our dataset."

            # Collect context used
            context_used = [result.question.variable_name for result in results]

            return ProcessedQuery(
                original_query=user_query.query_text,
                intent=intent,
                results=results,
                answer=answer,
                context_used=context_used
            )

        except Exception as e:
            # Fallback response for any errors
            print(f"Error processing query: {e}")
            return ProcessedQuery(
                original_query=user_query.query_text,
                intent=QueryIntent(
                    primary_intent="search",
                    secondary_intent=None,
                    filter_criteria={},
                    target_variables=[],
                    time_periods=[],
                    analysis_type=None,
                    confidence=0.5
                ),
                results=[],
                answer=f"I encountered an issue while processing your query. Please try rephrasing or asking a more specific question about the survey data.",
                context_used=[]
            )

    def analyze_longitudinal_patterns(self, query_results: List[QueryResult]) -> Dict[str, Any]:
        """
        Analyze longitudinal patterns in query results.
        
        Args:
            query_results: Results from a search query
            
        Returns:
            Comprehensive longitudinal analysis
        """
        analyzer = LongitudinalAnalyzer(self.data_manager)
        
        # Group variables by concept
        concept_groups = analyzer.find_longitudinal_variables(query_results)
        
        # Analyze changes for each concept
        analysis_results = analyzer.analyze_longitudinal_changes(concept_groups)
        
        # Generate summary
        summary = {
            'total_concepts': len(concept_groups),
            'concepts_analyzed': len(analysis_results),
            'concept_analyses': analysis_results,
            'overall_insights': self._generate_overall_insights(analysis_results)
        }
        
        return summary

    def _generate_overall_insights(self, analysis_results: Dict[str, Dict]) -> List[str]:
        """Generate overall insights across all concepts."""
        insights = []
        
        if not analysis_results:
            return ["No longitudinal patterns detected in the results."]
        
        # Calculate average consistency
        consistency_scores = [analysis['consistency_score'] for analysis in analysis_results.values()]
        avg_consistency = np.mean(consistency_scores)
        
        if avg_consistency > 0.9:
            insights.append("🎯 High consistency across waves - excellent for trend analysis")
        elif avg_consistency > 0.7:
            insights.append("📈 Moderate consistency - some harmonization may be needed")
        else:
            insights.append("⚠️ Low consistency - significant changes detected across waves")
        
        # Variable naming patterns
        naming_patterns = []
        for analysis in analysis_results.values():
            if analysis['changes']['variable_naming']['naming_consistent']:
                naming_patterns.append("consistent")
            else:
                naming_patterns.append("varied")
        
        if naming_patterns.count("consistent") > len(naming_patterns) / 2:
            insights.append("🏷️ Variable naming follows consistent patterns")
        else:
            insights.append("🔄 Variable naming patterns vary across concepts")
        
        return insights

    def explain_variable(self, variable_name: str) -> str:
        """
        Generate an explanation for a specific variable with error handling.

        Args:
            variable_name: The variable to explain

        Returns:
            Explanation text
        """
        # Create cache key for variable explanations
        cache_key = self._get_cache_key('explain', variable_name)

        # Check caches
        if 'explain' in self._memory_cache and cache_key in self._memory_cache['explain']:
            print(f"Using memory cache for explanation of {variable_name}")
            return self._memory_cache['explain'][cache_key]

        # Check disk cache
        cached_explanation = self.cache_manager.get(cache_key)
        if cached_explanation:
            print(f"Using disk cache for explanation of {variable_name}")
            # Ensure explain cache exists in memory
            if 'explain' not in self._memory_cache:
                self._memory_cache['explain'] = {}
            self._memory_cache['explain'][cache_key] = cached_explanation
            return cached_explanation

        try:
            # Get the question
            question = self.data_manager.get_question_by_variable(variable_name)
            
            if not question:
                explanation = f"Variable {variable_name} not found in the dataset."
                
                # Cache the negative result
                if 'explain' not in self._memory_cache:
                    self._memory_cache['explain'] = {}
                self._memory_cache['explain'][cache_key] = explanation
                self.cache_manager.set(cache_key, explanation)
                
                return explanation

            # Create context for the question
            context = create_structured_context(question)

            # Prompt template
            template = """
            You are an AI assistant explaining longitudinal survey data.

            TASK: Explain the variable {variable_name} in detail.

            VARIABLE DATA:
            {context}

            Provide a comprehensive explanation that includes:
            1. What this variable measures
            2. How the question is asked to respondents
            3. What the response options mean
            4. Any important context about this measure
            5. How this variable might be used in research

            EXPLANATION:
            """

            prompt = PromptTemplate(
                template=template,
                input_variables=["variable_name", "context"]
            )

            # Create a one-off chain using LCEL pattern
            chain = prompt | self.llm

            # Generate explanation
            explanation = chain.invoke({
                "variable_name": variable_name,
                "context": context
            })

            # Cache the result
            if 'explain' not in self._memory_cache:
                self._memory_cache['explain'] = {}
            self._memory_cache['explain'][cache_key] = explanation
            self.cache_manager.set(cache_key, explanation)

            return explanation

        except Exception as e:
            print(f"Error in explain_variable: {e}")
            
            # Fallback response
            if question:
                fallback = f"Variable {variable_name}: {question.description}\n\n"
                fallback += f"Question: {question.question}\n\n"
                fallback += f"Wave: {question.wave}, Section: {question.section}\n\n"
                fallback += f"This variable has {len(question.response_items)} response options."
            else:
                fallback = f"Variable {variable_name} not found in the dataset. Please check the variable name and try again."

            # Cache the fallback response
            if 'explain' not in self._memory_cache:
                self._memory_cache['explain'] = {}
            self._memory_cache['explain'][cache_key] = fallback
            self.cache_manager.set(cache_key, fallback)

            return fallback

    def compare_waves(self, variable_name: str, waves: List[str]) -> str:
        """
        Compare a variable across different waves with error handling.

        Args:
            variable_name: The variable to compare
            waves: List of waves to compare

        Returns:
            Comparison text
        """
        # Create cache key based on variable name and waves
        cache_key = self._get_cache_key('compare', variable_name, tuple(sorted(waves)))

        # Check memory cache first
        if 'compare' in self._memory_cache and cache_key in self._memory_cache['compare']:
            print(f"Using memory cache for comparison of {variable_name} across waves")
            return self._memory_cache['compare'][cache_key]

        # Check disk cache
        cached_comparison = self.cache_manager.get(cache_key)
        if cached_comparison:
            print(f"Using disk cache for comparison of {variable_name} across waves")
            # Ensure compare cache exists in memory
            if 'compare' not in self._memory_cache:
                self._memory_cache['compare'] = {}
            self._memory_cache['compare'][cache_key] = cached_comparison
            return cached_comparison

        try:
            # Get questions across waves
            questions = []
            for wave in waves:
                # Filter in the data manager
                filtered = self.data_manager.filter_questions(
                    filters={"variable_name": variable_name, "wave": wave},
                    limit=1
                )
                if filtered:
                    questions.append(filtered[0])

            if not questions:
                comparison = f"Variable {variable_name} not found in the specified waves."

                # Cache the negative result
                if 'compare' not in self._memory_cache:
                    self._memory_cache['compare'] = {}
                self._memory_cache['compare'][cache_key] = comparison
                self.cache_manager.set(cache_key, comparison)

                return comparison

            if len(questions) < 2:
                comparison = f"Variable {variable_name} was only found in {len(questions)} wave. At least 2 waves are needed for comparison."

                # Cache the insufficient result
                if 'compare' not in self._memory_cache:
                    self._memory_cache['compare'] = {}
                self._memory_cache['compare'][cache_key] = comparison
                self.cache_manager.set(cache_key, comparison)

                return comparison

            # Create context for each question
            context_parts = []
            for i, question in enumerate(questions):
                context = create_structured_context(question)
                context_parts.append(f"[WAVE: {question.wave}]\n{context}")

            context_text = "\n\n".join(context_parts)

            # Prompt template
            template = """
                You are an AI assistant analyzing longitudinal survey data.

                TASK: Compare the variable {variable_name} across different waves.

                DATA:
                {context}

                Provide a comparison analysis that includes:
                1. Changes in response distributions over time
                2. Notable trends or patterns
                3. Possible explanations for any changes
                4. Implications of these findings

                COMPARISON ANALYSIS:
                """

            prompt = PromptTemplate(
                template=template,
                input_variables=["variable_name", "context"]
            )

            # Create a one-off chain using LCEL pattern
            chain = prompt | self.llm

            # Generate and return comparison
            comparison = chain.invoke({
                "variable_name": variable_name,
                "context": context_text
            })

            # Cache the result
            if 'compare' not in self._memory_cache:
                self._memory_cache['compare'] = {}
            self._memory_cache['compare'][cache_key] = comparison
            self.cache_manager.set(cache_key, comparison)

            return comparison

        except Exception as e:
            print(f"Error in compare_waves: {e}")

            # Fallback response
            if questions:
                fallback = f"Comparing variable {variable_name} across {len(questions)} waves:\n\n"

                for q in questions:
                    fallback += f"- Wave {q.wave}: {q.description}\n"
                    fallback += f"  Question: {q.question}\n"
                    fallback += f"  Response options: {len(q.response_items)}\n\n"
            else:
                fallback = f"Variable {variable_name} not found in the specified waves. Please check the variable name and waves and try again."

            # Cache the fallback response
            if 'compare' not in self._memory_cache:
                self._memory_cache['compare'] = {}
            self._memory_cache['compare'][cache_key] = fallback
            self.cache_manager.set(cache_key, fallback)

            return fallback