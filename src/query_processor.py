import json
import re
from typing import List, Dict, Any, Optional

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field

from src.data_manager import DataManager
# Local imports
from src.data_models import QueryResult, UserQuery
from src.utils import create_structured_context, format_filters_description


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

        # Initialize intent analysis chain
        self.intent_chain = self._create_intent_chain()

        # Initialize answer generation chain
        self.answer_chain = self._create_answer_chain()

    def _create_intent_chain(self):
        """Create the chain for query intent analysis."""
        # Create the parser
        parser = PydanticOutputParser(pydantic_object=QueryIntent)

        # Intent analysis prompt
        intent_template = """
        You are an AI assistant analyzing user queries about longitudinal survey data.

        Given a user query, determine the primary intent, extract relevant filters,
        and identify any specific variables or time periods mentioned.

        Available waves: {available_waves}
        Available sections: {available_sections}
        Available variables: {sample_variables}

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

    def analyze_intent(self, query: str) -> QueryIntent:
        """
        Analyze the intent of a user query with improved error handling.

        Args:
            query: The user query string

        Returns:
            QueryIntent object with detected intent information
        """
        try:
            # Get available metadata for context
            available_waves = self.data_manager.get_unique_values("wave")
            available_sections = self.data_manager.get_unique_values("section")
            sample_variables = self.data_manager.df["variable_name"].sample(
                min(20, len(self.data_manager.df))
            ).tolist()

            # Create a more direct prompt to get just JSON (bypassing the chain)
            direct_template = """
            Analyze this query about survey data: "{query}"

            Available waves: {waves}
            Available sections: {sections}
            Available variables: {variables}

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

                    # Check if target_variables is a list of dictionaries and convert to strings
                    if "target_variables" in data and isinstance(data["target_variables"], list):
                        # Check if elements are dictionaries with a 'name' field
                        for i, var in enumerate(data["target_variables"]):
                            if isinstance(var, dict) and "name" in var:
                                # Replace the dictionary with just the name string
                                data["target_variables"][i] = var["name"]

                    # Create the QueryIntent object
                    intent = QueryIntent(**data)
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

        # Try to extract any variable names (e.g., KLB033)
        variable_pattern = r'\b([A-Z]{2,3}\d{3}[A-Z]?)\b'
        target_variables = re.findall(variable_pattern, query)

        # Extract any mentioned time periods (if waves are numeric)
        time_periods = []
        wave_pattern = r'\bwave\s+(\d+|[ivxlcdm]+)\b'
        wave_matches = re.findall(wave_pattern, query.lower())
        if wave_matches:
            time_periods = wave_matches

        # Create fallback intent
        return QueryIntent(
            primary_intent=primary_intent,
            secondary_intent=None,
            filter_criteria={},
            target_variables=target_variables,
            time_periods=time_periods,
            analysis_type=None,
            confidence=0.6
        )

    def find_relevant_questions(self, query: str, intent: QueryIntent, limit: int = 20) -> List[QueryResult]:
        """
        Find relevant survey questions based on query and intent with better handling
        for larger result sets.

        Args:
            query: The original user query
            intent: The analyzed query intent
            limit: Maximum number of results to return (default increased to 20)

        Returns:
            List of QueryResult objects
        """
        # Start with an empty list
        results = []

        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        query_terms = re.findall(r'\b\w+\b', query_lower)

        # Skip single-word general terms that might match too many variables
        if len(query_terms) == 1 and query_terms[0] in ['variable', 'question', 'survey', 'data']:
            query_terms = []

        # If specific variables were mentioned, prioritize those
        if intent.target_variables:
            for var_name in intent.target_variables:
                question = self.data_manager.get_question_by_variable(var_name)
                if question:
                    results.append(
                        QueryResult(
                            question=question,
                            similarity_score=1.0,  # Perfect match
                            relevance_explanation=f"Exact match for requested variable {var_name}"
                        )
                    )

        # Apply filters from intent
        filters = {}

        # Get a larger set of questions to search through
        try:
            all_questions = self.data_manager.filter_questions(limit=500)

            # Prioritize direct keyword matches with categorization
            exact_matches = []  # For perfect or near-perfect matches
            good_matches = []  # For good but not perfect matches
            partial_matches = []  # For partial matches

            for question in all_questions:
                description_lower = question.description.lower() if question.description else ""
                question_text_lower = question.question.lower() if question.question else ""
                combined_text = f"{description_lower} {question_text_lower}"

                # Calculate match quality metrics
                exact_phrase_match = query_lower in combined_text

                # Count matching terms
                matching_terms = sum(1 for term in query_terms if term in combined_text)
                match_ratio = matching_terms / len(query_terms) if query_terms else 0

                # Matching score calculation that considers phrase matches and term matches
                if exact_phrase_match:
                    # Direct phrase match (highest priority)
                    exact_matches.append(
                        QueryResult(
                            question=question,
                            similarity_score=0.95,
                            relevance_explanation=f"Direct match for query phrase in variable text"
                        )
                    )
                elif match_ratio == 1.0:
                    # All terms match but not as a complete phrase
                    exact_matches.append(
                        QueryResult(
                            question=question,
                            similarity_score=0.9,
                            relevance_explanation=f"All query terms found in variable text"
                        )
                    )
                elif match_ratio >= 0.75:
                    # Most terms match
                    good_matches.append(
                        QueryResult(
                            question=question,
                            similarity_score=0.85,
                            relevance_explanation=f"Most query terms found in variable text"
                        )
                    )
                elif match_ratio >= 0.5:
                    # Half or more terms match
                    partial_matches.append(
                        QueryResult(
                            question=question,
                            similarity_score=0.75,
                            relevance_explanation=f"Some query terms found in variable text"
                        )
                    )

            # Add matches in priority order, avoiding duplicates
            existing_vars = {result.question.variable_name for result in results}

            # Define a helper function to add matches without duplicates
            def add_matches(matches, existing_vars, results):
                for match in matches:
                    if match.question.variable_name not in existing_vars:
                        results.append(match)
                        existing_vars.add(match.question.variable_name)

            # Add matches in priority order
            add_matches(exact_matches, existing_vars, results)
            add_matches(good_matches, existing_vars, results)
            add_matches(partial_matches, existing_vars, results)

        except Exception as e:
            print(f"Error in direct text matching: {e}")

        # If we haven't reached the limit yet, perform semantic search
        if len(results) < limit:
            remaining = limit - len(results)
            existing_vars = {result.question.variable_name for result in results}

        # Sort results by similarity score to ensure most relevant are first
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        return results[:limit]  # Return at most 'limit' results

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
        try:
            # Check if we have results
            if not results:
                return "I couldn't find any survey questions related to your query. Please try rephrasing or being more specific about what you're looking for."

            # Organize results into multiple tiers by relevance
            exact_matches = []  # 0.9-1.0: Perfect or near-perfect matches
            high_relevance = []  # 0.8-0.89: Very relevant
            medium_relevance = []  # 0.7-0.79: Somewhat relevant
            low_relevance = []  # Below 0.7: Limited relevance

            for result in results:
                if result.similarity_score >= 0.9:
                    exact_matches.append(result)
                elif result.similarity_score >= 0.8:
                    high_relevance.append(result)
                elif result.similarity_score >= 0.7:
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
            return "\n".join(response_parts)

        except Exception as e:
            print(f"Error in structured answer generation: {e}")

            # [Fallback logic remains the same]

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

            # Find relevant questions
            results = self.find_relevant_questions(
                query=user_query.query_text,
                intent=intent,
                limit=user_query.limit
            )

            # If no results found through intent analysis, try direct semantic search
            if not results:
                try:
                    results = self.data_manager.query_similar(
                        query_text=user_query.query_text,
                        filters={},
                        limit=user_query.limit
                    )
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


    def explain_variable(self, variable_name: str) -> str:
        """
        Generate an explanation for a specific variable with error handling.

        Args:
            variable_name: The variable to explain

        Returns:
            Explanation text
        """
        try:
            # Get the question
            question = self.data_manager.get_question_by_variable(variable_name)

            if not question:
                return f"Variable {variable_name} not found in the dataset."

            # Create context
            context = create_structured_context(question)

            # Prompt template
            template = """
                You are an AI assistant helping users understand survey data.
    
                VARIABLE: {variable_name}
    
                DETAILS:
                {context}
    
                Provide a clear explanation of this survey variable, including:
                1. What the question is measuring
                2. The response options and their distribution
                3. Any notable patterns in the responses
                4. Potential significance of this variable in longitudinal research
    
                EXPLANATION:
                """

            prompt = PromptTemplate(
                template=template,
                input_variables=["variable_name", "context"]
            )

            # Create a one-off chain using LCEL pattern
            chain = prompt | self.llm

            # Generate and return explanation
            return chain.invoke({
                "variable_name": variable_name,
                "context": context
            })

        except Exception as e:
            print(f"Error in explain_variable: {e}")

            # Fallback response
            if question:
                return f"Variable {variable_name} ({question.description}) from wave {question.wave} asks: '{question.question}'. The response data shows {len(question.response_items)} different response options."
            else:
                return f"Variable {variable_name} not found in the dataset. Please check the variable name and try again."


    def compare_waves(self, variable_name: str, waves: List[str]) -> str:
        """
        Compare a variable across different waves with error handling.

        Args:
            variable_name: The variable to compare
            waves: List of waves to compare

        Returns:
            Comparison text
        """
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
                return f"Variable {variable_name} not found in the specified waves."

            if len(questions) < 2:
                return f"Variable {variable_name} was only found in {len(questions)} wave. At least 2 waves are needed for comparison."

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
            return chain.invoke({
                "variable_name": variable_name,
                "context": context_text
            })

        except Exception as e:
            print(f"Error in compare_waves: {e}")

            # Fallback response
            if questions:
                response = f"Comparing variable {variable_name} across {len(questions)} waves:\n\n"

                for q in questions:
                    response += f"- Wave {q.wave}: {q.description}\n"
                    response += f"  Question: {q.question}\n"
                    response += f"  Response options: {len(q.response_items)}\n\n"

                return response
            else:
                return f"Variable {variable_name} not found in the specified waves. Please check the variable name and waves and try again."