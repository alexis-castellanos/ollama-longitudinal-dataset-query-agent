import re
from typing import List, Dict, Any
import numpy as np
from src.data_models import SurveyResponse


def clean_text(text: str) -> str:
    """
    Clean and normalize text for better embedding and search.

    Args:
        text: The input text to clean

    Returns:
        Cleaned text string
    """
    if not text:
        return ""

    # Replace newlines with spaces
    text = text.replace('\n', ' ')

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove common survey formatting elements
    text = re.sub(r'\(Mark \(X\) .*?\)', '', text)

    # Strip and return
    return text.strip()


def normalize_query(query: str) -> str:
    """
    Normalize a query string to improve cache hit rates.

    Normalization steps:
    1. Convert to lowercase
    2. Remove extra whitespace
    3. Remove punctuation
    4. Standardize word forms (like "sad", "sadness" → similar forms)

    Args:
        query: Original query string

    Returns:
        Normalized query string
    """
    import re

    # Lowercase and strip whitespace
    query = query.lower().strip()

    # Remove punctuation and extra spaces
    query = re.sub(r'[^\w\s]', ' ', query)
    query = re.sub(r'\s+', ' ', query)

    # Optional: Use stemming or lemmatization for word normalization
    # This requires nltk to be installed:
    # try:
    #     from nltk.stem import PorterStemmer
    #     stemmer = PorterStemmer()
    #     words = query.split()
    #     words = [stemmer.stem(word) for word in words]
    #     query = ' '.join(words)
    # except ImportError:
    #     pass  # Skip stemming if nltk is not available

    # Remove common words that don't affect meaning
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'about', 'for', 'of', 'in', 'to', 'with'}
    words = query.split()
    words = [word for word in words if word not in stop_words]

    # Sort words for order independence (optional)
    # Enabling this would treat "depression symptoms" and "symptoms of depression" as the same query
    # words.sort()

    # Join back into a string
    return ' '.join(words)


def format_response_summary(responses: List[SurveyResponse]) -> str:
    """
    Create a readable summary of response options and counts.

    Args:
        responses: List of SurveyResponse objects

    Returns:
        Formatted response summary string
    """
    # Filter out empty responses and INAP
    valid_responses = [r for r in responses if r.count is not None
                       and "INAP" not in r.option and r.count != ""]

    # No valid responses
    if not valid_responses:
        return "No response data available"

    # Calculate total (excluding INAP)
    total = sum(r.count for r in valid_responses if isinstance(r.count, (int, float)))

    # Format each response with percentage
    summary_parts = []
    for resp in valid_responses:
        if isinstance(resp.count, (int, float)) and total > 0:
            percentage = (resp.count / total) * 100
            # Clean up option text
            option = resp.option
            if ". " in option:  # Remove numbering if present
                option = option.split(". ", 1)[1]
            summary_parts.append(f"{option}: {resp.count} ({percentage:.1f}%)")

    return ", ".join(summary_parts)


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length with ellipsis.

    Args:
        text: Input text
        max_length: Maximum length to keep

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."


def get_response_stats(responses: List[SurveyResponse]) -> Dict[str, Any]:
    """
    Calculate statistics for numeric responses.

    Args:
        responses: List of SurveyResponse objects

    Returns:
        Dictionary of statistics
    """
    # Extract numeric values (usually 1-5 scales in surveys)
    values = []
    for resp in responses:
        if resp.count is not None and "INAP" not in resp.option:
            # Try to extract numeric value from option (e.g., "1. STRONGLY DISAGREE" -> 1)
            match = re.match(r'^(\d+)\.', resp.option)
            if match:
                value = int(match.group(1))
                # Add this value to our list, weighted by its count
                values.extend([value] * resp.count)

    # Calculate statistics if we have values
    if not values:
        return {"count": 0}

    return {
        "count": len(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values)
    }


def extract_question_type(question_text: str) -> str:
    """
    Attempt to determine the question type from its text.

    Args:
        question_text: The question text

    Returns:
        Question type string
    """
    question_text = question_text.lower()

    if "agree or disagree" in question_text:
        return "agreement"
    elif "how often" in question_text:
        return "frequency"
    elif "rate" in question_text or "scale" in question_text:
        return "rating"
    elif "yes or no" in question_text:
        return "boolean"
    elif "describe" in question_text:
        return "descriptive"
    else:
        return "other"


def format_filters_description(filters: Dict[str, Any]) -> str:
    """
    Create a human-readable description of applied filters.

    Args:
        filters: Dictionary of filter criteria

    Returns:
        Formatted filter description string
    """
    if not filters:
        return "No filters applied"

    parts = []
    for key, value in filters.items():
        if isinstance(value, list):
            parts.append(f"{key} is one of {', '.join(map(str, value))}")
        else:
            parts.append(f"{key} is {value}")

    return ", ".join(parts)


def create_structured_context(question):
    """
    Create a structured context string for a question to feed to the LLM.

    Args:
        question: A SurveyQuestion object

    Returns:
        Formatted context string
    """
    # Get response stats if applicable
    stats = get_response_stats(question.response_items)

    context = f"""
QUESTION ID: {question.variable_name}
WAVE: {question.wave}
SECTION: {question.section}
DESCRIPTION: {question.description}

QUESTION TEXT:
{question.question}

RESPONSE OPTIONS AND COUNTS:
"""

    for resp in question.response_items:
        if resp.count is not None and resp.count != "":
            context += f"- {resp.option}: {resp.count}\n"

    # Add statistics if available
    if stats.get("count", 0) > 0:
        context += f"""
STATISTICS:
- Total responses: {stats.get('count')}
- Mean response value: {stats.get('mean', 'N/A'):.2f}
- Median response value: {stats.get('median', 'N/A'):.1f}
- Standard deviation: {stats.get('std', 'N/A'):.2f}
- Range: {stats.get('min', 'N/A')} to {stats.get('max', 'N/A')}
"""

    return context