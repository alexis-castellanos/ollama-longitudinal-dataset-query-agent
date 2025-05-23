import json
import re
from typing import List, Dict, Any, Optional
import numpy as np
from src.data_models import SurveyResponse


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from a text string.

    Args:
        text: Text that may contain a JSON object

    Returns:
        Extracted JSON as a dictionary, or None if no valid JSON found
    """
    try:
        # Try to find JSON object between curly braces
        json_pattern = r'({[\s\S]*?})'
        match = re.search(json_pattern, text)

        if match:
            json_str = match.group(1)
            return json.loads(json_str)

        return None

    except Exception:
        return None


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Trim whitespace
    text = text.strip()
    return text


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


def parse_variable_name(text: str) -> str:
    """Extract a likely variable name from text.

    Args:
        text: Text that may contain a variable name

    Returns:
        Extracted variable name or empty string
    """
    # Look for patterns like "variable X" or "X variable"
    patterns = [
        r'variable\s+([A-Z0-9]+)',  # "variable ABC123"
        r'([A-Z][A-Z0-9]+)\s+variable',  # "ABC123 variable"
        r'\b([A-Z][A-Z0-9]{2,})\b'  # Just look for uppercase acronyms with numbers
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return ""


def find_wave_references(text: str) -> List[str]:
    """Find references to waves in text.

    Args:
        text: Text that may contain wave references

    Returns:
        List of wave references found
    """
    # Look for patterns like "wave X", "X wave", years, etc.
    patterns = [
        r'wave\s+([a-z0-9]+)',  # "wave 1", "wave A"
        r'([a-z0-9]+)\s+wave',  # "1 wave", "A wave"
        r'(20\d\d)\s+core',  # "2016 core"
        r'(20\d\d)\s+survey',  # "2016 survey"
    ]

    waves = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            waves.append(match.group(1))

    return waves


def create_structured_context(question, include_wave=True):
    """
    Create a structured context for a survey question with enhanced wave information.

    Args:
        question: SurveyQuestion object
        include_wave: Whether to include wave information

    Returns:
        Structured context string
    """
    context_parts = []

    # Include wave prominently if requested
    if include_wave:
        context_parts.append(f"WAVE: {question.wave}")

    context_parts.extend([
        f"VARIABLE: {question.variable_name}",
        f"DESCRIPTION: {question.description}",
        f"QUESTION: {question.question}",
    ])

    # Add response information
    if question.response_items and len(question.response_items) > 0:
        response_parts = ["RESPONSE OPTIONS:"]
        for resp in question.response_items:
            count_info = f" (Count: {resp.count})" if resp.count is not None else ""
            response_parts.append(f"- {resp.option}{count_info}")
        context_parts.append("\n".join(response_parts))

    # Additional metadata
    metadata_parts = ["METADATA:"]
    metadata_parts.append(f"- Section: {question.section}")
    metadata_parts.append(f"- Level: {question.level}")
    metadata_parts.append(f"- Type: {question.type}")
    if include_wave and question.wave:  # Add wave to metadata too for redundancy
        metadata_parts.append(f"- Wave: {question.wave}")

    context_parts.append("\n".join(metadata_parts))

    return "\n\n".join(context_parts)


def normalize_query(query: str) -> str:
    """
    Normalize a query string for consistent caching and matching.

    Args:
        query: Original query string

    Returns:
        Normalized query string
    """
    # Convert to lowercase
    normalized = query.lower()

    # Remove special characters except spaces
    normalized = re.sub(r'[^\w\s]', '', normalized)

    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized


def format_filters_description(filters: Dict[str, Any]) -> str:
    """
    Format filter criteria into a human-readable description.

    Args:
        filters: Dictionary of filter criteria

    Returns:
        Human-readable description of filters
    """
    if not filters:
        return "No filters applied"

    descriptions = []

    for key, value in filters.items():
        if key == "wave":
            if isinstance(value, list):
                descriptions.append(f"Waves: {', '.join(value)}")
            else:
                descriptions.append(f"Wave: {value}")
        elif key == "section":
            if isinstance(value, list):
                descriptions.append(f"Sections: {', '.join(value)}")
            else:
                descriptions.append(f"Section: {value}")
        elif key == "variable_name":
            if isinstance(value, list):
                descriptions.append(f"Variables: {', '.join(value)}")
            else:
                descriptions.append(f"Variable: {value}")
        elif key == "level":
            if isinstance(value, list):
                descriptions.append(f"Levels: {', '.join(value)}")
            else:
                descriptions.append(f"Level: {value}")
        else:
            if isinstance(value, list):
                descriptions.append(f"{key}: {', '.join(map(str, value))}")
            else:
                descriptions.append(f"{key}: {value}")

    return ", ".join(descriptions)


# def format_response_summary(response_items, max_items=5):
#     """
#     Format response items into a concise summary.
#
#     Args:
#         response_items: List of response items
#         max_items: Maximum number of items to include
#
#     Returns:
#         Formatted summary string
#     """
#     if not response_items:
#         return "No response data available"
#
#     # Sort by count (if available) to show most common responses first
#     sorted_items = sorted(
#         response_items,
#         key=lambda x: x.count if x.count is not None else 0,
#         reverse=True
#     )
#
#     # Format items
#     summary_parts = []
#     for i, item in enumerate(sorted_items[:max_items]):
#         count_info = f" (Count: {item.count})" if item.count is not None else ""
#         summary_parts.append(f"{item.option}{count_info}")
#
#     # Add ellipsis if there are more items
#     if len(sorted_items) > max_items:
#         summary_parts.append(f"... and {len(sorted_items) - max_items} more options")
#
#     return "; ".join(summary_parts)