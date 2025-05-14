# Data Harmonization LLM - Code Changes and Troubleshooting

## Issue Overview

The Data-Harmonization-LLM system was encountering a problem where queries were only returning results from the first two years (2004 Core and 2006 Core) despite the system having access to data from multiple waves (1996-2018). This issue was affecting the core functionality of the application, preventing users from exploring longitudinal data across all available time periods.

## Root Causes Identified

After thorough analysis, we identified several related issues:

1. **Intent Analysis Format Mismatch**: The LLM was returning time periods in dictionary format with start/end dates (e.g., `{"start_date": 1996, "end_date": 2000}`), but the `QueryIntent` model expected strings (e.g., "1996 Core").

2. **Validation Errors**: This format mismatch caused Pydantic validation errors, resulting in empty time periods (`time_periods=[]`) in the intent analysis.

3. **Missing Filter Application**: Without time periods identified correctly, the system wasn't applying wave filters to the query.

4. **Wave Formatting Inconsistency**: Time periods sometimes lacked the " Core" suffix required to match database records.

5. **Result Limitation**: Even with proper filters, the system was only retrieving 500 questions at a time, which limited results to the earliest waves when sorted chronologically.

## Changes Implemented

### 1. Enhanced Time Period Handling in Intent Analysis

Updated the `analyze_intent` function in `query_processor.py` to properly handle dictionary-formatted time periods:

```python
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
```

This code now properly handles multiple formats of time periods from the LLM output and ensures they all have the correct " Core" suffix.

### 2. Updated Filter Application in Query Processing

Modified the `find_relevant_questions` function in `query_processor.py` to correctly use time periods in filters:

```python
# Apply filters from intent, including time periods
filters = {}

# IMPORTANT FIX: Add time_periods to filters if present
if intent.time_periods:
    print(f"Filtering by time periods: {intent.time_periods}")
    filters["wave"] = intent.time_periods

# Check if there are any other filter criteria in the intent
if intent.filter_criteria:
    # Merge with existing filters
    filters.update(intent.filter_criteria)
    
print(f"Applied filters: {filters}")

# Get a larger set of questions to search through
try:
    # IMPORTANT FIX: Pass filters to filter_questions to apply the time period filter
    all_questions = self.data_manager.filter_questions(filters=filters, limit=500)
    print(f"Retrieved {len(all_questions)} questions after filtering")
```

This ensures the time periods information is properly passed to the filtering function.

### 3. Enhanced Semantic Search with Filters

Updated the semantic search section of `find_relevant_questions` to use the same filters:

```python
# If we haven't reached the limit yet, perform semantic search
if len(results) < limit:
    remaining = limit - len(results)
    existing_vars = {result.question.variable_name for result in results}
    
    try:
        # IMPORTANT FIX: Also pass filters to semantic search for consistency
        semantic_results = self.data_manager.query_similar(
            query_text=query,
            filters=filters,  # Pass the filters including time_periods
            limit=remaining
        )
        
        # Filter out variables we already have
        for result in semantic_results:
            if result.question.variable_name not in existing_vars:
                results.append(result)
                existing_vars.add(result.question.variable_name)
                
        print(f"Added {len(semantic_results)} results from semantic search")
    except Exception as e:
        print(f"Error in semantic search: {e}")
```

This ensures consistent filtering across both direct matching and semantic search.

### 4. Added Diagnostic Output

Added diagnostic counting of waves in results to help debug distribution issues:

```python
# Debug: Print distribution of waves in results
wave_distribution = {}
for result in final_results:
    wave = result.question.wave
    wave_distribution[wave] = wave_distribution.get(wave, 0) + 1
print(f"Wave distribution in results: {wave_distribution}")
```

This helps track which waves are represented in the final results.

## Cache Management Issues

We encountered persistent caching issues where old results were being retrieved even after code changes. This was because:

1. The system implements multi-level caching (memory and disk) for performance optimization
2. Query results are cached based on query text and intent, so old results with incorrect filtering were being retrieved

### Manual Cache Clearing Solution

We found that manually deleting the cache directory before running the application ensures fresh results:

```bash
# From the project root directory
rm -rf ./cache
streamlit run app.py
```

This forces the system to re-run the entire query processing pipeline with our updated code.

## Remaining Issues and Next Steps

Despite our changes, we're still experiencing some limitations in result distribution across waves. Current diagnostics suggest:

1. **Limit Constraints**: The hardcoded limit of 500 questions in `filter_questions` may restrict results to earlier waves due to the order of retrieval.

2. **Sampling Imbalance**: Results are not evenly distributed across waves because the system takes the first N results without considering wave distribution.

### Proposed Next Steps

1. **Implement Balanced Sampling**: Modify `filter_questions` to sample evenly from each wave instead of taking the first N results.

2. **Increase or Remove Result Limits**: Consider increasing or removing the 500-question limit to ensure comprehensive coverage.

3. **Default to All Waves**: When no time periods are specified in a query, explicitly use all available waves instead of using no filter.

4. **Improve Wave Detection**: Enhance the intent analysis to better detect queries that should include all waves even when not explicitly stated.

These changes should further improve the system's ability to explore longitudinal data across all available time periods.

## Conclusion

The implemented changes have substantially improved the system's handling of time periods in query processing, but additional work is needed to ensure balanced representation across all waves. By addressing the remaining issues with result sampling and default wave filtering, we can provide users with a more comprehensive view of longitudinal data.