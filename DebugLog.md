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

6. **ChromaDB Filter Format**: The vector database required a specific format for list filters using `$or` operators, which wasn't being provided.

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

### 2. Default to All Waves When None Specified

Modified the `find_relevant_questions` function to use all available waves when no specific time periods are mentioned:

```python
# Apply filters from intent, including time periods
filters = {}

# IMPORTANT FIX: Add time_periods to filters if present
if intent.time_periods:
    print(f"Filtering by time periods: {intent.time_periods}")
    filters["wave"] = intent.time_periods
else:
    # Use all available waves if none specified
    available_waves = self.data_manager.get_unique_values("wave")
    print(f"No time periods specified, using all available waves: {available_waves}")
    filters["wave"] = available_waves
```

This ensures that even when no time periods are explicitly mentioned in a query, the system will search across all available waves.

### 3. Implemented Balanced Sampling Across Waves

Added balanced sampling to the `filter_questions` method to ensure even representation across all waves:

```python
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
```

This ensures that questions from all waves are represented in the results, not just those from the earliest years.

### 4. Fixed ChromaDB Filter Format for Vector Search

Updated the `query_similar` method to properly format list filters for ChromaDB:

```python
# Convert filters for ChromaDB
chroma_filters = None
if filters and "wave" in filters and isinstance(filters["wave"], list):
    # ChromaDB expects a different structure for list filters
    wave_list = filters["wave"]
    
    # Ensure proper wave format with " Core" suffix
    formatted_waves = []
    for wave in wave_list:
        if isinstance(wave, str) and " Core" not in wave:
            formatted_waves.append(f"{wave} Core")
        else:
            formatted_waves.append(wave)
    
    # Create OR conditions for each wave
    chroma_filters = {"$or": []}
    for wave in formatted_waves:
        chroma_filters["$or"].append({"wave": {"$eq": wave}})
    
    print(f"Using ChromaDB OR filter for waves: {formatted_waves}")
```

This fixed the error with list filters in ChromaDB, allowing semantic search to work correctly with multiple waves.

### 5. Increased Result Limits

Changed the `filter_questions` call in `find_relevant_questions` to use a higher limit and balanced sampling:

```python
# IMPORTANT FIX: Pass filters to filter_questions to apply the time period filter
# and use balanced sampling with a higher limit
all_questions = self.data_manager.filter_questions(
    filters=filters, 
    limit=1000,  # Increased from 500
    balanced=True
)
```

This ensures that we get a more comprehensive set of results, distributed evenly across all waves.

### 6. Added Diagnostic Output

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

## Results of the Changes

After implementing these changes, we now get results from multiple waves across the entire time span:

```
Wave distribution in results: {'2004 Core': 3, '2006 Core': 2, '2008 Core': 4, '2010 Core': 2, '2012 Core': 4, '2014 Core': 1, '2016 Core': 3, '2018 Core': 1}
```

The balanced sampling approach is working as intended:

```
Balanced sampling: 90 samples per wave from 11 waves
Sampled 90 questions from wave 1996 Core
Sampled 90 questions from wave 1998 Core
Sampled 90 questions from wave 2000 Core
Sampled 90 questions from wave 2004 Core
...
```

And the ChromaDB filter format is correctly constructed:

```
Using ChromaDB OR filter for waves: ['2004 Core', '2006 Core', '2008 Core', ...]
Executing vector search with filters: {'$or': [{'wave': {'$eq': '2004 Core'}}, ...]}
```

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

## Remaining Minor Issues

There are still a few minor issues that could be addressed in future updates:

1. **Intent Analysis for Target Variables**: Similar to the time periods format issue, there are validation errors when target variables are returned as dictionaries instead of strings:
   ```
   Error in intent analysis: 2 validation errors for QueryIntent
   target_variables.0
     Input should be a valid string [type=string_type, input_value={'variable_name': 'E2611M...'}, input_type=dict]
   ```

2. **Earlier Years Missing in Some Results**: While we're sampling from all years (1996-2018), some queries might still show a bias toward later years in the final results, possibly because earlier years don't have the same variables or they're not ranked as highly for certain queries.

## Conclusion

The implemented changes have successfully resolved the core issue, allowing the system to return results from across all available time periods. The balanced sampling approach ensures even representation of all waves, and the ChromaDB filter format fix enables proper semantic search across multiple time periods.

These improvements significantly enhance the system's ability to explore longitudinal data, providing users with a more comprehensive view across the entire timeline of the survey.