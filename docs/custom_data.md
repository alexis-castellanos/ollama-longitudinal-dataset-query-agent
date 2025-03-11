# Using Custom Data

This guide explains how to use LongitudinalLLM with your own longitudinal datasets.

## Data Requirements

Your data should meet these requirements:

1. **File Format**: CSV files are preferred, but the system can be extended to work with other formats
2. **Longitudinal Structure**: Data should have identifiers and timestamps
3. **Consistency**: Related datasets should share common identifiers for joining

## Quick Start

The simplest way to use your own data:

1. Place your CSV files in a directory
2. Run the application with the `--data` flag:

```bash
streamlit run app.py -- --data /path/to/your/data
```

## Adding Metadata

While the application can work with raw data files, adding metadata about transformations and column evolution will enhance the verification capabilities.

### Creating a Metadata File

Create a file named `metadata.json` in your data directory with this structure:

```json
{
  "dataset_versions": [
    {
      "dataset_name": "example_dataset",
      "version": "1.0",
      "parent_versions": [],
      "transformations": []
    },
    {
      "dataset_name": "example_dataset",
      "version": "2.0",
      "parent_versions": ["example_dataset_v1"],
      "transformations": [
        {
          "operation": "rename_columns",
          "parameters": {
            "old_column_name": "new_column_name"
          },
          "rationale": "Explanation for why this change was made",
          "timestamp": "2023-01-01T12:00:00"
        }
      ]
    }
  ],
  "column_evolutions": [
    {
      "original_name": "old_column_name",
      "dataset": "example_dataset",
      "versions": [
        {
          "name": "new_column_name",
          "transformation": "rename",
          "reason": "Standardized naming convention",
          "timestamp": "2023-01-01T12:00:00"
        }
      ]
    }
  ]
}
```

## Naming Conventions

For automatic version detection, follow these naming conventions:

- Dataset versions: `dataset_name_v1.csv`, `dataset_name_v2.csv`, etc.
- Related datasets: Use common prefixes (e.g., `patient_demographics.csv`, `patient_visits.csv`)

## Programmatic Integration

For advanced usage, you can extend the `DataManager` class:

```python
from src.data_manager import DataManager

class CustomDataManager(DataManager):
    def _load_datasets_from_dir(self, data_dir):
        # Custom loading logic
        pass
    
    def generate_custom_metadata(self):
        # Generate metadata programmatically
        pass
```

## Example: Clinical Data Integration

Here's an example of integrating EHR data:

1. Export your data as CSV files:
   - `patients.csv` - Patient demographics
   - `visits.csv` - Clinical visits
   - `labs.csv` - Laboratory results
   - `medications.csv` - Medication orders

2. Create metadata describing transformations:
   - Column renames for standardization
   - Derived metrics calculations
   - Date format conversions

3. Run the application:

```bash
streamlit run app.py -- --data /path/to/clinical_data
```

## Data Preprocessing Recommendations

Before using your data with LongitudinalLLM, consider these preprocessing steps:

1. **Handle missing values**: Fill or drop missing values appropriately
2. **Standardize date formats**: Convert all dates to ISO format (YYYY-MM-DD)
3. **Ensure consistent IDs**: Verify that identifiers are consistent across datasets
4. **Add column descriptions**: Create good descriptions for vector search matching

## Optimizing for RAG

LongitudinalLLM uses a RAG (Retrieval-Augmented Generation) architecture to understand your datasets. To optimize performance:

### Improving Column Recognition

The system uses vector embeddings to map natural language to your column names. For best results:

1. **Use descriptive column names**: Names like `patient_age` are easier to match than `pat_a`
2. **Add column metadata**: Create a metadata file with detailed descriptions of each column

```json
{
  "column_descriptions": [
    {
      "dataset": "patient_demographics",
      "column": "recovery_score",
      "descriptions": [
        "Patient recovery assessment on a scale of 1-10",
        "How well the patient has recovered from treatment",
        "Recovery level measured by clinicians"
      ]
    }
  ]
}
```

3. **Include domain terminology**: Add domain-specific terms that users might use

### Documentation Recommendations

For complex datasets:

1. Create a data dictionary with detailed explanations of each column
2. Document any transformations applied to the data
3. Explain the relationships between different datasets

For more information on how LongitudinalLLM understands your data schema, see the [RAG Architecture](rag_architecture.md) document.

## Example Custom Data Setup

Here's a complete example:

```
my_longitudinal_data/
├── cohort_v1.csv         # Original cohort data
├── cohort_v2.csv         # Updated cohort with renamed columns
├── measurements_v1.csv   # Original measurements
├── measurements_v2.csv   # Updated measurements with derived metrics
└── metadata.json         # Transformation metadata
```

Run the application:

```bash
streamlit run app.py -- --data ./my_longitudinal_data
```

Or use the CLI:

```bash
python -m src.cli interactive --data ./my_longitudinal_data
```