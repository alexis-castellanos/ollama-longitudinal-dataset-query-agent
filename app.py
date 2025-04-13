import streamlit as st
from src.data_loader import load_data
from src.embedding_generator import prepare_data_for_embeddings, get_recommendations
from src.chroma_db_manager import initialize_chroma_db, store_embeddings_in_chroma
import pandas as pd
import re


def parse_variable_data(doc_str):
    """Parse the complex variable description format into a structured dictionary."""
    # Extract main fields using regex patterns
    variable_info = {}

    # Extract key metadata fields
    patterns = {
        "VariableName": r"VariableName: ([^,]+)",
        "Description": r"Description: ([^,]+)",
        "Section": r"Section: ([^,]+)",
        "Level": r"Level: ([^,]+)",
        "Type": r"Type: ([^,]+)",
        "Width": r"Width: ([^,]+)",
        "Decimals": r"Decimals: ([^,]+)",
        "CAI Reference": r"CAI Reference: ([^,]*)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, doc_str)
        if match:
            variable_info[key] = match.group(1).strip()
        else:
            variable_info[key] = ""

    # Extract Question text
    question_match = re.search(r"Question: (.+?), Response:", doc_str, re.DOTALL)
    if question_match:
        variable_info["Question"] = question_match.group(1).strip()
    else:
        variable_info["Question"] = ""

    # Extract Response data
    response_match = re.search(r"Response: (\{.+\})", doc_str)
    if response_match:
        try:
            # Clean up the response string for proper parsing
            response_str = response_match.group(1)
            # Convert single quotes to double quotes for JSON parsing
            response_str = response_str.replace("'", '"')
            # Handle potential nested quotes in values
            response_dict = {}

            # Parse the response manually
            items = re.findall(r'"([^"]+)": (\d+|"[^"]+")', response_str)
            for key, value in items:
                try:
                    response_dict[key] = int(value)
                except ValueError:
                    response_dict[key] = value.strip('"')

            variable_info["Response"] = response_dict
        except Exception as e:
            variable_info["Response"] = {"Error parsing response": str(e)}
    else:
        variable_info["Response"] = {}

    return variable_info


def display_variable_data(doc_str, record_id):
    """Display the parsed variable data in multiple DataFrames for better readability."""
    # Parse the document string
    parsed_data = parse_variable_data(doc_str)

    # Create a DataFrame for main metadata
    metadata_fields = ["VariableName", "Description", "Section", "Level", "Type", "Width", "Decimals", "CAI Reference"]
    metadata_df = pd.DataFrame(
        [[parsed_data.get(field, "") for field in metadata_fields]],
        columns=metadata_fields
    )

    st.subheader(f"Variable: {parsed_data.get('VariableName', 'Unknown')} (ID: {record_id})")
    st.dataframe(metadata_df, use_container_width=True)

    # Display question text
    st.subheader("Question Text:")
    st.write(parsed_data.get("Question", ""))

    # Display response data as a separate DataFrame with counts
    if parsed_data.get("Response"):
        st.subheader("Response Distribution:")
        response_items = parsed_data.get("Response", {}).items()
        response_df = pd.DataFrame(response_items, columns=["Response Option", "Count"])

        # Convert to numeric where possible
        try:
            response_df["Count"] = pd.to_numeric(response_df["Count"])
        except Exception as e:
            print(f"Conversion failed: {e}")


        # Calculate percentage
        total = response_df["Count"].sum()
        if total > 0:
            response_df["Percentage"] = (response_df["Count"] / total * 100).round(1)
            response_df["Percentage"] = response_df["Percentage"].astype(str) + '%'

        st.dataframe(response_df, use_container_width=True)

        # Add a simple bar chart
        if not response_df.empty and all(isinstance(x, (int, float)) for x in response_df["Count"]):
            st.bar_chart(response_df.set_index("Response Option")["Count"])


# Main app code
def main():
    """Main application entry point"""
    st.title("📊 LongitudinalLLM")
    st.markdown("""
            Ask questions about longitudinal datasets using natural language. 
            This agent will understand your query, find the right data, and explain 
            what transformations have been applied.
            """)

    # Load and preprocess data
    data = load_data()

    # Initialize Chroma DB and store embeddings
    collection = initialize_chroma_db()
    processed_data = prepare_data_for_embeddings(data)
    store_embeddings_in_chroma(processed_data, collection)

    user_input = st.text_area("Enter your query:", height=100)

    submitted = st.button("Submit Query", type="primary")

    if submitted:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Processing query...")
        progress_bar.progress(25)

        # Get recommendations
        documents, ids = get_recommendations(user_input, collection)

        progress_bar.progress(100)
        status_text.text("Complete!")
        progress_bar.empty()
        status_text.empty()

        # Display results
        st.subheader("Results:")
        for doc_str, record_id in zip(documents[0], ids[0]):
            display_variable_data(doc_str, record_id)
            st.markdown("---")


if __name__ == "__main__":
    main()