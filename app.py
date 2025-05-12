import re
import uuid
import pandas as pd
import streamlit as st

# Local imports
from src.data_manager import DataManager
from src.data_models import SurveyQuestion
from src.query_processor import QueryProcessor, UserQuery

# Configure the page
st.set_page_config(
    page_title="Longitudinal LLM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Constants
MODEL_NAME = "llama3.2:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"
DATA_PATH = "data/hrs_data.json"

# Session state initialization
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.data_manager = None
    st.session_state.query_processor = None
    st.session_state.chat_history = []
    st.session_state.selected_question = None
    st.session_state.filter_params = {}
    st.session_state.available_waves = []
    st.session_state.available_sections = []


def initialize_system():
    """Initialize the data manager and query processor with fixed settings."""
    with st.spinner("Loading survey data and initializing models..."):
        # Initialize data manager
        data_manager = DataManager(
            data_path=DATA_PATH,
            embeddings_model=EMBEDDING_MODEL
        )

        # Load data
        data_manager.load_data()

        # Initialize vector database
        data_manager.initialize_vector_db()

        # Check if we need to embed data
        collection = data_manager.collection
        if collection.count() == 0:
            with st.spinner("First-time setup: Embedding survey data..."):
                data_manager.embed_and_store()

        # Create cache manager and assign to data manager
        from src.query_processor import CacheManager
        cache_manager = CacheManager(cache_dir="./cache", expiration_days=7)
        data_manager.cache_manager = cache_manager

        # Store in session state
        st.session_state.data_manager = data_manager

        # Initialize query processor with cache manager
        query_processor = QueryProcessor(
            data_manager=data_manager,
            model_name=MODEL_NAME
        )
        query_processor.cache_manager = cache_manager
        st.session_state.query_processor = query_processor

        # Store metadata
        st.session_state.available_waves = data_manager.get_unique_values("wave")
        st.session_state.available_sections = data_manager.get_unique_values("section")

        # Mark as initialized
        st.session_state.initialized = True

        st.success("System initialized successfully!")


def display_question_details(question: SurveyQuestion):
    """Display detailed information about a survey question."""
    st.subheader(f"{question.variable_name}: {question.description}")
    st.caption(f"Wave: {question.wave} | Section: {question.section}")

    # Display the question text
    st.markdown("#### Question")
    st.markdown(question.question)

    # Display response data
    st.markdown("#### Responses")

    # Convert to DataFrame for better display
    response_data = {
        "Option": [],
        "Count": [],
        "Percentage": []
    }

    total_valid = sum(r.count for r in question.response_items
                      if r.count is not None and "INAP" not in r.option)

    for resp in question.response_items:
        if resp.count is not None:
            response_data["Option"].append(resp.option)
            response_data["Count"].append(resp.count)

            # Calculate percentage for valid responses
            if "INAP" not in resp.option and total_valid > 0:
                percentage = (resp.count / total_valid) * 100
                response_data["Percentage"].append(f"{percentage:.1f}%")
            else:
                response_data["Percentage"].append("N/A")

    # Display as DataFrame
    df = pd.DataFrame(response_data)
    st.dataframe(df, use_container_width=True)

    # Display visualization if applicable
    if total_valid > 0:
        st.markdown("#### Visualization")

        # Filter out INAP
        valid_data = df[~df["Option"].str.contains("INAP")].copy()

        # If there are numeric prefixes (like "1. STRONGLY DISAGREE"), extract them
        valid_data["Sort"] = valid_data["Option"].str.extract(r'^(\d+)\.')
        valid_data["Sort"] = pd.to_numeric(valid_data["Sort"], errors='coerce')

        # Sort by the extracted prefix if available
        if not valid_data["Sort"].isna().all():
            valid_data = valid_data.sort_values("Sort")

        # Create the chart
        st.bar_chart(valid_data.set_index("Option")["Count"])


def process_user_query(query_text: str):
    """Process a user query and add to chat history."""
    if not query_text.strip():
        return

    # Add user message to chat
    st.session_state.chat_history.append({"role": "user", "content": query_text})

    # Process the query
    with st.spinner("Processing your query..."):
        # Create user query object with any filters from session state
        user_query = UserQuery(
            query_text=query_text,
            filters=st.session_state.filter_params,
            limit=20
        )

        # Process the query
        processed = st.session_state.query_processor.process_query(user_query)

        # Add assistant response to chat
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": processed.answer,
            "results": processed.results,
            "intent": processed.intent
        })


def sidebar():
    """Render a simplified sidebar with just initialization button."""
    st.sidebar.title("Data Assistant")

    # Initialize button
    if not st.session_state.initialized:
        if st.sidebar.button("Initialize System"):
            initialize_system()
    # else:
    #     # Display dataset info
    #     st.sidebar.divider()
    #     st.sidebar.subheader("Dataset Information")
    #
    #     data_manager = st.session_state.data_manager
    #
    #     # Count of questions
    #     question_count = len(data_manager.survey_data.questions)
    #     st.sidebar.metric("Total Questions", question_count)
    #
    #     # Waves summary
    #     waves_summary = data_manager.get_waves_summary()
    #     st.sidebar.caption(f"Waves: {len(waves_summary)}")
    #
    #     # Sections summary
    #     sections_summary = data_manager.get_sections_summary()
    #     st.sidebar.caption(f"Sections: {len(sections_summary)}")


def chat_interface():
    """Render the chat interface."""

    st.title("📊 Longitudinal Data Assistant")
    st.markdown("""
        Ask questions about longitudinal datasets using natural language. 
        This agent will understand your query, find the right data, and display the results.
        """)

    # Check if system is initialized
    if not st.session_state.initialized:
        st.info("Please initialize the system using the sidebar.")
        return

    # Initialize a unique conversation ID if not already set
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())[:8]

    # Chat container
    chat_container = st.container()

    # Input container at the bottom
    with st.container():
        query_text = st.chat_input("Ask about the survey data...")
        if query_text:
            process_user_query(query_text)

    # Display chat history
    with chat_container:
        for message_idx, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:  # assistant
                with st.chat_message("assistant"):
                    st.markdown(message["content"])

                    # If there are results, add expanders for each
                    if "results" in message:
                        st.divider()
                        st.caption("Relevant survey questions:")

                        # Create columns for result cards
                        cols = st.columns(3)

                        for i, result in enumerate(message["results"]):
                            col_idx = i % 3
                            with cols[col_idx]:
                                with st.expander(f"{result.question.variable_name}: {result.question.description}"):
                                    st.caption(f"Wave: {result.question.wave}")
                                    st.caption(f"Similarity: {result.similarity_score:.2f}")
                                    st.markdown(result.question.question)

                                    # Button to show details with a guaranteed unique key
                                    # Use conversation_id, message_idx, and result variable_name
                                    btn_key = f"btn_{st.session_state.conversation_id}_{message_idx}_{i}_{result.question.variable_name}"
                                    if st.button("Show Details", key=btn_key):
                                        st.session_state.selected_question = result.question

                    # If there's intent info, add an expander
                    if "intent" in message:
                        with st.expander("Query Analysis"):
                            intent = message["intent"]
                            st.markdown(f"**Primary Intent:** {intent.primary_intent}")
                            if intent.secondary_intent:
                                st.markdown(f"**Secondary Intent:** {intent.secondary_intent}")
                            if intent.analysis_type:
                                st.markdown(f"**Analysis Type:** {intent.analysis_type}")
                            if intent.target_variables:
                                st.markdown(f"**Target Variables:** {', '.join(intent.target_variables)}")
                            if intent.time_periods:
                                st.markdown(f"**Time Periods:** {', '.join(intent.time_periods)}")
                            st.markdown(f"**Confidence:** {intent.confidence:.2f}")

    # If a question is selected, display its details
    if st.session_state.selected_question:
        st.divider()
        display_question_details(st.session_state.selected_question)

        # Button to clear selection
        if st.button("Close Details"):
            st.session_state.selected_question = None


def data_explorer():
    """Render the data explorer interface."""
    st.title("Survey Data Explorer")
    st.markdown("""
    Explore and interact with your survey data with ease.

    Use the tools below to:
    - 🔍 Filter and search through responses
    - 📈 Visualize trends and distributions
    - 📋 View summary statistics and insights

    Start by selecting a dataset or applying filters to dig into specific segments of your data.
    """)
    # Check if system is initialized
    if not st.session_state.initialized:
        st.info("Please initialize the system using the sidebar.")
        return

    # Get data manager
    data_manager = st.session_state.data_manager

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Browse Questions", "Variable Explorer", "Compare Waves"])

    # Tab 1: Browse Questions
    with tab1:
        st.subheader("Browse Survey Questions")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            wave_filter = st.selectbox(
                "Wave",
                options=["All"] + st.session_state.available_waves,
                index=0
            )

        with col2:
            section_filter = st.selectbox(
                "Section",
                options=["All"] + st.session_state.available_sections,
                index=0
            )

        with col3:
            search_text = st.text_input("Search in description", "")

        # Apply filters
        filters = {}
        if wave_filter != "All":
            filters["wave"] = wave_filter
        if section_filter != "All":
            filters["section"] = section_filter

        # Add a limit selector
        limit_options = [50, 100, 200, 500, "All"]

        # Create two columns for the limit selector and page info
        limit_col, info_col = st.columns([1, 3])

        with limit_col:
            # Get the current limit from session state or default to 50
            if 'items_per_page' not in st.session_state:
                st.session_state.items_per_page = 50

            # Initialize current page if not present
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 1

            selected_limit = st.selectbox(
                "Results per page",
                options=limit_options,
                index=limit_options.index(
                    st.session_state.items_per_page) if st.session_state.items_per_page in limit_options else 0,
                key="limit_selector"
            )

            # If the limit changed, update session state and reset to page 1
            if selected_limit != st.session_state.items_per_page:
                st.session_state.items_per_page = selected_limit  # FIX 1: Use selected_limit directly
                # Keep current page as 1 when changing page size
                st.session_state.current_page = 1
                st.rerun()

        # Get questions per page from session state
        questions_per_page = st.session_state.items_per_page

        # Convert "All" to None for data fetching purposes
        query_limit = None if selected_limit == "All" else 5000  # High limit for data fetching

        # Get filtered questions with dynamic limit
        filtered_questions = data_manager.filter_questions(filters=filters, limit=query_limit)

        # Further filter by search text if provided
        if search_text:
            filtered_questions = [
                q for q in filtered_questions
                if search_text.lower() in q.description.lower() or search_text.lower() in q.question.lower()
            ]

        # Implement pagination if there are results
        total_questions = len(filtered_questions)

        # Determine how many items to show per page
        if questions_per_page == "All":
            # Show all results on one page
            display_questions = filtered_questions
            questions_per_page = total_questions

            # Simple results display for all results
            st.markdown(f"<div style='text-align: center'><b>Showing all {total_questions} results</b></div>",
                        unsafe_allow_html=True)
        else:
            # Calculate total pages
            max_pages = max(1, (total_questions + questions_per_page - 1) // questions_per_page)

            # Ensure current page is valid
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 1
            elif st.session_state.current_page > max_pages:
                st.session_state.current_page = max_pages
            elif st.session_state.current_page < 1:  # Ensure page is never less than 1
                st.session_state.current_page = 1

            page_number = st.session_state.current_page

            # Calculate which questions to display based on current page
            start_idx = (page_number - 1) * questions_per_page
            end_idx = min(start_idx + questions_per_page, total_questions)
            display_questions = filtered_questions[start_idx:end_idx]

            # Only show pagination controls if needed
            if total_questions > questions_per_page:
                col1, col2, col3 = st.columns([1, 4, 1])

                # Calculate correct items showing (this should match the selected page size)
                items_showing = end_idx - start_idx

                with col1:
                    # Define a callback for the previous button
                    def prev_page():
                        st.session_state.current_page -= 1
                        # Don't call st.rerun() here to avoid full page refresh

                    st.button("⏪ Previous",
                              disabled=(page_number <= 1),
                              key="prev_btn",
                              on_click=prev_page)

                with col2:
                    start_display = max(1, start_idx + 1)  # Never less than 1
                    end_display = max(start_display, end_idx)  # Never less than start_display

                    st.markdown(
                        f"<div style='text-align: center; padding: 10px;'><b>Showing {start_display}-{end_display} of {total_questions} results (Page {page_number} of {max_pages})</b></div>",
                        unsafe_allow_html=True)

                with col3:
                    # Define a callback for the next button
                    def next_page():
                        st.session_state.current_page += 1

                    st.button("Next ⏩",
                              disabled=(page_number >= max_pages),
                              key="next_btn",
                              on_click=next_page)
            else:
                st.markdown(
                    f"<div style='text-align: center'><b>Showing {len(display_questions)} of {total_questions} results</b></div>",
                    unsafe_allow_html=True)

        # Display as table
        if display_questions:
            question_data = []
            for q in display_questions:
                question_data.append({
                    "Variable": q.variable_name,
                    "Description": q.description,
                    "Wave": q.wave,
                    "Section": q.section
                })

            question_df = pd.DataFrame(question_data)

            # Use Streamlit's dataframe with selection
            selection = st.dataframe(
                question_df,
                use_container_width=True,
                column_config={
                    "Variable": st.column_config.TextColumn("Variable"),
                    "Description": st.column_config.TextColumn("Description"),
                    "Wave": st.column_config.TextColumn("Wave"),
                    "Section": st.column_config.TextColumn("Section")
                },
                hide_index=True
            )

            # Display selected question if user clicks
            if st.button("Show Selected Question"):
                selected_indices = selection.selected_rows
                if selected_indices and len(selected_indices) > 0:
                    selected_index = selected_indices[0]

                    if isinstance(selected_index, dict) and 'index' in selected_index:
                        row_index = selected_index['index']
                    else:
                        row_index = selected_index

                    if 0 <= row_index < len(question_df):
                        selected_var = question_df.iloc[row_index]["Variable"]
                        selected_question = data_manager.get_question_by_variable(selected_var)
                        if selected_question:
                            st.session_state.selected_question = selected_question
                            st.rerun()

    # Tab 2: Variable Explorer
    with tab2:
        st.subheader("Variable Explorer")

        # Variable search
        variable_search = st.text_input("Enter variable name", "")

        if variable_search:
            # Search for the variable
            question = data_manager.get_question_by_variable(variable_search)

            if question:
                # Display the question details
                display_question_details(question)

                # Generate explanation
                with st.spinner("Generating explanation..."):
                    explanation = st.session_state.query_processor.explain_variable(variable_search)
                    st.markdown("### Variable Explanation")
                    st.markdown(explanation)
            else:
                st.error(f"Variable '{variable_search}' not found in the dataset.")

    # Tab 3: Compare Waves
    with tab3:
        st.subheader("Compare Across Waves")

        # Variable selection
        variable_name = st.text_input("Enter variable name to compare", "")

        # Wave selection (multiselect)
        if variable_name:
            # Find all waves that have this variable
            all_waves = data_manager.df[
                data_manager.df["variable_name"] == variable_name
                ]["wave"].unique().tolist()

            if not all_waves:
                st.error(f"Variable '{variable_name}' not found in any wave.")
            else:
                # Let user select waves to compare
                selected_waves = st.multiselect(
                    "Select waves to compare",
                    options=all_waves,
                    default=all_waves[:min(len(all_waves), 3)]  # Default to first 3
                )

                if selected_waves and len(selected_waves) >= 2:
                    # Generate comparison
                    if st.button("Generate Comparison"):
                        with st.spinner("Comparing across waves..."):
                            comparison = st.session_state.query_processor.compare_waves(
                                variable_name=variable_name,
                                waves=selected_waves
                            )
                            st.markdown("### Wave Comparison")
                            st.markdown(comparison)

                            # Get data for visualization
                            data = []
                            for wave in selected_waves:
                                questions = data_manager.filter_questions(
                                    filters={"variable_name": variable_name, "wave": wave},
                                    limit=1
                                )
                                if questions:
                                    q = questions[0]
                                    for resp in q.response_items:
                                        if resp.count is not None and "INAP" not in resp.option:
                                            # Extract numeric value if present
                                            match = re.match(r'^(\d+)\.', resp.option)
                                            option_label = resp.option
                                            if match:
                                                option_label = resp.option.split(". ", 1)[
                                                    1] if ". " in resp.option else resp.option

                                            data.append({
                                                "Wave": wave,
                                                "Option": option_label,
                                                "Count": resp.count
                                            })

                            # Create DataFrame
                            if data:
                                df = pd.DataFrame(data)
                                pivot_df = df.pivot(index="Option", columns="Wave", values="Count").fillna(0)

                                # Plot
                                st.bar_chart(pivot_df)
                elif selected_waves:
                    st.info("Please select at least 2 waves to compare.")


def main():
    """Main application entry point."""
    sidebar()

    # Create tabs for different interfaces
    tab1, tab2 = st.tabs(["Chat Assistant", "Data Explorer"])

    with tab1:
        chat_interface()

    with tab2:
        data_explorer()


if __name__ == "__main__":
    main()