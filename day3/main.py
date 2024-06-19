import os

import streamlit as st
from assistant import ResearchAssistant
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

st.set_page_config(
    page_title="Research Assistant",
    page_icon=":orange_heart:",
)
st.title("Research Assistant powered by LLM")

def main() -> None:
    # Get model
    llm_model = st.sidebar.selectbox(
        "Select Model",
        options=[
            "gpt-4o",
            "llama3"
            "qwen2"
        ],
    )
    # Set assistant_type in session state
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        st.rerun()

    # Get topic for report
    input_topic = st.text_input(
        ":female-scientist: Enter a topic",
        value="Where does Messi play right now?",
    )
    # Button to generate report
    generate_report = st.button("Generate Report")
    if generate_report:
        st.session_state["topic"] = input_topic

    st.sidebar.markdown("## Trending Topics")

    if st.sidebar.button("AI in Healthcare"):
        st.session_state["topic"] = "AI in Healthcare"

    if st.sidebar.button("Language Agent Tree Search"):
        st.session_state["topic"] = "Language Agent Tree Search"

    if st.sidebar.button("Chromatic Homotopy Theory"):
        st.session_state["topic"] = "Chromatic Homotopy Theory"

    if "topic" in st.session_state:
        report_topic = st.session_state["topic"]
        research_assistant = ResearchAssistant(model=llm_model)
        #
        # with st.status("Searching Web", expanded=True) as status:
        #     with st.container():
        #         tavily_container = st.empty()
        #         tavily_search_results = tavily.search(
        #             query=report_topic, search_depth="advanced"
        #         )
        #         if tavily_search_results:
        #             tavily_container.markdown(tavily_search_results)
        #     status.update(label="Web Search Complete", state="complete", expanded=False)
        #
        # if not tavily_search_results:
        #     st.write("Sorry report generation failed. Please try again.")
        #     return

        with st.spinner("Generating Report"):
            final_report = research_assistant.generate_report(
                report_topic
            )
            st.markdown(final_report)

    st.sidebar.markdown("---")
    if st.sidebar.button("Restart"):
        st.rerun()


main()
