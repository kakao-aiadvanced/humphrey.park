import os
from pprint import pprint

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient
from workflow import workflow

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

st.set_page_config(
    page_title="Research Assistant",
    page_icon=":orange_heart:",
)
st.title("Document Assistant powered by LLM")

app = workflow.compile()


def main() -> None:
    # Get topic for report
    input_topic = st.text_input(
        ":female-scientist: Enter a topic",
        value="Where does Messi play right now?",
    )
    # Button to generate report
    generate_report = st.button("Query")
    if generate_report:
        st.session_state["topic"] = input_topic

    if "topic" in st.session_state:
        report_topic = st.session_state["topic"]

        inputs = {"question": report_topic}

        try:
            with st.spinner("Querying LLM"):
                for output in app.stream(inputs, {"recursion_limit": 10}):
                    for key, value in output.items():
                        pprint(f"Finished running: {key}:")
        except:
            value["generation"] = "No answer generated"

        with st.spinner("Generating Report"):
            final_report = value["generation"]
            st.markdown(final_report)

main()
