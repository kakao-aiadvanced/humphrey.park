import os
from pprint import pprint

from dotenv import load_dotenv

from workflow import workflow

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


class ResearchAssistant:
    def __init__(self, model: str = "llama3"):
        """Initialize the Research Assistant with a specific model."""
        self.model = model
        self.app = workflow.compile()

    def generate_report(self, topic: str) -> str:
        inputs = {"question": topic}

        for output in self.app.stream(inputs):
            for key, value in output.items():
                pprint(f"Finished running: {key}:")

        return value["generation"]