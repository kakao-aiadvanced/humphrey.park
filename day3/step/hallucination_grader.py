from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3", format="json", temperature=0)

# Prompt
prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
    an answer IS NOT GROUNDED in / supported by a set of facts. \nGive a binary 'yes' or 'no' to indicate
    whether the answer IS NOT GROUNDED in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'hallucination' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents}
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"]
)

hallucination_grader = prompt | llm | JsonOutputParser()