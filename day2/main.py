import bs4
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load .env
load_dotenv()

# 1. 3개의 블로그 포스팅 본문을 Load
# -----------------------------------
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content")
        )
    ),
)

docs = loader.load()

# 불러온 본문을 하나의 텍스트로 통합
combined_docs = "\n".join(doc.page_content for doc in docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# 2. 불러온 본문을 Split (Chunking) : recursive text splitter 활용
# https://python.langchain.com/v0.2/docs/how_to/recursive_text_splitter/
texts = text_splitter.create_documents([combined_docs])

# 3. Chunks 를 임베딩하여 Vector store 저장: openai 사용
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# texts 객체의 형식을 확인하고, 필요한 경우 변환
texts_content = [text.page_content for text in texts]

# FAISS 를 사용하여 Vector store 저장
vectorstore = FAISS.from_texts(texts=texts_content, embedding=embeddings)

# Vector store 를 retriever 로 변환
retriever = vectorstore.as_retriever()

# 4. User query = ‘agent memory’ 를 받아 관련된 chunks를 retrieve
# https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.VectorStore.html#langchain_core.vectorstores.VectorStore.as_retriever
query = "agent memory"
results_from_vectorstore = retriever.invoke(query)

# 5. User query와 retrieved chunk 에 대해 relevance 가 있는지를 평가하는 시스템 프롬프트 작성:
# retrieval 퀄리티를 LLM 이 스스로 평가하도록 하고,
#   관련이 있으면 {‘relevance’: ‘yes’}
#   관련이 없으면 {‘relevance’: ‘no’} 라고 출력하도록 함.
# ( JsonOutputParser() 를 활용 ) - llama3 prompt format 준수
# https://python.langchain.com/v0.2/docs/how_to/output_parser_json/#without-pydantic
llm = ChatOllama(model="llama3:8b", temperature=0)


class RelevanceResults(BaseModel):
    relevance: str = Field(
        description="Whether the document is relevant to the sentence. "
                    "If relevant, set to 'yes'; otherwise, set to 'no'.")


relevance_result_parser = JsonOutputParser(pydantic_object=RelevanceResults)

relevance_inquiry_prompt = PromptTemplate(
    template=
    "Your task is to determine the similarity between the document and the sentence. \n"
    "Consider both semantic meaning and contextual relevance in your evaluation.\n\n"
    "You are an assistant that provides answers in JSON format.\n"
    "{format_instructions}\n\n"
    "### sentence: \n"
    "---\n"
    "{query}\n"
    "---\n\n"
    "### document: \n"
    "---\n"
    "{docs}\n"
    "---\n\n"
    "### Instructions:\n"
    "1. Analyze the content of the document and the sentence.\n"
    "2. Determine the semantic similarity and contextual relevance between the document and the sentence.\n"
    "3. Answer \"yes\" if the sentence is meaningfully related to the document, otherwise answer \"no\".\n"
    "\n\nJSON Response:",

    input_variables=["query", 'docs'],
    partial_variables={"format_instructions": relevance_result_parser.get_format_instructions()},
)

relevance_check_chain = relevance_inquiry_prompt | llm | relevance_result_parser

# 6. 5 에서 모든 docs에 대해 ‘no’ 라면 디버깅
# (Splitter, Chunk size, overlap, embedding model, vector store, retrieval 평가 시스템 프롬프트 등)
query_results = [relevance_check_chain.invoke({"query": query, "docs": result.page_content}) for result in
                 results_from_vectorstore]
print(query_results)
assert all(result["relevance"] == "yes" for result in query_results)

# 7. 5에서 ‘yes’ 라면 질문과 명확히 관련 없는 docs 나 질문 (예: ‘I like an apple’)에 대해서는 ‘no’ 라고 나오는지 테스트 프롬프트 및 평가 코드 작성.
# 이 때는 관련 없다는 답변 작성
# - llama3 prompt format 준수
invalid_query = "I like an apple."
invalid_results_from_vectorstore = retriever.invoke(invalid_query)

# for result in invalid_results_from_vectorstore:
#     print(result.page_content + '\n------\n')

invalid_query_results = [relevance_check_chain.invoke({"query": invalid_query, "docs": result.page_content}) for result
                         in invalid_results_from_vectorstore]
print(invalid_query_results)
assert all(result["relevance"] == "no" for result in invalid_query_results)


# 8. ‘yes’ 이고 7의 평가에서도 문제가 없다면, 4의 retrieved chunk 를 가지고 답변 작성
# prompt | llm | parser 형태로 작성

# 9. 생성된 답안에 Hallucination 이 있는지 평가하는 시스템 프롬프트 작성.
# LLM이 스스로 평가하도록 하고, hallucination 이
#   있으면 {‘hallucination’: ‘yes’}
#   없으면 {‘hallucination’: ‘no’}
# 라고 출력하도록 함
# - llama3 prompt format 준수
class HallucinationResults(BaseModel):
    hallucination: str = Field(
        description="Whether the document is relevant to the sentence. "
                    "If relevant, set to 'yes'; otherwise, set to 'no'.")


hallucination_json_parser = JsonOutputParser(pydantic_object=HallucinationResults)

hallucination_check_prompt = PromptTemplate(
    template=
    "You are given a response that evaluates the similarity between a document and a sentence. \n"
    "Your task is to determine whether the explanation provided contains any hallucinations. \n"
    "A hallucination occurs when the response is 'yes' even if the sentence cannot be inferred from the document.\n"
    "You are an assistant that provides answers in JSON format.\n"
    "{format_instructions}\n\n"
    "### sentence: \n"
    "---\n"
    "{query}\n"
    "---\n\n"
    "### document: \n"
    "---\n"
    "{docs}\n"
    "---\n\n"
    "### response: \n"
    "---\n"
    "Answer: {relevance}\n"
    "---\n\n"
    "### Instructions:\n"
    "1. Analyze the content of the document, the sentence, and the explanation.\n"
    '2. Answer "yes" if there are hallucinations, otherwise answer "no".\n'
    "\n\nJSON Response:",
    input_variables=["query", 'docs', 'relevance'],
    partial_variables={"format_instructions": hallucination_json_parser.get_format_instructions()},
)

invalid_results_with_relevance_results = list(zip(invalid_results_from_vectorstore, invalid_query_results))

# 10. 9 에서 ‘yes’ 면 8 로 돌아가서 다시 생성, ‘no’ 면 답변 생성하고 유저에게 답변 생성에 사용된 출처와 함께 출력
hallucination_check_chain = hallucination_check_prompt | llm | hallucination_json_parser
hallucination_check_results = [hallucination_check_chain.invoke(
    {"query": invalid_query, "docs": result[0].page_content, "relevance": result[1]["relevance"]}) for result in
    invalid_results_with_relevance_results]

print(hallucination_check_results)
assert all(result["hallucination"] == "no" for result in hallucination_check_results)

for result in invalid_results_with_relevance_results:
    print(f'Query: {invalid_query}')
    print(f'Relevance results: \n----\n{result[0].page_content}\n----\n')
