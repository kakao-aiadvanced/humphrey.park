from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema.document import Document
import bs4
from dotenv import load_dotenv
import os
import faiss

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
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
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

# embedding: https://python.langchain.com/v0.2/docs/integrations/text_embedding/openai/
doc_embeddings = embeddings.embed_documents(texts_content)

# FAISS 인덱스 생성
dimension = len(doc_embeddings[0])  # 임베딩 벡터의 차원
index = faiss.IndexFlatL2(dimension)  # L2 거리 기반의 FAISS 인덱스 생성

# 문서 저장소 생성 및 벡터 저장소 초기화
documents = [Document(page_content=text) for text in texts_content]
docstore = InMemoryDocstore(dict(enumerate(documents)))
index_to_docstore_id = {i: i for i in range(len(documents))}

vectorstore = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=embeddings)

# 임베딩을 FAISS 인덱스에 추가
vectorstore.add_texts(texts_content)

# 4. User query = ‘agent memory’ 를 받아 관련된 chunks를 retrieve
# https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.VectorStore.html#langchain_core.vectorstores.VectorStore.as_retriever
query = "agent memory"
results = vectorstore.similarity_search(query, 5)

# 5. User query와 retrieved chunk 에 대해 relevance 가 있는지를 평가하는 시스템 프롬프트 작성:
# retrieval 퀄리티를 LLM 이 스스로 평가하도록 하고,
#   관련이 있으면 {‘relevance’: ‘yes’}
#   관련이 없으면 {‘relevance’: ‘no’} 라고 출력하도록 함.
# ( JsonOutputParser() 를 활용 ) - llama3 prompt format 준수
# https://python.langchain.com/v0.2/docs/how_to/output_parser_json/#without-pydantic
llm = ChatOllama(model="llama3:8b", temperature=0)

class RelevanceResults(BaseModel):
    relevance: str = Field(description="Whether the retrieved document is relevant to the query. If relevant, set to 'yes'; otherwise, set to 'no'.")

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=RelevanceResults)

prompt = PromptTemplate(
    template="You are an assistant that provides answers in JSON format.\n{format_instructions}\nQuery: {query}\n\nJSON Response:",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

# 6. 5 에서 모든 docs에 대해 ‘no’ 라면 디버깅
# (Splitter, Chunk size, overlap, embedding model, vector store, retrieval 평가 시스템 프롬프트 등)
query_results = [chain.invoke({"query": result.page_content}) for result in results]
assert all(result["relevance"] == "yes" for result in query_results)

# 7. 5에서 ‘yes’ 라면 질문과 명확히 관련 없는 docs 나 질문 (예: ‘I like an apple’)에 대해서는 ‘no’ 라고 나오는지 테스트 프롬프트 및 평가 코드 작성.
# 이 때는 관련 없다는 답변 작성
# - llama3 prompt format 준수
invalid_results = vectorstore.similarity_search("I like an apple", 5)
print(invalid_results)
assert all(result["relevance"] == "no" for result in invalid_results)

# 8. ‘yes’ 이고 7의 평가에서도 문제가 없다면, 4의 retrieved chunk 를 가지고 답변 작성
# prompt | llm | parser 형태로 작성

# 9. 생성된 답안에 Hallucination 이 있는지 평가하는 시스템 프롬프트 작성.
# LLM이 스스로 평가하도록 하고, hallucination 이
#   있으면 {‘hallucination’: ‘yes’}
#   없으면 {‘hallucination’: ‘no’}
# 라고 출력하도록 함
# - llama3 prompt format 준수

# 10. 9 에서 ‘yes’ 면 8 로 돌아가서 다시 생성, ‘no’ 면 답변 생성하고 유저에게 답변 생성에 사용된 출처와 함께 출력