import os.path
from langchain_community.llms import HuggingFacePipeline
import streamlit as st
from main import process_urls
from constant import VECTORSTORE_PICKLE_FILEPATH
from langchain.chains import RetrievalQAWithSourcesChain
from transformers import pipeline
from input_output import load_pickle_file
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values

config = dotenv_values(".env")

st.title("Research Tool")

st.sidebar.title("Article Research URLs")

urls = []

for i in range(1):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

processed_url_clicked = st.sidebar.button("Process URL")

if processed_url_clicked:
    process_urls(urls)

query = st.text_input("Question")

# qa_pipeline = pipeline(
#     "text-generation", model="google/flan-t5-large", max_new_tokens=512, truncation=True
# )
# llm = HuggingFacePipeline(pipeline=qa_pipeline)
llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3.1:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=config["OPEN_ROUTER_API_KEY"],
)

if query:
    if os.path.exists(VECTORSTORE_PICKLE_FILEPATH):
        vectorstore = load_pickle_file()
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        result = chain({"question": query}, return_only_outputs=True)
        st.title("Answer")
        st.subheader(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources")
            sources_list = sources.split("\n")

            for source in sources_list:
                st.write(source)
