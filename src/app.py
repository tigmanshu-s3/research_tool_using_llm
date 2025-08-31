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
        # Layout
        st.markdown("## 📌 Results")

        with st.container():
            st.markdown("### ❓ Question")
            st.info(query, icon="💬")

        with st.container():
            st.markdown("### ✅ Answer")
            st.success(result["answer"], icon="🤖")

        sources = result.get("sources", "")
        if sources:
            with st.container():
                st.markdown("### 📚 Sources")
                for idx, source in enumerate(sources.split("\n"), start=1):
                    if source.strip():
                        st.write(f"**{idx}.** {source}")
