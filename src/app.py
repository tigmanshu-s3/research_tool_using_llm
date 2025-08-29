import os.path
import pickle
from langchain import OpenAI
import streamlit as st
from main import process_urls
from constant import VECTORSTORE_PICKLE_FILEPATH
from langchain.chains import RetrievalQAWithSourcesChain

from input_output import load_pickle_file

st.title("Research Tool")

st.sidebar.title("Article Research URLs")

urls = []

for i in range(1):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

processed_url_clicked = st.sidebar.button("Process URL")

if processed_url_clicked:
    process_urls(urls)
    # pass

query = st.text_input("Question")

llm = OpenAI(temperature=0.9, max_tokens=500)
if query:
    if os.path.exists(VECTORSTORE_PICKLE_FILEPATH):
        vectorstore = load_pickle_file()
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever()
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
