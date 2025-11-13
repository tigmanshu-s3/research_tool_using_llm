import os.path
import streamlit as st
from main import process_urls
from constant import VECTORSTORE_PICKLE_FILEPATH
from langchain.chains import RetrievalQAWithSourcesChain
from input_output import load_pickle_file
from langchain_openai import ChatOpenAI
from dotenv import dotenv_values

config = dotenv_values(".env")

st.title("Research Tool")

st.sidebar.title("Article Research URLs")

# --- Dynamic URL Input Management ---
if "urls" not in st.session_state:
    st.session_state.urls = [""]  # start with one empty field


def add_url():
    st.session_state.urls.append("")


def remove_url(index):
    if len(st.session_state.urls) > 1:  # keep at least one input
        st.session_state.urls.pop(index)


# Render dynamic inputs with remove button
for i, url in enumerate(st.session_state.urls):
    cols = st.sidebar.columns([4, 1])
    st.session_state.urls[i] = cols[0].text_input(f"URL {i+1}", url, key=f"url_{i}")
    if cols[1].button("➖", key=f"remove_{i}"):
        remove_url(i)

# Add button
st.sidebar.button("➕ Add another URL", on_click=add_url)

# Process button
processed_url_clicked = st.sidebar.button("Process URLs")
if processed_url_clicked:
    process_urls([u for u in st.session_state.urls if u.strip()])

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
