import streamlit as st
from main import process_urls

st.title("Research Tool")

st.sidebar("Article Research URLs")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")

processed_url_clicked = st.sidebar.button("Process URL")

if processed_url_clicked:
    process_urls(urls)
