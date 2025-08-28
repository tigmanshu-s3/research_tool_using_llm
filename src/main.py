import pickle
from langchain import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv

load_dotenv()


def load_text(urls):
    loader = UnstructuredURLLoader(urls=urls)


def process_urls(urls):
    pass
