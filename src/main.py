from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

from input_output import save_pickle_file

load_dotenv()


def load_text(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    return data


def split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","], chunk_size=200
    )
    docs = text_splitter.split_documents(data)

    return docs


def create_and_save_embeddings_to_db(docs):
    embeddings = HuggingFaceEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    return vectorstore_openai


def process_urls(urls):
    data = load_text(urls)
    docs = split_text(data)
    vectorstore = create_and_save_embeddings_to_db(docs)
    save_pickle_file(vectorstore)


# if __name__ == "__main__":
#     process_urls(
#         [
#             "https://www.geeksforgeeks.org/devops/how-to-use-docker-for-machine-learning/",
#             "https://www.renpy.org/doc/html/quickstart.html",
#         ]
#     )
