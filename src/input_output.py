import pickle
from constant import VECTORSTORE_PICKLE_FILEPATH


def save_pickle_file(vectorstore):
    with open(VECTORSTORE_PICKLE_FILEPATH, "wb") as f:
        pickle.dump(vectorstore, f)


def load_pickle_file():
    with open(VECTORSTORE_PICKLE_FILEPATH, "rb") as f:
        vectorstore = pickle.load(f)

        return vectorstore
