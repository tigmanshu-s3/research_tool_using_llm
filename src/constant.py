import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_PICKLE_FILEPATH = os.path.join(BASE_DIR, "data", "vectorstore.pkl")
