# 🧠 Research Assistant using LLMs
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-3E8EDE)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-009688)](https://faiss.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)

> 🚀 A lightweight **AI Research Tool** that fetches content from web URLs, stores it in a local vector database, and answers questions contextually using **Retrieval-Augmented Generation (RAG)** with LLMs like **DeepSeek**, **Flan-T5**, or **Mistral**.

🧠 How It Works

1. 📰 Enter URLs in the Streamlit sidebar

2. ⚙️ Click Process URL → the tool fetches text, splits it, embeds it, and stores it locally using FAISS

3. 💬 Ask a question in the input box

4. 🤖 The LLM retrieves the most relevant chunks and generates an answer with references

---

## ✨ Features

- 🌐 Extracts and processes text directly from article URLs  
- 🧩 Creates vector embeddings with **HuggingFace** and **FAISS**  
- 🤖 Answers your queries using **LLMs via OpenRouter**  
- 📚 Provides accurate **answers with source references**  
- 🪄 Interactive **Streamlit UI** with dynamic URL add/remove  

---

## 🧱 Tech Stack

| Layer | Tool / Library |
|-------|----------------|
| **Frontend / UI** | Streamlit |
| **LLM Provider** | OpenRouter (DeepSeek, Mistral, Flan-T5) |
| **Framework** | LangChain |
| **Embeddings** | HuggingFaceEmbeddings |
| **Vector Store** | FAISS |
| **Data Loading** | UnstructuredURLLoader |

---

## 🗂️ Project Structure

src/
├── app.py # Streamlit app (UI + QA interface)
├── main.py # Core logic: load, split, embed, and store
├── input_output.py # Handles saving/loading vectorstore as pickle
├── constant.py # Constants (paths, configs)
├── .env # API keys (e.g., OPEN_ROUTER_API_KEY)

---

## ⚙️ Installation & Setup

```bash
# 1️⃣ Clone the repository
git clone https://github.com/yourusername/research-tool-llm.git
cd research-tool-llm

# 2️⃣ Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # (Windows: .venv\Scripts\activate)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Add your API key
echo "OPEN_ROUTER_API_KEY=your_api_key_here" > .env

▶️ Run the App
streamlit run src/app.py
Then open your browser at http://localhost:8501
 🎉