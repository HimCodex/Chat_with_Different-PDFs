# 📄 Chat with Different PDFs🤖

A powerful AI application that allows you to **upload multiple PDF documents and ask questions from them** using **Google Gemini + RAG (Retrieval-Augmented Generation)**.

---

## 🚀 Features

* 📂 Upload multiple PDFs
* 🔍 Extract and process text automatically
* ✂️ Smart text chunking for better context
* 🧠 Local embeddings using HuggingFace (no API cost)
* ⚡ Fast similarity search using FAISS
* 🤖 Accurate answers using Google Gemini
* 🎯 Context-based responses (no hallucination)

---

## 🧠 How It Works

```
PDFs → Text Extraction → Chunking → Embeddings → FAISS (Vector DB)
                                               ↓
User Question → Similarity Search → Context → Gemini → Answer
```

This project uses **RAG (Retrieval-Augmented Generation)** to provide accurate answers from your documents.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Google Gemini
* **Embeddings:** HuggingFace (Sentence Transformers)
* **Vector DB:** FAISS
* **PDF Processing:** PyPDF2
* **Framework:** LangChain (latest modular structure)

---

## 📦 Installation

```bash
git clone https://github.com/HimCodex/Chat_with_Different-PDFs.git
activate virtual environments: (#in windows)
python -m venv .venv
.venv\Scripts\activate   

install all packages at once:
pip install -r requirements.txt
```

---

## 🔑 Setup API Key

Create a `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ Run the App from terminal

```git bash
streamlit run app.py
```

---

## 📌 Usage

1. Upload one or more PDF files
2. Click **Submit & Process**
3. Ask questions from the documents
4. Get accurate answers instantly

---

## ⚡ Important Notes

* Uses **local embeddings** → no quota issues
* Gemini is used only for answering
* First run may take time (embedding creation)
* FAISS index is saved locally for faster reuse

---
