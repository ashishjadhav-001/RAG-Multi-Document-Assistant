# 📚 RAG Multi-Document Assistant

A powerful **Retrieval-Augmented Generation (RAG)** application that allows users to upload multiple PDF documents and ask questions. The system retrieves relevant context from documents and generates accurate answers using LLMs.

---

## 🚀 Features

* 📄 Upload **multiple PDFs**
* 🔍 Semantic search across all documents
* 🤖 AI-powered question answering
* 📚 Answers generated strictly from document context
* 📌 Displays **source document name**
* 🧹 Clear chat functionality
* ⚡ Fast retrieval using vector database (Chroma)

---

## 🧠 How It Works

1. User uploads one or more PDF files
2. PDFs are split into chunks
3. Chunks are converted into embeddings
4. Stored in a vector database (Chroma)
5. User asks a question
6. Relevant chunks are retrieved
7. LLM generates answer using context

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **LLM:** Mistral AI (`mistral-small`)
* **Embeddings:** Mistral Embeddings
* **Vector DB:** ChromaDB
* **Framework:** LangChain

---

## 📂 Project Structure

```
RAG-Multi-Document-Assistant/
│
├── app.py
├── rag_pipeline.py
├── vector_store.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd RAG-Multi-Document-Assistant
```

---

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add Environment Variables

Create a `.env` file in root directory:

```
MISTRAL_API_KEY=your_api_key_here
```

---

### 5. Run the App

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This app can be deployed on:

* Streamlit Cloud (recommended)
* Render
* Docker

> Note: File storage is temporary in cloud environments.

---

## 🧪 Example Use Cases

* 📖 Study assistant for books & notes
* 📊 Research document analysis
* 🧾 Company document Q&A system
* 📚 Knowledge base chatbot

---

## 🧹 Important Notes

* Do NOT push `.env` file
* Vector database is generated dynamically
* Temporary files are not stored permanently

---

## 📌 Future Improvements

* 🔍 Highlight answer in document
* 📄 Show page numbers
* 🎯 Filter by specific PDF
* 🌐 API backend integration
* 🧠 Better reranking for accuracy

---

## 💼 Resume Highlight

> Built a multi-document RAG (Retrieval-Augmented Generation) system using LangChain and Mistral AI, enabling semantic search and AI-powered question answering across PDFs with source attribution.

---

## 👨‍💻 Author

Ashish Jadhav

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
