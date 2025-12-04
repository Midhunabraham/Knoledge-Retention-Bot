# Knowledge Retention & Training Bot

Streamlit + ChromaDB (Local Vector DB) + Ollama (Local LLM)

A lightweight, private, and offline-capable AI assistant designed for
organizational knowledge retention, document Q&A, and automated
summarization. The tool processes PDFs/Text documents, stores embeddings
locally using ChromaDB, and uses Ollama to generate accurate responses
without sending data outside the system --- ensuring 100% data privacy.

## 🚀 Features

-   Fully local -- No cloud usage, all processing done on your machine\
-   Document ingestion -- Upload PDFs, text files, or manuals\
-   Automatic chunking & summarization\
-   Local Vector Search using ChromaDB\
-   Ollama-based LLM responses\
-   Fast interactive UI with Streamlit\
-   Secure -- Ideal for confidential organizational data\
-   Embeddings automatically created and stored for reuse

## 🧱 Architecture Overview

     ┌────────┐      ┌───────────┐      ┌──────────┐      ┌──────────────┐
     │  PDF / │      │ Text       │      │ ChromaDB │      │   Ollama      │
     │  Files │ ---> │ Chunk/Sum  │ ---> │ Embedding│ ---> │ Local LLM     │
     └────────┘      └───────────┘      └──────────┘      └──────────────┘
                          │                     ↑
                          └───────── Query ─────┘

## 📦 Tech Stack

  Component   Description
  ----------- ------------------
  Streamlit   UI/Frontend
  Ollama      Local LLM engine
  ChromaDB    Vector database
  Python      Core logic
  PyPDF       PDF parsing

## 🛠️ Installation

### 1. Install Python dependencies

    pip install -r requirements.txt

### 2. Install Ollama

Download from: https://ollama.com/download\
Check version:

    ollama --version

### 3. Pull a model

    ollama pull llama3

## ▶️ Run the App

    streamlit run app.py

## 📁 Project Structure

    project/
    │── app.py
    │── requirements.txt
    │── vector_store/
    │── utils/
    │── README.md

## 📝 How It Works

1.  Upload a PDF or text file\
2.  App extracts text → splits into chunks\
3.  Long chunks auto‑summarized\
4.  Embeddings generated using ChromaDB\
5.  User query → semantic search → Ollama model response

## 🔒 Data Privacy

-   No cloud\
-   All data local\
-   Safe for confidential documents

## 💡 Use Cases

-   Internal knowledge base\
-   Training assistant\
-   Documentation chatbot\
-   Offline secure Q&A system

## 🐞 Troubleshooting

  Issue                   Fix
  ----------------------- ------------------------
  chromadb not found      `pip install chromadb`
  Ollama not responding   Restart Ollama
  PDF too large           Split PDF

## 🤝 Contributing

Open to pull requests and suggestions.

## 📜 License

MIT License

# Core Web App
streamlit==1.50.0

# Vector DB
chromadb==1.3.4

# Embedding utilities (Ollama-compatible)
sentence-transformers==5.1.1

# PDF Reading
pypdf==6.1.0

# HTTP Requests
requests==2.32.5

# Misc Utilities
numpy==2.3.3

# Optional: FAISS (if using)
faiss-cpu==1.12.0

