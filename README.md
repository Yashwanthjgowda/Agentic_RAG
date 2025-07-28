# 🤖 Agentic RAG Chatbot (Gemini + LangChain + FAISS)

An interactive chatbot that combines **Google Gemini LLM**, **LangChain agents**, and **FAISS vector search** to answer questions from uploaded PDF documents.

![Streamlit Screenshot](./screenshot.png)

## 🔍 Features

- 📄 Upload and process any PDF file
- 🔗 Uses LangChain + FAISS for chunked document similarity search
- 🧠 Gemini LLM answers your questions and decides when to search the document
- 🛠️ Tool-using Agent (Conversational ReAct Agent) via LangChain
- 🗂️ Uses HuggingFace `all-MiniLM-L6-v2` for dense embeddings
- 💬 Fully interactive chat UI using Streamlit

---

## ⚙️ How It Works

1. User uploads a PDF document.
2. PDF is split into chunks and embedded using HuggingFace transformers.
3. FAISS is used to store and search similar chunks.
4. A LangChain agent with Gemini LLM determines whether to search or respond directly.
5. Conversations are preserved using session memory.

---


