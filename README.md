# 📚 EU AI Act RAG-Based Document Q&A System

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about the **EU AI Act** and receive answers based on the content of the document.

The project combines document processing, semantic search, vector databases, and an LLM to provide grounded answers with retrieved source information.

---

## 🚀 Features

- 📄 Load and process the EU AI Act PDF
- ✂️ Split the document into smaller chunks
- 🧠 Generate semantic embeddings using Sentence Transformers
- 🔎 Perform similarity-based document retrieval
- 🗄️ Store and search embeddings using ChromaDB
- 🤖 Generate answers using Groq LLM
- 💬 Interactive Streamlit web interface
- 📑 Display retrieved source passages
- 🔐 Secure API-key handling using `.env`

---

## 🏗️ RAG Architecture

```text
                    User Question
                          │
                          ▼
                  Streamlit Interface
                          │
                          ▼
                    ChromaDB Search
                          │
                          ▼
              Retrieve Relevant Chunks
                          │
                          ▼
                 Context + Question
                          │
                          ▼
                    Groq LLM
                          │
                          ▼
                    Final Answer
                          │
                          ▼
                  Retrieved Sources