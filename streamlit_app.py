import streamlit as st
import os

from dotenv import load_dotenv
from groq import Groq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================
# LOAD ENVIRONMENT VARIABLES
# ============================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Groq API key not found. Check your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="EU AI Act Q&A",
    page_icon="📚",
    layout="centered"
)

st.title("📚 EU AI Act Q&A Assistant")

st.write(
    "Ask questions about the EU AI Act using a "
    "Retrieval-Augmented Generation (RAG) system."
)

st.divider()


# ============================================
# LOAD CHROMADB
# ============================================

@st.cache_resource
def load_database():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )

    return vectordb


# ============================================
# QUESTION BOX
# ============================================

question = st.text_input(
    "Ask a question:",
    placeholder="What are high-risk AI systems?"
)


# ============================================
# ASK BUTTON
# ============================================

if st.button("🔍 Ask Question"):

    if not question:
        st.warning("Please enter a question.")

    else:

        # ----------------------------------------
        # Retrieve documents
        # ----------------------------------------

        with st.spinner("Searching the EU AI Act..."):

            vectordb = load_database()

            documents = vectordb.similarity_search(
                question,
                k=3
            )

        # Combine retrieved text
        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )


        # ----------------------------------------
        # Create RAG prompt
        # ----------------------------------------

        prompt = f"""
You are an assistant answering questions about the EU AI Act.

Use ONLY the information provided in the context below.

Do not invent information.
Do not use outside knowledge.

If the answer is not present in the context, say:

"I cannot find the answer in the provided EU AI Act."

Give a clear and concise answer.

Context:
{context}

Question:
{question}

Answer:
"""


        # ----------------------------------------
        # Generate answer using Groq
        # ----------------------------------------

        with st.spinner("Generating answer..."):

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            answer = response.choices[0].message.content


        # ========================================
        # DISPLAY ANSWER
        # ========================================

        st.subheader("💡 Answer")

        st.write(answer)


        # ========================================
        # DISPLAY SOURCES
        # ========================================

        st.subheader("📄 Sources")

        for i, doc in enumerate(documents, 1):

            with st.expander(f"Source {i}"):

                st.write(doc.page_content)