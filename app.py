from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# ============================================
# 1. LOAD QWEN LLM
# ============================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading LLM...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=False,
    return_full_text=False
)

llm = HuggingFacePipeline(
    pipeline=text_generator
)

print("LLM loaded successfully")


# ============================================
# 2. LOAD EMBEDDING MODEL
# ============================================

print("Loading embeddings...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings loaded successfully")


# ============================================
# 3. LOAD CHROMA DATABASE
# ============================================

print("Loading ChromaDB...")

vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

print("ChromaDB loaded successfully")


# ============================================
# 4. CREATE RETRIEVER
# ============================================

retriever = vectordb.as_retriever(
    search_kwargs={"k": 3}
)

print("Retriever created successfully")


# ============================================
# 5. CREATE PROMPT
# ============================================

prompt = PromptTemplate.from_template(
    """You are a helpful assistant answering questions about the EU AI Act.

Use ONLY the information provided in the context.

Do not make up information.

If the answer is not available in the context, say:

"I cannot find the answer in the provided EU AI Act."

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================
# 6. CREATE RAG CHAIN
# ============================================

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={
        "prompt": prompt
    },
    return_source_documents=True
)

print("RAG chain created successfully")


# ============================================
# 7. ASK QUESTION
# ============================================

question = "What are high-risk AI systems?"

print("\n==============================")
print("Question:")
print(question)
print("==============================")


result = qa_chain.invoke({
    "query": question
})


# ============================================
# 8. DISPLAY ANSWER
# ============================================

print("\nAnswer:")
print(result["result"])


# ============================================
# 9. DISPLAY SOURCES
# ============================================

print("\nSources:")

for i, doc in enumerate(result["source_documents"], 1):
    print(f"\nSource {i}:")
    print(doc.page_content[:500])