from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

query = "What are high-risk AI systems?"

results = vectordb.similarity_search(query, k=2)

for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(doc.page_content)