# create_vectorstore.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and split PDF
loader = PyPDFLoader("txtbk.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(pages)

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/sentence-t5-large")

# Create and save FAISS
db = FAISS.from_documents(docs, embedding_model)
db.save_local("faiss_index", index_name="index")

print("✅ FAISS index created: faiss_index/")
