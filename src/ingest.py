import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "../data"
DB_FAISS_PATH = "../vectorstore"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

texts = []
metadatas = []

# Procesar documentos
for filename in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, filename)
    if os.path.isfile(path) and filename.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_text(text)
        texts.extend(chunks)
        metadatas.extend([{"source": filename} for _ in chunks])

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
db.save_local(DB_FAISS_PATH)

print("✅ Base vectorial creada en", DB_FAISS_PATH)