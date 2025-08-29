import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------------------
# Configuración
# ----------------------------
DATA_DIR = "data"          # Carpeta con todos los documentos .txt
DB_FAISS_PATH = "vectorstore"  # Carpeta donde se guardará FAISS
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ----------------------------
# Inicializar lista de chunks y metadata
# ----------------------------
texts = []
metadatas = []

# ----------------------------
# Procesar todos los documentos
# ----------------------------
for filename in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, filename)
    
    if os.path.isfile(file_path) and filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Dividir en chunks
        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_text(text)
        
        # Guardar chunks y metadata
        texts.extend(chunks)
        metadatas.extend([{"source": filename} for _ in chunks])

# ----------------------------
# Crear embeddings
# ----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Crear vectorstore FAISS con metadata
# ----------------------------
db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
db.save_local(DB_FAISS_PATH)

print("✅ Base vectorial creada en", DB_FAISS_PATH)
print(f"📄 Documentos procesados: {len(texts)} chunks")
