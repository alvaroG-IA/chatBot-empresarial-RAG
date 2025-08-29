# src/retriever.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = "../vectorstore"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def buscar_chunks(query, k=3):
    """Devuelve los chunks más relevantes con metadata"""
    results = db.similarity_search(query, k=k)
    return [{"texto": r.page_content, "fuente": r.metadata.get("source", "desconocida")} for r in results]


if __name__ == "__main__":
    consulta = input("Ingresa tu consulta: ")
    resultados = buscar_chunks(consulta)
    for r in resultados:
        print(r["texto"])
        print("Fuente:", r["fuente"])
        print("---")