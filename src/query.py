# query.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def buscar(query, k=3):
    """
    Realiza una búsqueda semántica en la base FAISS.
    
    Args:
        query (str): Texto de consulta.
        k (int): Número de chunks más relevantes a devolver.
    
    Returns:
        List of dicts: Cada dict contiene 'texto' y 'fuente'.
    """
    results = db.similarity_search(query, k=k)
    respuesta = []
    for r in results:
        respuesta.append({
            "texto": r.page_content,
            "fuente": r.metadata.get("source", "desconocida")
        })
    return respuesta

# ----------------------------
# Configuración
# ----------------------------
DB_FAISS_PATH = "vectorstore"  # Carpeta donde guardaste la base FAISS

# ----------------------------
# Inicializar embeddings (misma configuración que en ingest.py)
# ----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Cargar la base FAISS
# ----------------------------
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# ----------------------------
# Ejemplo de uso
# ----------------------------
if __name__ == "__main__":
    consulta = input("Ingresa tu consulta: ")
    resultados = buscar(consulta, k=1)
    
    print("\n🔹 Resultados:")
    for idx, r in enumerate(resultados, 1):
        print(f"{idx}. Texto: {r['texto']}")
        print(f"   Fuente: {r['fuente']}")
        print("---")
