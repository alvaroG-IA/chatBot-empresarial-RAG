# src/chatbot.py
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from retriever import db  # tu base FAISS ya cargada

# ----------------------------
# Configurar LLM ligero
# ----------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",    # modelo pequeño, rápido, CPU-friendly
    max_new_tokens=150,     # tokens a generar (no usa max_length)
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=generator)

# ----------------------------
# Crear cadena RAG
# ----------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=db.as_retriever(),
    return_source_documents=True
)

# ----------------------------
# Función para obtener respuesta
# ----------------------------
def obtener_respuesta(query, k=2):
    """
    Devuelve respuesta oficial + fuentes
    """
    result = qa({"query": query})
    respuesta = result['result']
    fuentes = [
        {
            "fuente": doc.metadata.get("source", "desconocida"),
            "chunk": doc.page_content[:200] + "..."
        } 
        for doc in result['source_documents']
    ]
    return respuesta, fuentes

# ----------------------------
# Prueba rápida
# ----------------------------
if __name__ == "__main__":
    consulta = input("Pregunta sobre RRHH: ")
    respuesta, fuentes = obtener_respuesta(consulta, k=1)
    print("Respuesta oficial:", respuesta)
    print("Fuentes:")
    for f in fuentes:
        print(f"- {f['fuente']}: {f['chunk']}")
