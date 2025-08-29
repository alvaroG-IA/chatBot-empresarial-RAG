# src/app.py

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# ----------------------------
# Configuración
# ----------------------------
DB_FAISS_PATH = "vectorstore"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Modelo LLM gratuito (elige uno ligero si tu RAM es limitada)
LLM_MODEL = "tiiuae/falcon-7b-instruct"  # Requiere RAM alta
MAX_LENGTH = 512
TEMPERATURE = 0.7

# ----------------------------
# Cargar embeddings y vectorstore
# ----------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# ----------------------------
# Configurar modelo LLM local gratuito
# ----------------------------
generator = pipeline(
    "text-generation",
    model=LLM_MODEL,
    max_length=MAX_LENGTH,
    temperature=TEMPERATURE
)
llm = HuggingFacePipeline(pipeline=generator)

# ----------------------------
# Crear cadena RAG
# ----------------------------
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(), return_source_documents=True)

# ----------------------------
# Interfaz con Streamlit
# ----------------------------
st.title("💼 Chatbot RRHH - RAG Optimizado")
st.markdown("Haz preguntas sobre los documentos de RRHH. El chatbot responderá usando el contexto de los documentos.")

user_input = st.text_input("Pregunta:")

if user_input:
    # Ejecutar búsqueda RAG
    result = qa({"query": user_input})
    respuesta = result['result']
    fuentes = result['source_documents']

    # Mostrar respuesta oficial
    st.markdown("**Respuesta del chatbot:**")
    st.write(respuesta)

    # Mostrar fuentes
    st.markdown("**Fuentes utilizadas:**")
    for doc in fuentes:
        fuente = doc.metadata.get("source", "desconocida")
        st.write(f"- {fuente}: {doc.page_content[:200]}...")  # Muestra los primeros 200 caracteres
