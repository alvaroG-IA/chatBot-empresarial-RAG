# src/app.py
import streamlit as st
from chatbot import obtener_respuesta

st.title("💼 Chatbot RRHH - RAG (Ligero)")

st.markdown("Pregunta sobre tus documentos de RRHH y obtén respuestas oficiales con fuente.")

user_input = st.text_input("Pregunta:")

if user_input:
    with st.spinner("Generando respuesta..."):
        respuesta, fuentes = obtener_respuesta(user_input, k=1)

    st.markdown("**Respuesta oficial:**")
    st.write(respuesta)

    st.markdown("**Fuentes utilizadas:**")
    for f in fuentes:
        st.write(f"- *{f['fuente']}*: {f['chunk']}")
