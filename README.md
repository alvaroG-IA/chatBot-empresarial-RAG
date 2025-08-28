# chatBot-empresarial-RAG
🤖 Chatbot RAG Empresarial

Un asistente de Recursos Humanos basado en RAG (Retrieval-Augmented Generation) que responde preguntas sobre documentos internos de la empresa (manuales, políticas, FAQs).

Construido con LangChain, OpenAI, FAISS y Streamlit.

✨ Características

Búsqueda semántica en documentos internos (PDF, TXT, etc.).

Generación de respuestas con LLM (GPT u otro).

Recuperación de contexto preciso mediante FAISS.

Interfaz web sencilla con Streamlit.

Arquitectura modular, fácil de ampliar a otros departamentos o casos de uso.

📂 Estructura del proyecto
```
chatbot-rag-empresarial/
│
├── data/            # Documentos internos de la empresa
├── src/             # Código principal
│   ├── ingest.py    # Ingesta de documentos y creación de la base vectorial
│   ├── retriever.py # Recuperación desde FAISS
│   ├── chatbot.py   # Lógica del chatbot RAG
│   └── app.py       # Interfaz web (Streamlit)
├── vectorstore/     # Persistencia local de FAISS
├── requirements.txt # Dependencias
├── .env.example     # Variables de entorno
└── README.md        # Documentación
```

🚀 Instalación

Clona el repositorio:

```
git clone https://github.com/usuario/chatbot-rag-empresarial.git
cd chatbot-rag-empresarial
```


Instala dependencias:
```
pip install -r requirements.txt
```

Configura tu API key en .env:
```
cp .env.example .env
```

Edita .env y añade:

`OPENAI_API_KEY=tu_api_key_aqui`

▶️ Uso

Ingestar documentos y crear la base vectorial:

`python src/ingest.py`


Ejecutar el chatbot en interfaz web:

`streamlit run src/app.py`


Abre el navegador en:
👉 http://localhost:8501

📊 Ejemplo de interacción

Pregunta:
"¿Cuántos días de vacaciones tiene un empleado al año?"

Respuesta del chatbot:
"Según la política de RRHH, cada empleado tiene derecho a 23 días laborables de vacaciones al año."

🔮 Posibles mejoras

- Integrar autenticación para usuarios internos.

- Añadir soporte a más formatos de documentos (Word, Excel).

- Reemplazar Streamlit por FastAPI para integraciones en Slack o MS Teams.

- Usar vectores híbridos (BM25 + embeddings).

📜 Licencia

- Este proyecto está disponible bajo licencia MIT.

🙌 Créditos

--- Proyecto desarrollado por Álvaro García Velasco ---
