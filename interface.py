import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
import tempfile

load_dotenv()

st.set_page_config(page_title="Mi Cerebro AI", page_icon="🧠")
st.title("🧠 Mi Asistente de Documentos Personal")

# --- Lógica de Procesamiento ---
def process_pdf(uploaded_file):
    # Guardar temporalmente el archivo subido
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OpenAIEmbeddings()
    )
    return vector_db

# --- Interfaz de Usuario ---
uploaded_file = st.file_uploader("Sube tu PDF aquí", type="pdf")

if uploaded_file:
    # Usamos session_state para no procesar el PDF en cada click
    if "vector_db" not in st.session_state:
        with st.spinner("Analizando documento..."):
            st.session_state.vector_db = process_pdf(uploaded_file)
            st.success("Documento listo!")

    # Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada del usuario
    if prompt := st.chat_input("¿Qué quieres saber del documento?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Respuesta de la IA
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_db.as_retriever()
        )
        
        with st.chat_message("assistant"):
            response = qa_chain.invoke(prompt)["result"]
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})