"""
Brain AI — RAG Document Assistant with Enhanced UI

A document Q&A system that grounds LLM responses in source documents,
mitigating hallucinations through retrieval-augmented generation (RAG).

Features:
- Streaming responses
- Page-level source citations
- Persistent vector storage
- Content-hash caching to avoid re-embedding
- Input validation and error handling
- Modern, professional UI
"""

import os
import tempfile
import hashlib

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ============================================================================
# CONFIG
# ============================================================================
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "100"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))


# ============================================================================
# PAGE SETUP
# ============================================================================
st.set_page_config(
    page_title="Asistente de Textos",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* ---- Fonts: DM Sans for body, Lora (serif) for the title ---- */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Lora:wght@500;600&display=swap');

    /* ---- Base typography ---- */
    html, body, [class*="css"], .stMarkdown, .stText, .stChatMessage {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #24313a;
    }

    /* ---- Page ---- */
    .stApp {
        background: linear-gradient(180deg, #f6faf9 0%, #ffffff 320px);
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 820px;
    }

    /* ---- Hide Streamlit branding ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}

    /* ---- Header ---- */
    .app-header {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        padding: 0 0 1.4rem 0;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid #dce5e8;
    }
    .app-header .logo {
        width: 52px;
        height: 52px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        background: linear-gradient(135deg, #2f6f6a 0%, #4e9d94 100%);
        box-shadow: 0 4px 12px rgba(47, 111, 106, 0.28);
    }
    .app-header .logo svg {
        width: 26px;
        height: 26px;
        stroke: #ffffff;
        fill: none;
        stroke-width: 1.6;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .app-header h1 {
        margin: 0;
        color: #173b43;
        font-family: 'Lora', Georgia, serif;
        font-size: 1.75rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    .app-header p {
        margin: 0.2rem 0 0 0;
        color: #718087;
        font-size: 0.92rem;
    }
    .tech-badges {
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin-bottom: 1.4rem;
    }
    .tech-badges span {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        color: #34736e;
        background: #eaf4f2;
        border: 1px solid #d3e6e2;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
    }

    /* ---- Active document chip ---- */
    .doc-chip {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        background: #eaf4f2;
        border: 1px solid #cfe3df;
        border-radius: 10px;
        padding: 0.55rem 0.9rem;
        margin: 0.4rem 0 1rem 0;
        font-size: 0.88rem;
    }
    .doc-chip svg {
        width: 15px;
        height: 15px;
        stroke: #34736e;
        fill: none;
        stroke-width: 1.7;
        stroke-linecap: round;
        stroke-linejoin: round;
        flex-shrink: 0;
    }
    .doc-chip .doc-name {
        font-weight: 600;
        color: #24434a;
    }
    .doc-chip .doc-meta {
        color: #5f7d79;
        font-size: 0.8rem;
        margin-left: auto;
    }

    /* ---- File uploader ---- */
    [data-testid="stFileUploader"] {
        border: 1.5px dashed #a9bfbd;
        border-radius: 12px;
        padding: 0.6rem;
        background: #fbfcfc;
        transition: border-color 0.15s ease, background 0.15s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #4e8f8a;
        background: #f4f9f8;
    }

    /* ---- Chat messages ---- */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e0e8e7;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(23, 59, 67, 0.04);
    }
    /* User messages get a subtle tint to stand apart from the assistant */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #f0f7f6;
        border-color: #d7e7e4;
    }

    /* ---- Chat input ---- */
    [data-testid="stChatInput"] {
        border-radius: 12px;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        border: 1px solid #d5e1df;
    }
    .stButton > button:hover {
        border-color: #4e8f8a;
        color: #34736e;
    }

    /* ---- Source citations card ---- */
    .source-card {
        background: #f7f9f8;
        border: 1px solid #e0e8e7;
        border-left: 3px solid #6ba19c;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .source-card .page-badge {
        display: inline-block;
        background: #34736e;
        color: white;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .source-card .preview {
        color: #52525b;
        font-style: italic;
        line-height: 1.5;
    }

    /* ---- Empty state: how-it-works steps ---- */
    .steps {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin: 1.6rem 0;
    }
    .step-card {
        background: #ffffff;
        border: 1px solid #e0e8e7;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(23, 59, 67, 0.05);
    }
    .step-card .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2f6f6a, #4e9d94);
        color: white;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }
    .step-card h4 {
        margin: 0 0 0.3rem 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: #24434a;
    }
    .step-card p {
        margin: 0;
        font-size: 0.82rem;
        color: #718087;
        line-height: 1.45;
    }
    @media (max-width: 640px) {
        .steps { grid-template-columns: 1fr; }
    }

    /* ---- Footer ---- */
    .app-footer {
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #e6edec;
        text-align: center;
        font-size: 0.78rem;
        color: #9aa8ab;
    }

    /* ---- Success/info alerts ---- */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: none;
    }

    /* ---- Expander ---- */
    .streamlit-expanderHeader {
        border-radius: 12px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def file_fingerprint(uploaded_file) -> str:
    """Stable hash of file contents — used to cache vector indexes."""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()[:16]


def format_docs_for_context(docs) -> str:
    """Flatten retrieved chunks into a single context string with page markers."""
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in docs
    )


def process_pdf(uploaded_file):
    """Load, chunk, embed, and persist a PDF. Uses content-hash caching."""
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File is {file_size_mb:.1f} MB. Max is {MAX_FILE_SIZE_MB} MB.")

    collection_name = f"doc_{file_fingerprint(uploaded_file)}"
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Cache check — if we've indexed this exact file before, reuse it
    try:
        existing = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
        count = existing._collection.count()
        if count > 0:
            return existing, None, count
    except Exception:
        pass

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        if len(docs) > MAX_PAGES:
            raise ValueError(f"PDF has {len(docs)} pages. Max is {MAX_PAGES}.")

        for doc in docs:
            doc.metadata["source_file"] = uploaded_file.name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)

        vector_db = Chroma.from_documents(
            documents=chunks, embedding=embeddings,
            collection_name=collection_name, persist_directory=PERSIST_DIR,
        )
        return vector_db, len(docs), len(chunks)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ============================================================================
# RAG CHAIN
# ============================================================================
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions about a document. "
        "Use ONLY the provided context to answer. If the answer isn't in the "
        "context, say you don't know — do not invent information. "
        "Answer in the same language the user used. "
        "When citing facts, reference the relevant page numbers.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}"),
])


def build_rag_chain(vector_db):
    """Build the streaming RAG chain."""
    retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, streaming=True)

    chain = (
        {"context": retriever | format_docs_for_context, "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )
    return chain, retriever


def render_sources(sources: list[dict]) -> None:
    """Render source citations as beautiful cards."""
    with st.expander(f"Ver {len(sources)} fragmentos citados"):
        for src in sources:
            st.markdown(
                f"""
                <div class="source-card">
                    <span class="page-badge">Page {src['page']}</span>
                    <span class="preview">{src['preview']}{'…' if src['truncated'] else ''}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="app-header">
    <div class="logo">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7z"/>
            <path d="M14 2v5h5"/>
            <line x1="9" y1="12" x2="15" y2="12"/>
            <line x1="9" y1="16" x2="13" y2="16"/>
        </svg>
    </div>
    <div>
        <h1>Asistente de Textos</h1>
        <p>Subí un PDF y hacé preguntas sobre su contenido, con citas de página.</p>
    </div>
</div>
<div class="tech-badges">
    <span>Streamlit</span>
    <span>LangChain</span>
    <span>ChromaDB</span>
    <span>OpenAI</span>
    <span>RAG</span>
</div>
""", unsafe_allow_html=True)

# Fail fast if API key missing
if not os.getenv("OPENAI_API_KEY"):
    st.error("No se encontró `OPENAI_API_KEY`. Agregala a tu archivo `.env` y reiniciá la app.")
    st.stop()


# ============================================================================
# MAIN CONTENT
# ============================================================================
uploaded_file = st.file_uploader(
    "Elegí un documento PDF",
    type="pdf",
    help=f"Hasta {MAX_FILE_SIZE_MB} MB y {MAX_PAGES} páginas.",
)

if uploaded_file:
    is_new_file = (
        "current_file" not in st.session_state
        or st.session_state.current_file != uploaded_file.name
    )

    if is_new_file:
        with st.spinner(f"Preparando {uploaded_file.name}..."):
            try:
                vector_db, n_pages, n_chunks = process_pdf(uploaded_file)
                st.session_state.vector_db = vector_db
                st.session_state.current_file = uploaded_file.name
                st.session_state.n_chunks = n_chunks
                st.session_state.messages = []

                if n_pages is None:
                    st.success(f"Documento listo. **{n_chunks} fragmentos** preparados.")
                else:
                    st.success(
                        f"Listo: **{n_pages} páginas** procesadas. Preguntá lo que quieras."
                    )
            except ValueError as e:
                st.error(f"{e}")
                st.stop()
            except Exception as e:
                st.error(f"Something went wrong.\n\n`{type(e).__name__}: {e}`")
                st.stop()


# ============================================================================
# CHAT
# ============================================================================
if "vector_db" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown(f"""
    <div class="doc-chip">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7z"/>
            <path d="M14 2v5h5"/>
        </svg>
        <span class="doc-name">{st.session_state.current_file}</span>
        <span class="doc-meta">{st.session_state.n_chunks} fragmentos indexados</span>
    </div>
    """, unsafe_allow_html=True)

    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    # New user message
    if prompt := st.chat_input("Escribí tu pregunta sobre el documento..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                chain, retriever = build_rag_chain(st.session_state.vector_db)

                retrieved_docs = retriever.invoke(prompt)
                sources = [
                    {
                        "page": doc.metadata.get("page", 0) + 1,
                        "preview": doc.page_content[:220].replace("\n", " "),
                        "truncated": len(doc.page_content) > 220,
                    }
                    for doc in retrieved_docs
                ]

                response_placeholder = st.empty()
                full_response = ""
                for token in chain.stream(prompt):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

                render_sources(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                })

            except Exception as e:
                st.error(f"Couldn't generate response.\n\n`{type(e).__name__}: {e}`")

else:
    # Empty state — shown when no PDF uploaded
    st.markdown("""
    <div class="steps">
        <div class="step-card">
            <span class="step-num">1</span>
            <h4>Subí tu PDF</h4>
            <p>Arrastrá o elegí un documento de hasta 10 MB.</p>
        </div>
        <div class="step-card">
            <span class="step-num">2</span>
            <h4>Se indexa el contenido</h4>
            <p>El texto se divide en fragmentos y se convierte en vectores para buscarlo por significado.</p>
        </div>
        <div class="step-card">
            <span class="step-num">3</span>
            <h4>Preguntá lo que quieras</h4>
            <p>Las respuestas se basan solo en el documento, con cita de página.</p>
        </div>
    </div>
    <div class="app-footer">
        Asistente de Textos · Proyecto personal · RAG con LangChain + ChromaDB
    </div>
    """, unsafe_allow_html=True)
