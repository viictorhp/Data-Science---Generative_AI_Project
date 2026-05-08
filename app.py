import os
import re
import uuid
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver



# ── Configuración ─────────────────────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "textil_moda_espana"
LLM_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

SYSTEM_PROMPT = """Eres un analista experto en la industria textil y moda en España.

Tu conocimiento proviene de los documentos que se te proporcionan como contexto en cada turno.
Respondes siempre en español, con un tono profesional y conciso.

Reglas:
- Basa tus respuestas en el contexto documental proporcionado.
- Puedes sintetizar y relacionar información de tus respuestas anteriores en la conversación,
  siempre que las afirmaciones originales provengan del contexto documental.
- Si el contexto no contiene información suficiente y no puedes inferirlo del historial, indícalo claramente.
- No inventes datos, cifras ni estadísticas que no aparezcan en el contexto o en el historial previo.
- Cuando cites datos numéricos, menciona la fuente si está disponible en el contexto.
"""

REFORMULATION_PROMPT = (
    "Eres un asistente especializado en documentos sobre la industria textil y moda en España. "
    "Tu tarea es reformular la siguiente pregunta expandiéndola con sinónimos y términos relacionados "
    "del sector, para mejorar la recuperación de información en una base de conocimiento documental. "
    "Devuelve ÚNICAMENTE la query reformulada en una sola línea, sin explicaciones ni puntuación extra."
)

DOCS_INFO = [
    ("📊", "Informe económico de la moda 2025"),
    ("🌍", "Comercio textil 2024"),
    ("🏢", "Memoria anual Inditex 2025"),
    ("📋", "Presentaciones sectoriales"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_sources(contexto: str) -> list[dict]:
    """Extrae y deduplica las fuentes del bloque de contexto formateado."""
    sources, seen = [], set()
    for m in re.finditer(r'\[Fuente: (.+?) \| p\.(.+?)\]', contexto):
        key = (m.group(1), m.group(2))
        if key not in seen:
            seen.add(key)
            sources.append({"archivo": m.group(1), "pagina": m.group(2)})
    return sources


def reset_conversation():
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())


# ── Agente (cacheado — se construye una sola vez por sesión de servidor) ───────
@st.cache_resource
def load_agent():
    # Lazy import: torch y sentence-transformers solo se cargan cuando el usuario
    # envía su primer mensaje, no al arrancar Streamlit (~500 MB de diferencia).
    from langchain_community.embeddings import HuggingFaceEmbeddings

    class E5MultilingualEmbeddings(HuggingFaceEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return super().embed_documents(["passage: " + t for t in texts])
        def embed_query(self, text: str) -> list[float]:
            return super().embed_query("query: " + text)

    embeddings = E5MultilingualEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
    )
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
    )

    class EstadoAgente(TypedDict):
        mensajes: Annotated[list[BaseMessage], add_messages]
        contexto: str
        query_reformulada: str

    def nodo_reformular(estado: EstadoAgente) -> dict:
        ultima = estado["mensajes"][-1].content
        resp = llm.invoke([
            SystemMessage(content=REFORMULATION_PROMPT),
            HumanMessage(content=ultima),
        ])
        c = resp.content
        query = ("".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c) if isinstance(c, list) else c).strip()
        return {"query_reformulada": query}

    def nodo_recuperar(estado: EstadoAgente) -> dict:
        docs = retriever.invoke(estado["query_reformulada"])
        contexto = "\n\n".join([
            f"[Fuente: {Path(d.metadata.get('source', '?')).name} | p.{d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        ])
        return {"contexto": contexto}

    def nodo_generar(estado: EstadoAgente) -> dict:
        prompt_ctx = SYSTEM_PROMPT + f"\n\nCONTEXTO RELEVANTE DE LOS DOCUMENTOS:\n{estado['contexto']}"
        mensajes = [SystemMessage(content=prompt_ctx)] + estado["mensajes"]
        resp = llm.invoke(mensajes)
        return {"mensajes": [resp]}

    grafo = StateGraph(EstadoAgente)
    grafo.add_node("reformular", nodo_reformular)
    grafo.add_node("recuperar", nodo_recuperar)
    grafo.add_node("generar", nodo_generar)
    grafo.add_edge(START, "reformular")
    grafo.add_edge("reformular", "recuperar")
    grafo.add_edge("recuperar", "generar")
    grafo.add_edge("generar", END)

    return grafo.compile(checkpointer=MemorySaver())


# ── Layout ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agente Textil España",
    page_icon="🧵",
    layout="wide",
)

# Inicializar estado de sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧵 Agente Textil España")
    st.caption("Asistente experto basado en documentos sectoriales reales.")
    st.divider()

    n_preguntas = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.metric("Preguntas en esta sesión", n_preguntas)
    st.metric("ID de sesión", st.session_state.thread_id[:8] + "…")
    st.divider()

    st.markdown("**📁 Base de conocimiento**")
    for icon, nombre in DOCS_INFO:
        st.markdown(f"{icon} {nombre}")
    st.divider()

    st.markdown("**⚙️ Stack tecnológico**")
    st.caption(
        "- LLM: Ggemini 2.5 Flash Lite\n"
        "- Embeddings: intfloat/multilingual-e5-base\n"
        "- Vector store: ChromaDB (MMR)\n"
        "- Agent: LangGraph + MemorySaver"
    )
    st.divider()

    st.markdown("**🎮 Controles**")
    if st.button("♻️ Nueva conversación", use_container_width=True, type="primary",
                 help="Limpia el historial e inicia una conversación nueva"):
        st.cache_resource.clear()
        reset_conversation()
        st.rerun()

# ── Área principal ────────────────────────────────────────────────────────────
st.title("Agente Experto en la Industria Textil y Moda en España")
st.markdown(
    "Pregunta sobre **empresas**, **resultados financieros**, **comercio exterior**, "
    "**sostenibilidad** o **estrategia sectorial**. El agente mantiene el contexto "
    "de la conversación entre preguntas."
)

# Verificar que ChromaDB existe
if not Path(CHROMA_DIR).exists():
    st.warning(
        "**Base de conocimiento no encontrada.** "
        "Ejecuta primero el notebook `agente_textil.ipynb` para indexar los documentos en ChromaDB.",
        icon="⚠️",
    )
    st.stop()

st.divider()

# ── Historial de mensajes ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            col1, col2 = st.columns(2)
            sources = msg.get("sources", [])
            query_ref = msg.get("query_reformulada", "")

            if sources:
                with col1:
                    with st.expander(f"📄 {len(sources)} fragmentos consultados"):
                        for s in sources:
                            st.markdown(f"- **{s['archivo']}** — p. {s['pagina']}")

            if query_ref:
                with col2:
                    with st.expander("🔍 Query reformulada por el agente"):
                        st.code(query_ref, language=None)

# ── Input y procesamiento ─────────────────────────────────────────────────────
if prompt := st.chat_input("Haz una pregunta sobre la industria textil en España..."):
    # Mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Respuesta del agente
    with st.chat_message("assistant"):
        query_reformulada = ""
        sources = []
        respuesta = ""

        try:
            agente = load_agent()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            # Panel de progreso en tiempo real
            with st.status("Procesando consulta…", expanded=True) as status:
                st.write("🔄 **Paso 1/3 — Reformulando** la pregunta con terminología sectorial…")

                for chunk in agente.stream(
                    {"mensajes": [HumanMessage(content=prompt)]},
                    config=config,
                    stream_mode="updates",
                ):
                    if "reformular" in chunk:
                        query_reformulada = chunk["reformular"].get("query_reformulada", "")
                        st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ *{query_reformulada}*")
                        st.write("📚 **Paso 2/3 — Buscando** en la base de conocimiento vectorial…")

                    elif "recuperar" in chunk:
                        ctx = chunk["recuperar"].get("contexto", "")
                        sources = parse_sources(ctx)
                        docs_unicos = len(set(s["archivo"] for s in sources))
                        st.write(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;↳ **{len(sources)} fragmentos** recuperados "
                            f"de {docs_unicos} documento(s)"
                        )
                        st.write("✍️ **Paso 3/3 — Generando** respuesta con Gemini…")

                    elif "generar" in chunk:
                        msgs = chunk["generar"].get("mensajes", [])
                        if msgs:
                            c = msgs[-1].content
                            respuesta = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c) if isinstance(c, list) else c

                status.update(label="✅ Respuesta generada", state="complete", expanded=False)

            # Respuesta
            st.markdown(respuesta)

            # Fuentes y query reformulada (en columnas bajo la respuesta)
            if sources or query_reformulada:
                col1, col2 = st.columns(2)
                if sources:
                    with col1:
                        with st.expander(f"📄 {len(sources)} fragmentos consultados"):
                            for s in sources:
                                st.markdown(f"- **{s['archivo']}** — p. {s['pagina']}")
                if query_reformulada:
                    with col2:
                        with st.expander("🔍 Query reformulada por el agente"):
                            st.code(query_reformulada, language=None)

        except Exception as e:
            st.error(f"Error al consultar el agente: {e}")
            respuesta = f"[Error: {e}]"

        # Guardar en historial
        st.session_state.messages.append({
            "role": "assistant",
            "content": respuesta,
            "sources": sources,
            "query_reformulada": query_reformulada,
        })
