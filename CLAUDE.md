# CLAUDE.md

## Project Overview

Agente experto en la industria textil y moda en España. Implementa RAG con HuggingFace Embeddings (E5), ChromaDB y LangGraph.

## Structure

- `notebooks/agente_textil.ipynb` — notebook principal con el agente completo
- `app.py` — interfaz Streamlit del agente
- `docs/` — 4 PDFs fuente de la base de conocimiento
- `chroma_db/` — vector store persistido (se regenera al ejecutar el notebook)
- `.env` — API key de Gemini (no commitear); variable: `GOOGLE_API_KEY`
- `pyproject.toml` — dependencias Python (gestor: uv)

## Tech Stack

- LLM: `langchain-google-genai` (Gemini 2.5 Flash Lite) — usa `GOOGLE_API_KEY`
- Embeddings: `intfloat/multilingual-e5-base` via `langchain-community` (HuggingFace, local, sin cuotas)
- Vector store: ChromaDB via `langchain-chroma`
- Agent framework: LangGraph con MemorySaver
- PDF loading: PyPDFLoader
- UI: Streamlit

## Running

```bash
uv sync
# Añadir GOOGLE_API_KEY al .env
jupyter notebook notebooks/agente_textil.ipynb
# O para la app web:
uv run streamlit run app.py
```

## Notes

- Los embeddings con `gemini-embedding-001` fueron descartados: el free tier (100 req/min) generaba errores 429 irrecuperables durante la indexación de 681 chunks.
- El modelo E5 requiere prefijo `"passage: "` al indexar y `"query: "` al buscar — implementado en la clase `E5MultilingualEmbeddings`.
- `circularidad_textil.pdf` fue eliminado intencionadamente de la base de conocimiento.
