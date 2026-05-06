# CLAUDE.md

## Project Overview

Agente experto en la industria textil y moda en España. Implementa RAG con Google Gemini Embeddings, ChromaDB y LangGraph.

## Structure

- `notebooks/agente_textil.ipynb` — notebook principal con el agente completo
- `notebooks/Untitled12.ipynb` — experimentos de aprendizaje de LangGraph (no es entregable)
- `docs/` — 5 PDFs fuente de la base de conocimiento
- `chroma_db/` — vector store persistido (se regenera al ejecutar el notebook)
- `.env` — API key de Gemini (no commitear)
- `requirements.txt` — dependencias Python

## Tech Stack

- LLM + Embeddings: `langchain-google-genai` (Gemini 2.5 Flash Lite + text-embedding-004)
- Vector store: ChromaDB via `langchain-community`
- Agent framework: LangGraph con MemorySaver
- PDF loading: PyPDFLoader

## Running

```bash
pip install -r requirements.txt
# Añadir GEMINI_API_KEY al .env
jupyter notebook notebooks/agente_textil.ipynb
```
