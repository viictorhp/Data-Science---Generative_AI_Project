# Agente Experto en la Industria Textil y Moda en España

Asistente de inteligencia artificial que responde preguntas sobre la industria textil y moda en España a partir de una base de conocimiento documental propia. Aplica **Retrieval-Augmented Generation (RAG)** con embeddings locales HuggingFace E5, ChromaDB como vector store, Google Gemini como LLM y LangGraph para orquestar el flujo con memoria de conversación. Incluye interfaz web interactiva en Streamlit.

---

## Dominio elegido

**Industria textil y moda en España** — sector con gran riqueza documental que permite ejercitar el pipeline RAG con consultas que requieren síntesis entre fuentes heterogéneas:

- 📊 **Análisis económico**: evolución del número de empresas, facturación y empleo del sector
- 🌍 **Comercio exterior**: balanza comercial, principales mercados de exportación e importación
- 🏢 **Estrategia empresarial**: resultados financieros, estructura y política de sostenibilidad de Inditex
- 📋 **Posición europea**: comparativa del sector textil español dentro de la UE
- 💡 **Tendencias sectoriales**: digitalización, ecommerce y evolución de los subsectores

### Base de conocimiento — 4 documentos PDF (322 páginas)

| Documento | Contenido |
|-----------|-----------|
| `informe-economico-de-la-moda-en-espana-2025.pdf` | Informe económico anual del sector moda en España |
| `comercio_textil_2024.pdf` | Importaciones, exportaciones y balanza comercial textil |
| `memoria_anual_inditex_2025.pdf` | Estrategia, sostenibilidad y resultados del Grupo Inditex |
| `presentaciones_sectoriales_textil.pdf` | Estadísticas y evolución del tejido empresarial textil |

---

## Stack tecnológico

| Componente | Tecnología |
|------------|------------|
| **LLM** | Google Gemini (`gemini-2.0-flash`) via `langchain-google-genai` |
| **Embeddings** | `intfloat/multilingual-e5-base` (HuggingFace, local, sin cuotas) |
| **Vector store** | ChromaDB via `langchain-chroma` |
| **Agent framework** | LangGraph + `MemorySaver` |
| **PDF loading** | `PyPDFLoader` (`langchain-community`) |
| **Interfaz web** | Streamlit |

> **Por qué E5 y no Gemini Embeddings**: se intentó usar `gemini-embedding-001` pero el free tier (100 req/min) generaba errores 429 irrecuperables al indexar 681 chunks. `multilingual-e5-base` corre en local, no tiene cuotas y está entrenado específicamente para tareas de retrieval en múltiples idiomas incluido el español.

---

## Guía de ejecución

### 1. Requisitos previos

- Python 3.13+
- API key de Google Gemini

### 2. Instalación

```bash
git clone <url-del-repo>
cd gemini_agent
```

### 3. Configurar la API key

Crear el fichero `.env` en la raíz del proyecto:

```
GOOGLE_API_KEY=tu_clave_aqui
```

### 4. Generar la base de conocimiento vectorial

Ejecutar el notebook completo en orden

Esto indexa los 4 PDFs en ChromaDB con embeddings E5. Solo es necesario ejecutarlo una vez (o cuando cambien los documentos).

### 5. Lanzar la interfaz web (Streamlit Cloud)

[Streamlit Cloud - Agente Textil](https://geminiagent-z9tcmwt4mrmgarz4svyrjk.streamlit.app/)

## Arquitectura del grafo

El agente se implementa como un grafo de estados en LangGraph con tres nodos secuenciales:

```
START → [nodo_reformular] → [nodo_recuperar] → [nodo_generar] → END
```

### Nodos

**`nodo_reformular`**
El LLM reescribe la pregunta del usuario expandiéndola con sinónimos y vocabulario específico del sector textil antes de buscar en ChromaDB. Mejora significativamente la recuperación semántica cuando el usuario formula preguntas en lenguaje coloquial.

```
Entrada → pregunta original del usuario
Salida  → query_reformulada (enriquecida con vocabulario sectorial)
```

**`nodo_recuperar`**
Busca los fragmentos más relevantes en ChromaDB usando la query reformulada. Utiliza **MMR (Maximal Marginal Relevance)** con `fetch_k=20`, `k=6` y `lambda_mult=0.7` (70% relevancia / 30% diversidad), lo que evita recuperar múltiples fragmentos del mismo párrafo.

```
Entrada → query_reformulada
Salida  → contexto (6 chunks formateados con fuente y página)
```

**`nodo_generar`**
Llama al LLM con el system prompt, el contexto recuperado y el **historial completo de la conversación**, lo que permite dar respuestas coherentes que hacen referencia a turnos anteriores.

```
Entrada → system prompt + contexto RAG + historial de mensajes
Salida  → respuesta del agente
```

### Memoria

Implementada con `MemorySaver` de LangGraph. El campo `mensajes` del estado usa `add_messages` como reducer, acumulando el historial entre turnos mediante `thread_id`. Cada sesión en la app Streamlit tiene su propio `thread_id` generado con `uuid4`.

---

## Justificación del System Prompt

```
Eres un analista experto en la industria textil y moda en España.

Tu conocimiento proviene de los documentos que se te proporcionan como contexto en cada turno.
Respondes siempre en español, con un tono profesional y conciso.

Reglas:
- Basa tus respuestas en el contexto documental proporcionado.
- Puedes sintetizar y relacionar información de tus respuestas anteriores en la conversación,
  siempre que las afirmaciones originales provengan del contexto documental.
- Si el contexto no contiene información suficiente y no puedes inferirlo del historial, indícalo claramente.
- No inventes datos, cifras ni estadísticas que no aparezcan en el contexto o en el historial previo.
- Cuando cites datos numéricos, menciona la fuente si está disponible en el contexto.
```

| Decisión | Justificación |
|----------|---------------|
| **Rol de analista experto** | Orienta el tono hacia respuestas estructuradas y profesionales, adecuadas para un dominio técnico-económico. |
| **"Basa tus respuestas en el contexto documental"** | Ancla el modelo a los documentos indexados y reduce alucinaciones sobre datos y estadísticas sectoriales. |
| **Síntesis del historial permitida** | Permite respuestas de seguimiento coherentes sin fabricar información nueva. La síntesis solo se permite si la fuente original era documental. |
| **"Indica claramente si no tienes información"** | El agente admite explícitamente los límites de su base de conocimiento en lugar de inventar respuestas plausibles. |
| **Citar fuentes cuando estén disponibles** | Permite trazabilidad directa a los documentos originales, especialmente útil con datos numéricos. |
| **`temperature=0.3`** | Temperatura baja para respuestas consistentes y factuales. Un dominio económico-sectorial requiere reproducibilidad, no creatividad. |

---

## Dependencias

Gestionadas en `requirements.txt`:

```
requires-python = ">=3.13"
dependencies = [
    "chromadb",               # Vector store
    "google-generativeai",    # Listado de modelos Gemini disponibles
    "ipywidgets",             # Widgets para Jupyter
    "langchain",              # Framework base LangChain
    "langchain-chroma",       # Integración ChromaDB (sin deprecation warning)
    "langchain-community",    # PyPDFLoader, HuggingFaceEmbeddings
    "langchain-google-genai", # ChatGoogleGenerativeAI (LLM)
    "langchain-text-splitters",
    "langgraph",              # Orquestación del agente con grafo de estados
    "notebook",               # Jupyter Notebook
    "pypdf",                  # Carga de PDFs
    "python-dotenv",          # Carga de variables de entorno desde .env
    "sentence-transformers",  # Backend de HuggingFaceEmbeddings
    "streamlit",              # Interfaz web
]
```
