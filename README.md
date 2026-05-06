# Agente Experto en la Industria Textil y Moda en España

Asistente de inteligencia artificial capaz de responder preguntas sobre la industria textil y moda en España utilizando su propia base de conocimiento vectorial. Aplica técnicas de **Retrieval-Augmented Generation (RAG)** con Google Gemini como LLM y embeddings, incluye memoria de conversación implementada con LangGraph, e incorpora una interfaz web interactiva en Streamlit.

---

## Índice

1. [Dominio elegido](#1-dominio-elegido)
2. [Stack tecnológico](#2-stack-tecnológico)
3. [Estructura del proyecto](#3-estructura-del-proyecto)
4. [Arquitectura del agente](#4-arquitectura-del-agente)
5. [System Prompt — diseño y justificación](#5-system-prompt--diseño-y-justificación)
6. [Decisiones técnicas de la pipeline RAG](#6-decisiones-técnicas-de-la-pipeline-rag)
7. [Memoria de conversación](#7-memoria-de-conversación)
8. [Interfaz Streamlit](#8-interfaz-streamlit)
9. [Instalación y ejecución](#9-instalación-y-ejecución)
10. [Requisitos y dependencias](#10-requisitos-y-dependencias)

---

## 1. Dominio elegido

**Industria textil y moda en España** — sector económico de gran relevancia que combina análisis económico, sostenibilidad, comercio exterior y estrategia empresarial. La elección responde a la disponibilidad de fuentes documentales heterogéneas y complementarias que permiten formular preguntas que requieren síntesis entre documentos, lo que ejercita al máximo el pipeline RAG.

### Base de conocimiento — 5 documentos PDF

| Documento | Contenido | Páginas aprox. |
|-----------|-----------|---------------|
| `informe-economico-de-la-moda-en-espana-2025.pdf` | Informe económico anual del sector moda en España (2025) | ~80 |
| `comercio_textil_2024.pdf` | Importaciones, exportaciones y balanza comercial textil (2024) | ~60 |
| `memoria_anual_inditex_2025.pdf` | Estrategia, sostenibilidad y resultados del Grupo Inditex (2025) | ~120 |
| `circularidad_textil.pdf` | Economía circular aplicada al textil: modelos, iniciativas y regulación europea | ~50 |
| `presentaciones_sectoriales_textil.pdf` | Estadísticas y evolución del tejido empresarial textil | ~40 |

> Total: más de 350 páginas indexadas, muy por encima del mínimo exigido de ~20 páginas.

---

## 2. Stack tecnológico

| Componente | Tecnología |
|------------|------------|
| LLM | Google Gemini 2.5 Flash Lite (`gemini-2.5-flash-lite`) |
| Embeddings | Google Gemini Embeddings (`models/text-embedding-004`) |
| Base de conocimiento vectorial | ChromaDB (persistido localmente en `chroma_db/`) |
| Framework de agente | LangGraph + LangChain |
| Memoria de conversación | `MemorySaver` con `thread_id` |
| Carga de documentos | PyPDFLoader (LangChain Community) |
| Interfaz notebook | Jupyter Notebook |
| Interfaz web (bonus) | Streamlit |

---

## 3. Estructura del proyecto

```
gemini_agent/
├── app.py                          # Interfaz web Streamlit
├── requirements.txt                # Dependencias Python
├── .env                            # API key (no subir al repo)
├── .gitignore
├── README.md
├── CLAUDE.md
├── docs/                           # Documentos fuente de la base de conocimiento
│   ├── informe-economico-de-la-moda-en-espana-2025.pdf
│   ├── comercio_textil_2024.pdf
│   ├── memoria_anual_inditex_2025.pdf
│   ├── circularidad_textil.pdf
│   └── presentaciones_sectoriales_textil.pdf
├── notebooks/
│   ├── agente_textil.ipynb         # Notebook principal — entregable
│   └── Untitled12.ipynb            # Experimentos de LangGraph (no entregable)
└── chroma_db/                      # Vector store persistido (generado al ejecutar el notebook)
```

---

## 4. Arquitectura del agente

El agente se construye como un grafo de estados en LangGraph con tres nodos secuenciales:

```
START → [nodo_reformular] → [nodo_recuperar] → [nodo_generar] → END
```

### Nodos

**`nodo_reformular`**
Antes de buscar en ChromaDB, el LLM reescribe la pregunta del usuario expandiéndola con sinónimos y términos específicos del sector textil. Esto mejora significativamente la recuperación semántica cuando el usuario formula preguntas en lenguaje coloquial.

```
Entrada: pregunta original del usuario
Salida:  query_reformulada (enriquecida con vocabulario sectorial)
```

**`nodo_recuperar`**
Busca los fragmentos más relevantes en ChromaDB usando la query reformulada. Utiliza **MMR (Maximal Marginal Relevance)** en lugar de similitud coseno pura, lo que equilibra relevancia con diversidad y evita recuperar 6 fragmentos del mismo párrafo.

```
Entrada: query_reformulada
Salida:  contexto (6 chunks formateados con fuente y página)
```

**`nodo_generar`**
Llama al LLM de Gemini con el system prompt, el contexto recuperado de ChromaDB y el **historial completo de la conversación**, lo que permite dar respuestas coherentes que hacen referencia a turnos anteriores.

```
Entrada: system prompt + contexto RAG + historial de mensajes
Salida:  respuesta del agente
```

### Estado del grafo

```python
class EstadoAgente(TypedDict):
    mensajes:          Annotated[list[BaseMessage], add_messages]  # historial acumulativo
    contexto:          str    # chunks recuperados de ChromaDB
    query_reformulada: str    # pregunta expandida para el retriever
```

---

## 5. System Prompt — diseño y justificación

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

### Justificación de las decisiones de diseño

| Decisión | Justificación |
|----------|---------------|
| **Rol de analista experto** | Orienta el tono hacia respuestas estructuradas y profesionales, adecuadas para un dominio técnico-económico. Evita respuestas demasiado informales o genéricas. |
| **"Basa tus respuestas en el contexto documental"** | Ancla el modelo a los documentos indexados y reduce alucinaciones sobre datos, cifras y estadísticas sectoriales que el modelo podría inventar de su preentrenamiento. |
| **Síntesis del historial permitida** | Permite respuestas de seguimiento coherentes ("compara eso con lo que explicaste antes") sin fabricar información nueva. La síntesis solo se permite si la fuente original era documental. |
| **"Indica claramente si no tienes información"** | Evita respuestas falsamente confiadas. El agente admite explícitamente los límites de su base de conocimiento, lo que es más útil y honesto que inventar una respuesta plausible. |
| **Citar fuentes cuando estén disponibles** | Permite trazabilidad directa a los documentos originales y refuerza la confianza en las respuestas, especialmente con datos numéricos. |
| **`temperature=0.3`** | Temperatura baja para respuestas consistentes y factuales. Un dominio de análisis económico-sectorial requiere reproducibilidad, no creatividad. |

---

## 6. Decisiones técnicas de la pipeline RAG

### Chunking

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""],
)
```

- **`chunk_size=1000`**: Tamaño suficiente para que cada fragmento contenga una idea completa. Con 500 caracteres (tamaño inicial probado) varios fragmentos quedaban cortados a mitad de una estadística o tabla, lo que degradaba la calidad de las respuestas.
- **`chunk_overlap=200`**: Solapa del 20% para evitar que conceptos que cruzan el límite de un chunk se pierdan al recuperar.
- **Separadores en orden de preferencia**: Se prioriza dividir por párrafo (`\n\n`), luego línea, luego frase. Solo se divide por espacio o carácter como último recurso.

### Embeddings — Gemini `text-embedding-004`

Se usan los embeddings nativos de Google Gemini en lugar de modelos locales de HuggingFace. Esto garantiza coherencia semántica entre los embeddings del índice y los embeddings de las queries en tiempo de consulta, al usar el mismo modelo en ambos momentos. El modelo `text-embedding-004` produce vectores de 768 dimensiones optimizados para recuperación de información.

### Retriever MMR

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
)
```

- **MMR**: Recupera candidatos por similitud (`fetch_k=20`) y luego selecciona los `k=6` que maximizan diversidad además de relevancia.
- **`lambda_mult=0.7`**: Peso 70% relevancia / 30% diversidad. Evita recuperar múltiples fragmentos del mismo párrafo del mismo documento.

---

## 7. Memoria de conversación

La memoria se implementa con `MemorySaver` de LangGraph, que persiste el estado del grafo entre invocaciones usando un `thread_id`:

```python
memoria = MemorySaver()
agente  = grafo.compile(checkpointer=memoria)

config  = {"configurable": {"thread_id": "sesion-01"}}
agente.invoke({"mensajes": [HumanMessage(content=pregunta)]}, config=config)
```

Cada llamada al agente con el mismo `thread_id` recibe el historial completo de mensajes anteriores. El campo `mensajes` del estado usa `add_messages` como reducer, que acumula los mensajes en lugar de sobreescribirlos.

**Demostración de memoria** (Ejemplo 4 del notebook):

```
Pregunta 2: "¿Cuál es la estrategia medioambiental de Inditex?"
Pregunta 3: "¿Qué es la economía circular aplicada al sector textil?"
Pregunta 4: "¿Y cómo se compara esa estrategia de Inditex con los principios
            de economía circular que acabas de explicar?"
```

El agente responde a la pregunta 4 relacionando correctamente los contenidos de los dos turnos anteriores sin necesidad de que el usuario los repita.

---

## 8. Interfaz Streamlit

La interfaz web (`app.py`) reutiliza exactamente el mismo grafo LangGraph del notebook. Se añade únicamente la capa de UI por encima.

### Características principales

**Panel de progreso en tiempo real**
Usa `agente.stream()` con `stream_mode="updates"` para recibir el output de cada nodo conforme se ejecuta y mostrarlo al usuario:

```
🔄 Paso 1/3 — Reformulando la pregunta con terminología sectorial…
   ↳ [query expandida]
📚 Paso 2/3 — Buscando en la base de conocimiento vectorial…
   ↳ 6 fragmentos recuperados de 2 documento(s)
✍️ Paso 3/3 — Generando respuesta con Gemini…
✅ Respuesta generada
```

**Transparencia del agente**
Bajo cada respuesta el usuario puede desplegar:
- 📄 **Fragmentos consultados** — qué archivos y páginas usó el agente
- 🔍 **Query reformulada** — cómo expandió el agente la pregunta original antes de buscar

**Controles de sesión**

| Botón | Acción |
|-------|--------|
| 🗑️ Nueva conversación | Limpia el historial y genera un nuevo `thread_id`. El agente permanece cargado en memoria (rápido). |
| ♻️ Recargar agente | Llama a `st.cache_resource.clear()` y reconstruye el agente desde cero. Útil si se regenera ChromaDB. |

**Rendimiento**: `@st.cache_resource` garantiza que el agente y ChromaDB se cargan una única vez por sesión de servidor, sin reconstruirse en cada rerun de Streamlit.

---

## 9. Instalación y ejecución

### Requisitos previos

- Python 3.10 o superior
- API key de Google Gemini — obtenible en [Google AI Studio](https://aistudio.google.com/app/apikey)

### Paso 1 — Instalación

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd gemini_agent

# Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 2 — Configurar la API key

Crear el fichero `.env` en la raíz del proyecto:

```
GEMINI_API_KEY=tu_api_key_aqui
```

> La API key **nunca** debe incluirse en el código ni subirse al repositorio. Está ignorada por `.gitignore`.

### Paso 3 — Ejecutar el notebook (obligatorio)

```bash
jupyter notebook notebooks/agente_textil.ipynb
```

Ejecutar todas las celdas en orden. La primera ejecución indexa los 5 documentos PDF en ChromaDB con Gemini Embeddings. El directorio `chroma_db/` se crea automáticamente.

> Tiempo estimado de primera indexación: 2-4 minutos en función de la velocidad de la API.

### Paso 4 — Interfaz Streamlit (opcional / bonus)

> Requiere haber completado el Paso 3 para que `chroma_db/` exista.

```bash
streamlit run app.py
```

La app estará disponible en `http://localhost:8501`.

---

## 10. Requisitos y dependencias

```
# LLM + Embeddings (Google Gemini)
langchain>=0.3.0
langchain-text-splitters>=0.3.0
langchain-google-genai>=2.0.0
langchain-community>=0.3.0
google-generativeai>=0.8.0

# Base de conocimiento vectorial
chromadb>=0.5.0

# Framework de agente
langgraph>=0.2.0

# Procesamiento de PDFs
pypdf>=4.0.0

# Utilidades
python-dotenv>=1.0.0
ipywidgets>=8.0.0
notebook>=7.0.0

# Interfaz web (bonus)
streamlit>=1.35.0
```
