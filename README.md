# Agente Experto en la Industria Textil y Moda en España

Asistente de inteligencia artificial capaz de responder preguntas sobre la industria textil y moda en España utilizando su propia base de conocimiento vectorial. Aplica técnicas de **Retrieval-Augmented Generation (RAG)** con Google Gemini como LLM y embeddings, incluye memoria de conversación implementada con LangGraph, e incorpora una interfaz web interactiva en Streamlit.

---

## Dominio elegido

**Industria textil y moda en España** — sector económico de gran relevancia que combina análisis económico, sostenibilidad, comercio exterior y estrategia empresarial. La elección responde a la disponibilidad de fuentes documentales heterogéneas y complementarias que permiten formular preguntas que requieren síntesis entre documentos, lo que ejercita al máximo el pipeline RAG.

### Base de conocimiento — 5 documentos PDF

| Documento | Contenido | Páginas aprox. |
|-----------|-----------|---------------|
| `informe-economico-de-la-moda-en-espana-2025.pdf` | Informe económico anual del sector moda en España (2025) | ~80 |
| `comercio_textil_2024.pdf` | Importaciones, exportaciones y balanza comercial textil (2024) | ~60 |
| `memoria_anual_inditex_2025.pdf` | Estrategia, sostenibilidad y resultados del Grupo Inditex (2025) | ~120 |
| `circularidad_textil.pdf` | Economía circular aplicada al textil: modelos, iniciativas y regulación europea | ~50 |
| `presentaciones_sectoriales_textil.pdf` | Estadísticas y evolución del tejido empresarial textil | ~40 |

> Total: casi 400 páginas indexadas

---

## Arquitectura del agente

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

---

## System Prompt — diseño y justificación

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

## Decisiones técnicas de la pipeline RAG

### Chunking

- **`chunk_size=1000`**: Tamaño suficiente para que cada fragmento contenga una idea completa. Con 500 caracteres (tamaño inicial probado) varios fragmentos quedaban cortados a mitad de una estadística o tabla, lo que degradaba la calidad de las respuestas.
- **`chunk_overlap=200`**: Solapa del 20% para evitar que conceptos que cruzan el límite de un chunk se pierdan al recuperar.
- **Separadores en orden de preferencia**: Se prioriza dividir por párrafo (`\n\n`), luego línea, luego frase. Solo se divide por espacio o carácter como último recurso.

### Embeddings — Gemini `text-embedding-004`

Se usan los embeddings nativos de Google Gemini en lugar de modelos locales de HuggingFace. Esto garantiza coherencia semántica entre los embeddings del índice y los embeddings de las queries en tiempo de consulta, al usar el mismo modelo en ambos momentos. El modelo `text-embedding-004` produce vectores de 768 dimensiones optimizados para recuperación de información.

### Retriever MMR

- **MMR**: Recupera candidatos por similitud (`fetch_k=20`) y luego selecciona los `k=6` que maximizan diversidad además de relevancia.
- **`lambda_mult=0.7`**: Peso 70% relevancia / 30% diversidad. Evita recuperar múltiples fragmentos del mismo párrafo del mismo documento.

---

## Memoria de conversación

La memoria se implementa con `MemorySaver` de LangGraph, que persiste el estado del grafo entre invocaciones usando un `thread_id`.

Cada llamada al agente con el mismo `thread_id` recibe el historial completo de mensajes anteriores. El campo `mensajes` del estado usa `add_messages` como reducer, que acumula los mensajes en lugar de sobreescribirlos.

**Demostración de memoria** (Ejemplo 4 del notebook).

## Interfaz Streamlit

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