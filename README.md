# Expert Agent for the Textile and Fashion Industry in Spain

Artificial intelligence assistant that answers questions about the textile and fashion industry in Spain from its own documentary knowledge base. Applies **Retrieval-Augmented Generation (RAG)** with local HuggingFace E5 embeddings, ChromaDB as vector store, Google Gemini as LLM, and LangGraph to orchestrate the flow with conversation memory. Includes interactive web interface in Streamlit.

---

## Chosen Domain

**Textile and fashion industry in Spain** — a sector with rich documentary sources that allows exercising the RAG pipeline with queries requiring synthesis across heterogeneous sources:

- 📊 **Economic analysis**: evolution of the number of companies, revenue, and employment in the sector
- 🌍 **Foreign trade**: trade balance, main export and import markets
- 🏢 **Business strategy**: financial results, structure, and sustainability policy of Inditex
- 📋 **European position**: comparison of the Spanish textile sector within the EU
- 💡 **Sectoral trends**: digitalization, ecommerce, and evolution of subsectors

### Knowledge base — 4 PDF documents (322 pages)

| Document | Content |
|-----------|-----------|
| `economic-report-fashion-spain-2025.pdf` | Annual economic report of the fashion sector in Spain |
| `textile-trade-2024.pdf` | Textile imports, exports, and trade balance |
| `inditex-annual-report-2025.pdf` | Strategy, sustainability, and results of Inditex Group |
| `sectoral-presentations-textile.pdf` | Statistics and evolution of the textile business ecosystem |

---

## Technology Stack

| Component | Technology |
|------------|------------|
| **LLM** | Google Gemini (`gemini-2.0-flash`) via `langchain-google-genai` |
| **Embeddings** | `intfloat/multilingual-e5-base` (HuggingFace, local, no quotas) |
| **Vector store** | ChromaDB via `langchain-chroma` |
| **Agent framework** | LangGraph + `MemorySaver` |
| **PDF loading** | `PyPDFLoader` (`langchain-community`) |
| **Web interface** | Streamlit |

> **Why E5 and not Gemini Embeddings**: We attempted to use `gemini-embedding-001` but the free tier (100 req/min) generated unrecoverable 429 errors when indexing 681 chunks. `multilingual-e5-base` runs locally, has no quotas, and is specifically trained for retrieval tasks in multiple languages including Spanish.

---

## Execution Guide

### 1. Prerequisites

- Python 3.13+
- Google Gemini API key

### 2. Installation

```bash
git clone <url-del-repo>
cd gemini_agent
```

### 3. Configure API key

Create the `.env` file in the project root:

```
GOOGLE_API_KEY=your_key_here
```

### 4. Generate the vector knowledge base

Run the complete notebook in order

This indexes the 4 PDFs in ChromaDB with E5 embeddings. Only needs to be executed once (or when documents change).

### 5. Launch the web interface (Streamlit Cloud)

[Streamlit Cloud - Textile Agent](https://geminiagent-z9tcmwt4mrmgarz4svyrjk.streamlit.app/)

## Graph Architecture

The agent is implemented as a state graph in LangGraph with three sequential nodes:

```
START → [reformulate_node] → [retrieve_node] → [generate_node] → END
```

### Nodes

**`reformulate_node`**
The LLM rewrites the user's question by expanding it with synonyms and specific textile sector vocabulary before searching in ChromaDB. Significantly improves semantic retrieval when the user formulates questions in colloquial language.

```
Input  → original user question
Output → reformulated_query (enriched with sectoral vocabulary)
```

**`retrieve_node`**
Searches for the most relevant fragments in ChromaDB using the reformulated query. Uses **MMR (Maximal Marginal Relevance)** with `fetch_k=20`, `k=6`, and `lambda_mult=0.7` (70% relevance / 30% diversity), which prevents retrieving multiple fragments from the same paragraph.

```
Input  → reformulated_query
Output → context (6 chunks formatted with source and page)
```

**`generate_node`**
Calls the LLM with the system prompt, the retrieved context, and the **complete conversation history**, allowing coherent responses that reference previous turns.

```
Input  → system prompt + RAG context + message history
Output → agent response
```

### Memory

Implemented with LangGraph's `MemorySaver`. The `messages` field of the state uses `add_messages` as reducer, accumulating history between turns via `thread_id`. Each session in the Streamlit app has its own `thread_id` generated with `uuid4`.

---

## System Prompt Justification

```
You are an expert analyst in the textile and fashion industry in Spain.

Your knowledge comes from the documents provided as context in each turn.
You always respond in Spanish, with a professional and concise tone.

Rules:
- Base your answers on the provided documentary context.
- You can synthesize and relate information from your previous responses in the conversation,
  as long as the original statements come from the documentary context.
- If the context does not contain sufficient information and you cannot infer it from the history, clearly indicate this.
- Do not invent data, figures, or statistics that do not appear in the context or previous history.
- When citing numerical data, mention the source if available in the context.
```

| Decision | Justification |
|----------|---------------|
| **Expert analyst role** | Orients the tone toward structured and professional responses, appropriate for a technical-economic domain. |
| **"Base your answers on documentary context"** | Anchors the model to indexed documents and reduces hallucinations about sectoral data and statistics. |
| **History synthesis allowed** | Enables coherent follow-up responses without fabricating new information. Synthesis is only allowed if the original source was documentary. |
| **"Clearly indicate if you don't have information"** | The agent explicitly admits the limits of its knowledge base instead of inventing plausible answers. |
| **Cite sources when available** | Allows direct traceability to original documents, especially useful with numerical data. |
| **`temperature=0.3`** | Low temperature for consistent and factual responses. An economic-sectoral domain requires reproducibility, not creativity. |

---

## Dependencies

Managed in `requirements.txt`:

```
requires-python = ">=3.13"
dependencies = [
    "chromadb",               # Vector store
    "google-generativeai",    # List of available Gemini models
    "ipywidgets",             # Widgets for Jupyter
    "langchain",              # LangChain base framework
    "langchain-chroma",       # ChromaDB integration (no deprecation warning)
    "langchain-community",    # PyPDFLoader, HuggingFaceEmbeddings
    "langchain-google-genai", # ChatGoogleGenerativeAI (LLM)
    "langchain-text-splitters",
    "langgraph",              # Agent orchestration with state graph
    "notebook",               # Jupyter Notebook
    "pypdf",                  # PDF loading
    "python-dotenv",          # Load environment variables from .env
    "sentence-transformers",  # HuggingFaceEmbeddings backend
    "streamlit",              # Web interface
]
```

---

## Possible Improvements

**Context classifier node (graph short-circuit)**
Add an initial `classify_node` that uses the LLM to determine whether the question is relevant to the Spanish textile/fashion domain. If the answer is negative, the graph jumps directly to an `out_of_context_node` that responds politely without executing any RAG search. This avoids unnecessary calls to ChromaDB and the complete pipeline when the user formulates questions outside the domain.

```
                          ┌─ [reformulate] → [retrieve] → [generate] → END
START → [classify] ───────┤
                          └─ [out_of_context] → END
```

The implementation in LangGraph requires adding the `in_context: bool` field to the state and using `add_conditional_edges` from the classifier node with a routing function that returns `"reformulate"` or `"out_of_context"` based on the result.

**Other possible improvements**
- **Reranking**: add a reranking step (e.g., with `cross-encoder/ms-marco-MiniLM`) after retrieval to reorder chunks before passing them to the LLM, improving precision especially with ambiguous questions.
- **Automated evaluation**: implement an evaluation pipeline with RAGAS metrics (faithfulness, answer relevancy, context recall) to objectively measure system quality when making changes to the retriever or prompt.
- **Incremental index update**: instead of reindexing all documents when adding a new one, implement an incremental update strategy in ChromaDB that only processes new chunks.
