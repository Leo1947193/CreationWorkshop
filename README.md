# Creative Workshop V1 – Socratic World-Building & Risk Analysis Loop

This repository implements an experimental “Creative Workshop” loop: an AI helps you flesh out a fictional world, builds an internal knowledge graph from your rules, runs a deep risk analysis (RATT), and then turns the highest–impact defect into a short story that makes the consequences emotionally vivid.

The system is **LLM‑driven end‑to‑end** (no hand‑coded defect templates) and is designed for local experimentation, not production.

---

## High‑Level Architecture

The system is composed of four interacting modules that share a single `GlobalState` (see `src/core/schemas.py`):

1. **SEE – Socratic Elaboration Engine (Module 1)**
   - Front‑facing chat agent that asks you Socratic questions about your world.
   - Each user message is:
     - Stored in `conversation_history`.
     - Parsed into axioms/facts by WM‑KG.
     - Persisted into a vector store (Chroma).

2. **WM‑KG – World‑Model Knowledge Graph (Module 2)**
   - Implemented in `src/modules/wm_kg.py`.
   - Uses a sentence‑transformer embedding model (configurable) + ChromaDB to store axioms/facts per session.
   - Keeps a versioned snapshot of “what the world currently believes”.

3. **CDA – Consequence & Defect Analyzer (Module 3)**
   - Implemented in `src/modules/cda_agent.py`.
   - A true **RATT (Retrieval‑Augmented Tree‑of‑Thought)** implemented with LangGraph:
     - Seeds a small set of “risk directions” from axioms.
     - Iteratively expands thought nodes, retrieves context via Dual RAG, and lets the LLM judge which branches matter.
     - Produces a list of `DefectReport` objects, sorted by risk score.
   - **Important:** There are *no* hard‑coded defect patterns or keyword rules; all risk identification lives in LLM prompts.

4. **CDNG – Conflict‑Driven Narrative Generator (Module 4)**
   - Implemented in `src/modules/cdng_chain.py`.
   - Takes `top_defect + axioms`, calls the LLM once, and asks it to produce a self‑contained short story where the defect is the core conflict.
   - If the LLM call fails, a simple placeholder text is generated instead; there is no frozen narrative template.

The modules are orchestrated by a top‑level LangGraph (`src/modules/main_graph.py`) and exposed via a FastAPI backend (`src/main.py`), with a React/Tailwind single‑page frontend (`frontend/`) providing the UI.

---

## Technology Stack

**Backend**
- Python 3.12+ (managed via [`uv`](https://github.com/astral-sh/uv))
- FastAPI (`src/main.py`) + Uvicorn
- LangChain / LangGraph
- ChromaDB (local persistent vector store)
- Sentence‑Transformers (embedding model, configurable)
- OpenRouter (all LLM calls go through a single OpenAI‑compatible endpoint)

**Frontend**
- Vite + React + TypeScript (`frontend/`)
- Tailwind CSS

---

## Project Layout

Key files and directories:

- `pyproject.toml` – Python project metadata and dependencies.
- `src/core/schemas.py` – Pydantic models for:
  - `GlobalState` (shared across all modules),
  - `WorldSpecificationSnapshot`,
  - `DefectReport`.
- `src/core/llm.py` – Single entry point for LLM access:
  - Uses OpenRouter by default (via `langchain_openai.ChatOpenAI`).
  - Respects `LLM_MODEL` and `OPENROUTER_*` env vars.
- `src/core/state_manager.py` – Disk‑backed persistence of `GlobalState` per session (JSON files under `db_storage/`).
- `src/main.py` – FastAPI app, CORS, and endpoints:
  - `POST /api/v1/session/init`
  - `POST /api/v1/chat`
  - `POST /api/v1/analyze`
- `src/modules/see_agent.py` – SEE LangGraph:
  - Adds user messages to state, calls WM‑KG, and generates Socratic questions via LLM.
- `src/modules/wm_kg.py` – World‑Model / Knowledge Graph:
  - Extracts axioms/facts via LLM, persists embeddings and metadata to ChromaDB.
  - Provides helper to aggregate axioms as text for CDA/CDNG.
- `src/modules/dual_rag.py` – Dual RAG retrievers:
  - `DualRARetriever`: merges internal (axioms) and external (web) context.
  - `ValidatedWebRetriever`: Tavily‑based web retrieval + light deduplication.
- `src/modules/cda_agent.py` – CDA RATT implementation:
  - Defines structured models (`DefectCandidate`, `ThoughtNode`, etc.).
  - Implements RATT nodes: `seed_problem_space`, `generate_thoughts`, `retrieve_context`, `judge_thought`, `evaluate_and_prune`, `generate_final_report`.
  - No hard‑coded defect templates; LLM decides risk dimensions and defects.
- `src/modules/cdng_chain.py` – Narrative generator:
  - Uses LLM to generate a story given axioms + top defect.
- `frontend/` – React + Tailwind SPA:
  - `src/App.tsx` implements:
    - Chat window (SEE),
    - “Analyze my world” button (CDA+CDNG),
    - Story display area.

---

## Environment & Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/Leo1947193/CreationWorkshop.git
cd CreationWorkshop   # your actual repo path

uv venv .venv
source .venv/bin/activate
```

### 2. Install backend dependencies

```bash
uv pip install -e .
```

This uses `pyproject.toml` to install FastAPI, LangGraph, Chroma, LangChain, etc.

### 3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in the required keys:

```bash
cp .env.example .env
```

The important variables are:

- `LLM_MODEL`  
  OpenRouter model slug to use for *all* LLM calls, e.g.:
  - `openrouter/openai/gpt-5.1-chat`
  - `google/gemini-pro`
  - `meta-llama/llama-3.1-70b-instruct`

- `OPENROUTER_API_KEY`  
  Your OpenRouter API key.

- `OPENROUTER_BASE_URL` (optional)  
  Usually `https://openrouter.ai/api/v1`.

- `OPENROUTER_REFERER`, `OPENROUTER_TITLE` (optional, recommended)  
  Used by OpenRouter for attribution and dashboard display.

- `EMBEDDING_MODEL`  
  The sentence‑transformer model name, e.g. `BAAI/bge-large-zh-v1.5`.

- `TAVILY_API_KEY` (optional but recommended if you want web retrieval).

> **Note:** The backend expects `LLM_MODEL` and `OPENROUTER_API_KEY` to be present; if they are missing it will raise at startup.

---

## Running the System

### Start the backend

In one terminal:

```bash
cd /home/leo/CreationWorkshop  # adjust to your project path
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)  # load env vars (Linux/macOS)

uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will listen on `http://0.0.0.0:8000`.

### Start the frontend

In another terminal:

```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

Vite will show the dev server URL, typically `http://localhost:5173`.

Open that URL in your browser; the UI will call FastAPI on port 8000.

---

## Usage Workflow

1. **Initialize a session**
   - The frontend automatically calls `POST /api/v1/session/init` to get a `session_id`.
   - The backend creates a new `GlobalState` and a dedicated Chroma directory for this session.

2. **Iterate with SEE (chat)**
   - Type your world‑building ideas in the chat.
   - Each message:
     - is appended to `GlobalState.conversation_history`,
     - is analyzed by WM‑KG to extract axioms/facts,
     - is persisted to Chroma (`db_storage/{session_id}/internal_kb/`).
   - SEE then asks you a Socratic follow‑up question, often pointing out missing or ambiguous parts of your design.

3. **Trigger analysis**
   - Click **“Analyze My World”**.
   - The frontend calls `POST /api/v1/analyze` with your `session_id`.
   - The backend:
     1. Loads state and axioms from disk.
     2. Runs CDA’s RATT graph:
        - Seeds risk directions,
        - Expands a small tree of thoughts,
        - Uses Dual RAG + LLM judge to identify defects,
        - Produces a sorted list of `DefectReport`.
     3. If at least one defect is found, sets `top_defect` and triggers CDNG; otherwise, returns a message suggesting you add more specific rules.

4. **Read the story**
   - Once CDA completes, CDNG runs:
     - It retrieves all axioms (as text),
     - Builds a narrative prompt with axioms + `top_defect`,
     - Calls OpenRouter LLM once to generate a short story.
   - The story is displayed in the right‑hand panel.
   - The idea is that the story makes the defect’s consequences “felt” rather than just listed.

5. **Refine and repeat**
   - After reading the story, you can go back to SEE:
     - Adjust your world rules (e.g., add energy backup, social checks, resource flows),
     - Click “Analyze My World” again,
     - Observe how CDA discovers new, subtler defects over iterations.

---

## API Overview

All endpoints are JSON‑based and live under `src/main.py`:

- `POST /api/v1/session/init`
  - Request body: `{}`  
  - Response: `{"session_id": "uuid-..."}`  
  - Initializes (or reuses) a per‑session state + storage directory.

- `POST /api/v1/chat`
  - Request: `{"session_id": "...", "message": "..."}`  
  - Response:  
    ```json
    {
      "session_id": "...",
      "response": "Socratic follow-up question...",
      "conversation_length": 7
    }
    ```

- `POST /api/v1/analyze`
  - Request: `{"session_id": "..."}`  
  - Response:
    ```json
    {
      "session_id": "...",
      "story": "Generated short story...",
      "top_defect": {
        "defect_id": "D-001",
        "description": "...",
        "ratt_branch": "...",
        "likelihood": 4,
        "severity": 5,
        "risk_score": 20,
        "long_term_consequence": "..."
      }
    }
    ```

---

## Performance Notes

The CDA module is intentionally LLM‑heavy (true RATT with Dual RAG). To keep analysis time reasonable:

- The RATT graph is bounded by:
  - `max_depth = 2`,
  - `max_nodes = 12`.
- Each iteration:
  - seeds at most 3 initial risk directions,
  - expands at most 3 parent nodes,
  - generates at most 2 child thoughts per parent,
  - judges at most 6 nodes per iteration.

If you want deeper analysis (at the cost of latency), you can adjust these parameters in `src/modules/cda_agent.py` – but beware that each extra node usually adds at least one LLM call via OpenRouter.

For profiling, the repo supports using tools like `pyinstrument` or `scalene` by wrapping the uvicorn entry point.

---

## Development Notes

- **State & Persistence**
  - State is JSON‑serialized in `db_storage/{session_id}/state.json`.
  - Internal KB (Chroma) lives under `db_storage/{session_id}/internal_kb/`.

- **Testing**
  - There is currently no automated test suite; manual testing via the frontend + API calls is recommended.

- **Extensibility**
  - You can plug in any OpenRouter model by changing `LLM_MODEL`.
  - Embedding model is fully configurable via `EMBEDDING_MODEL`.
  - CDA RATT prompts can be refined to steer what kinds of defects are surfaced (while avoiding hard‑coded patterns).

---

## License

This project is an experimental prototype. Add your preferred license here (e.g., MIT, Apache‑2.0) before publishing publicly.

