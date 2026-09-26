# GenAI Document Assistant — Microsoft Foundry RAG Q&A System

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that ingests
unstructured documents, indexes them for semantic search, and answers
questions with grounded, citation-backed responses generated through a
Microsoft Foundry project (Azure OpenAI + Azure AI Search).

```
┌────────────┐    ┌─────────────┐    ┌───────────────────┐    ┌──────────────────┐
│  Documents  │ →  │  Chunking   │ →  │ Azure OpenAI       │ →  │  Azure AI Search  │
│ .txt/.pdf/  │    │ (token-     │    │ Embeddings         │    │  (vector +        │
│ .docx       │    │  aware,     │    │ (text-embedding-3) │    │  semantic index)  │
│             │    │  overlap)   │    │                    │    │                   │
└────────────┘    └─────────────┘    └───────────────────┘    └─────────┬─────────┘
                                                                          │ hybrid search
                                                                          ▼
┌──────────────┐   ┌────────────────────┐   ┌────────────────────────────────────┐
│ Groundedness/ │ ← │  Azure OpenAI Chat  │ ← │  Top-k retrieved passages +        │
│ Relevance     │   │  (grounded prompt) │   │  question → grounded prompt        │
│ Evaluation    │   │                    │   │                                     │
└──────────────┘   └────────────────────┘   └────────────────────────────────────┘
```

## What this demonstrates (mapped to the resume bullets)

| Resume bullet | Where it lives in the code |
|---|---|
| Built an end-to-end RAG pipeline using Microsoft Foundry to ingest and chunk unstructured documents | `src/ingestion.py` — token-aware chunking with overlap over .txt/.pdf/.docx |
| Implemented embedding generation and semantic search using Azure OpenAI and Azure AI Search | `src/embeddings.py`, `src/search_index.py`, `src/retrieval.py` — vector + hybrid semantic search |
| Configured LLM-powered response generation with grounded prompts, reducing hallucinations | `src/generation.py` — grounding rules baked into the system prompt, citations enforced |
| Evaluated response groundedness and relevance against retrieved context | `src/evaluation.py` — LLM-as-judge scorer + optional `azure-ai-evaluation` SDK integration |

## Project layout

```
rag-document-assistant/
├── .env.example              # All Foundry/Azure env vars — copy to .env
├── requirements.txt
├── main.py                   # CLI: ingest / ask / evaluate / demo
├── data/sample_docs/         # Sample docs for a self-contained demo
└── src/
    ├── config.py              # Loads & validates env vars
    ├── ingestion.py            # Load + token-aware chunk documents
    ├── embeddings.py           # Azure OpenAI embeddings client
    ├── search_index.py         # Azure AI Search index schema + upload
    ├── retrieval.py            # Hybrid vector + keyword + semantic re-rank
    ├── generation.py           # Grounded prompt + chat completion
    ├── evaluation.py           # Groundedness / relevance scoring
    └── mock_clients.py         # Offline mode — no Azure credentials needed
```

## Setup

### 1. Provision Azure resources (one-time, in the Azure/Foundry portal)
1. Create (or open) a **Microsoft Foundry** project.
2. Inside it, deploy:
   - A **chat model** (e.g. `gpt-4o`) → note the *deployment name*.
   - An **embedding model** (e.g. `text-embedding-3-large`) → note the *deployment name*.
3. Create an **Azure AI Search** service (Basic tier or above supports vector + semantic search).
4. Grab the keys/endpoints for both resources from the Azure Portal.

### 2. Configure environment variables
```bash
cp .env.example .env
```
Then fill in `.env` with your real values:

```bash
AZURE_OPENAI_ENDPOINT=https://<your-foundry-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-azure-openai-key>
AZURE_OPENAI_API_VERSION=2024-06-01
AZURE_OPENAI_CHAT_DEPLOYMENT=<your-chat-deployment-name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<your-embedding-deployment-name>

FOUNDRY_PROJECT_NAME=<your-foundry-project-name>
FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project-name>

AZURE_SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
AZURE_SEARCH_API_KEY=<your-search-admin-key>
AZURE_SEARCH_INDEX_NAME=document-assistant-index
```

**Never commit `.env`.** It's already covered by `.gitignore`.

### 3. Install dependencies
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run it

With real Azure credentials:
```bash
python main.py ingest --docs data/sample_docs      # chunk, embed, index
python main.py ask --question "What is covered under the standard product warranty?"
python main.py evaluate --question "What is covered under the standard product warranty?"
```

**Without any Azure account (offline demo mode)** — useful for a live interview
walkthrough on a machine with no provisioned resources:
```bash
python main.py --mock demo
```
This runs ingest → ask → evaluate in one shot using deterministic local
embeddings and a canned grounded-answer generator, so you can show the full
architecture working end-to-end with zero cloud dependency or cost.

## How grounding/hallucination-reduction actually works here

The system prompt in `src/generation.py` requires the model to (1) answer
only from retrieved context, (2) explicitly say when it doesn't know, and
(3) cite the source chunk for every claim. This is enforced at the prompt
layer, not via fine-tuning — cheap to iterate on and directly auditable.

## How evaluation works

`src/evaluation.py` implements two paths:
- A lightweight **LLM-as-judge** (`llm_judge`) that scores groundedness and
  relevance 1–5 with a rationale, using the same chat deployment.
- The **production path** (`azure_ai_evaluation_judge`) wired to Microsoft's
  official `GroundednessEvaluator` / `RelevanceEvaluator` from the
  `azure-ai-evaluation` SDK, which integrates with Foundry's evaluation and
  observability tooling.

---

## Interview talking points

**"Walk me through the architecture."**
Documents are loaded and split into token-bounded, overlapping chunks so
neither the embedding model's nor the LLM's context window is exceeded and
sentence-boundary information isn't lost. Each chunk is embedded via Azure
OpenAI and pushed into an Azure AI Search index that supports both vector
similarity (HNSW) and keyword (BM25) search. At query time I run hybrid
search — vector + keyword — then apply Azure's semantic re-ranker, take the
top-k passages, and feed them into a grounded prompt for the chat model.
Finally, I score the answer's groundedness and relevance against the
retrieved context to catch regressions.

**"Why hybrid search instead of pure vector search?"**
Pure embedding similarity can blur exact terms — product codes, acronyms,
proper nouns — that keyword/BM25 search catches reliably. Combining both,
then semantically re-ranking, consistently gave better recall on
domain-specific queries than either alone.

**"How did you reduce hallucinations?"**
Primarily through prompt-level grounding constraints: instructing the model
to answer only from provided context, to say "I don't know" instead of
guessing, and to cite sources per claim — combined with a low temperature
setting. This is measurable, not just a hope: the groundedness evaluator
gives a concrete score I can track across prompt changes.

**"How do you measure whether the system is actually working?"**
Two axes: groundedness (are the claims actually supported by retrieved
context — this is what catches hallucination) and relevance (does the
answer address what was asked, independent of grounding). I implemented
this both as a lightweight custom LLM-judge and wired up to Microsoft's
official `azure-ai-evaluation` evaluators for the production path.

**"What would you improve with more time?"**
- Add re-ranking with a cross-encoder for even tighter top-k precision.
- Add per-chunk metadata filters (department, doc date) for scoped retrieval.
- Move batch evaluation into a CI step that runs on a fixed golden Q&A set
  so groundedness/relevance regressions are caught before deploy.
- Add streaming responses for lower perceived latency in the chat UI.

**"What was the trickiest part?"**
Tuning chunk size/overlap and top-k together — too small a chunk loses
context needed to ground an answer; too large wastes context window and
dilutes the retrieved signal. I treated this as a small offline evaluation
loop: fix a golden question set, sweep chunk size/top-k, and pick the
config that maximized groundedness score, not just qualitative "looks right."

---

## Notes on the `--mock` flag

`src/mock_clients.py` provides deterministic, hash-based pseudo-embeddings
and a canned grounded-answer generator so the entire pipeline can be
demoed offline. This is intentional engineering, not a shortcut to hide —
it decouples the pipeline's control flow from any specific cloud dependency,
which is exactly what makes it possible to unit-test retrieval logic or do
a live interview demo without live Azure credentials or incurring API cost.
