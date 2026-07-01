# GenAI Document Intelligence Assistant

A document intelligence system that uses Retrieval-Augmented Generation, vector search, LlamaIndex/LangChain workflows, local LLM experimentation, and agent-based tool execution to search, compare, and reason over unstructured documents.

The project focuses on a practical GenAI workflow: processing large PDF/text sources, converting them into searchable chunks, retrieving the most relevant context, generating grounded answers with source references, and using deterministic tools for structured follow-up decisions.

The sample data uses public annual reports, but the same workflow can be applied to policies, compliance documents, financial reports, SOPs, legal files, HR manuals, insurance documents, or internal knowledge bases.

```text
Documents -> chunks -> embeddings -> vector retrieval -> grounded LLM answer -> tool execution -> evidence traceability
```

---

## Why I Built This

Important business information is often buried inside long documents. Keyword search can help, but it usually does not understand the user’s question or connect related context across multiple files.

I built this project to demonstrate how a RAG-based document assistant can retrieve relevant evidence first, then use an LLM to generate a grounded answer based on that source context.

The project also includes an agent-based tool layer. The LLM is used for summarization and reasoning over retrieved context, while deterministic Python tools handle structured operations such as risk classification, scoring, priority assignment, and recommended action.

---

## What This Project Includes

### Multi-document RAG

The project processes multiple PDF/text documents, splits them into chunks, creates embeddings, and retrieves the most relevant sections for a user query.

The output includes the query, retrieved sources, page numbers, similarity scores, and evidence snippets.

### Grounded Llama 3 answer generation

A focused workflow retrieves compact source evidence and sends only that context to Llama 3 through Hugging Face Inference API.

This keeps the answer grounded in retrieved documents instead of relying on open-ended generation.

### LlamaIndex workflow

The LlamaIndex implementation creates document-specific query engines and exposes them through a tool-based interface. This is useful when different documents need to be searched separately and compared.

### LangChain / Phi-3 local RAG workflow

The repository also includes a LangChain + Phi-3 Mini workflow with ChromaDB experimentation. This part is included as a local LLM RAG implementation path.

### Agent-based risk triage

The agent-based workflow takes retrieved evidence and passes it into Python tools that classify the signal, calculate a score, assign priority, and recommend the next action.

This avoids asking the LLM to generate operational scores or decisions without deterministic logic.

---

## Repository Structure

| Path | Purpose |
|---|---|
| `src/run_document_risk_intelligence_demo.py` | Main end-to-end demo combining retrieval, Llama 3 response generation, tool execution, and evidence traceability |
| `src/run_llama3_grounded_rag.py` | Focused Llama 3 grounded RAG workflow |
| `src/rag_agent_tools.py` | Agent-based helper tools for risk classification, scoring, priority assignment, and action recommendation |
| `src/run_rag_demo.py` | Lightweight custom retrieval demo using embeddings and cosine similarity |
| `src/rag_llamaindex.py` | LlamaIndex-based document retrieval workflow |
| `src/llm_init.py` | LLM initialization and Hugging Face token handling |
| `src/llm_tasks.py` | Shared helper functions for LLM and retrieval tasks |
| `src/rag_phi3_langchain.py` | LangChain + Phi-3 local RAG workflow |
| `src/rag_phi3_demo.py` | Demo runner for the Phi-3 workflow |
| `sample_docs/` | Sample documents used for RAG workflows |
| `outputs/` | Screenshots of successful runs |

---

## Output Examples

### Document intelligence workflow

This run demonstrates the full workflow in one place: document ingestion, chunking, embedding-based retrieval, Llama 3 response generation, agent-based triage, source pages, similarity scores, and final decision.

![Document intelligence workflow](outputs/document_intelligence_workflow.png)

---

### Grounded Llama 3 RAG answer

This output focuses on the Llama 3 RAG workflow. The assistant retrieves source-specific evidence first and then generates an answer using only that retrieved context.

![Grounded Llama 3 RAG answer](outputs/llama3_grounded_rag_answer.png)

---

### Agentic risk triage workflow

This output demonstrates the tool-execution layer. A retrieved evidence chunk is passed into deterministic tools that return a risk category, score, priority, recommended action, and final decision.

![Agentic risk triage workflow](outputs/agentic_risk_triage_workflow.png)

---

## Tech Stack

- Python
- Llama 3
- Hugging Face Inference API
- SentenceTransformers
- LlamaIndex
- LangChain
- ChromaDB
- Microsoft Phi-3 Mini
- PDF parsing
- Vector similarity search
- Agent-based Python tools

---

## Setup

Clone the repository:

```bash
git clone https://github.com/MahendraMurari/genai_document_intelligence_assistant.git
cd genai_document_intelligence_assistant
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Add your Hugging Face token locally:

```text
HF_TOKEN=your_huggingface_token_here
```

Do not commit `.env` to GitHub.

---

## Running the Project

Run the main document intelligence workflow:

```bash
python src/run_document_risk_intelligence_demo.py
```

Run the focused Llama 3 grounded RAG workflow:

```bash
python src/run_llama3_grounded_rag.py
```

Run the agent-based tool workflow:

```bash
python src/rag_agent_tools.py
```

Run the optional Phi-3 LangChain workflow:

```bash
python src/rag_phi3_demo.py
```

---

## Notes

The main demo uses local PDF parsing, embedding generation, and in-memory vector similarity search. The LangChain/Phi-3 workflow includes ChromaDB-based vector-store experimentation.

Generated vector indexes, local cache files, and secrets are intentionally excluded from the repository.

Before pushing, check that no real Hugging Face token is present:

```bash
grep -R "hf_" .
```

Only `.env.example` should include:

```text
HF_TOKEN=your_huggingface_token_here
```

---

## What This Project Demonstrates

This project demonstrates how a document assistant can be built around retrieval-first design:

- split long documents into searchable chunks
- retrieve relevant context with embeddings
- generate grounded answers from source evidence
- compare information across documents
- preserve source names, page numbers, and similarity scores
- use deterministic tools for structured decisions
- keep secrets and generated artifacts out of source control

The final result is a practical GenAI document intelligence workflow that is inspectable, modular, and extensible.
