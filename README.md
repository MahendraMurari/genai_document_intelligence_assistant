# GenAI Document Intelligence Assistant

A document intelligence system that uses Retrieval-Augmented Generation, vector search, local LLM workflows, and agent-style tool execution to search, compare, and reason over unstructured documents.

The project focuses on a practical GenAI pattern: taking large PDF/text sources, converting them into searchable knowledge, retrieving the most relevant context, generating grounded answers with source references, and using deterministic tools for structured follow-up decisions.

The core workflow is:

```text
Documents → chunks → embeddings → vector retrieval → grounded LLM response → agent-style tool execution → evidence traceability
```

For demonstration, the project uses public annual reports as sample documents. The same architecture can be adapted to enterprise policies, compliance manuals, financial reports, insurance documents, legal files, SOPs, HR knowledge bases, and customer support documentation.

---

## Why I Built This

A common problem in enterprise AI is that useful information is buried inside long documents such as reports, policies, manuals, internal knowledge files, and operational references.

A normal keyword search can find matching words, but it does not always understand the question, connect related context across documents, or provide enough evidence for a user to trust the result.

This project explores how a retrieval-based assistant can improve that workflow by combining semantic search, document chunking, vector retrieval, LLM-based summarization, and controlled tool execution.

The goal is not just to generate an answer. The goal is to make the answer traceable through retrieved source context, document names, page references, similarity scores, and structured follow-up actions.

---

## What This Project Demonstrates

This repository includes multiple connected workflows that show different parts of a document intelligence system.

### 1. Multi-document RAG over unstructured documents

The system loads large PDF/text documents, splits them into chunks, embeds the text, and retrieves the most relevant sections for a user query.

The main examples use public business reports as sample data to demonstrate document comparison, evidence retrieval, grounded summarization, page-level traceability, and similarity scoring.

### 2. Grounded Llama 3 responses with source traceability

The project uses retrieved context as the base for LLM responses instead of relying on open-ended generation.

The focused Llama 3 workflow retrieves compact evidence from multiple documents and sends only that context to Llama 3 through Hugging Face Inference API. The output includes an executive summary, key findings, source document names, page numbers, similarity scores, and retrieved evidence snippets.

### 3. LlamaIndex-based document retrieval

The LlamaIndex workflow creates document-specific query engines and exposes them through a tool-style interface.

This makes it possible to route questions to the right document source and retrieve targeted context from each file. This is useful for multi-document question answering where different files represent different knowledge sources.

### 4. Agent-style tool execution from retrieved evidence

The project includes an agentic risk triage workflow where retrieved document evidence is passed into deterministic Python tools.

Instead of asking the LLM to invent business decisions, the tool layer classifies the retrieved signal, calculates a confidence-weighted score, assigns review priority, and recommends the next action.

This demonstrates a practical agent-style pattern:

```text
Retrieved evidence → tool invocation → structured tool output → final decision
```

### 5. Local LLM and vector-store experimentation

The repository also includes supporting LangChain/Phi-3 workflows with ChromaDB-based vector storage experimentation.

These files show an additional local LLM RAG path, while the main showcase workflow focuses on cleaner grounded generation, evidence traceability, and deterministic tool execution.

---

## Repository Structure

| Path | Purpose |
|---|---|
| `src/run_document_risk_intelligence_demo.py` | Main end-to-end workflow combining document ingestion, vector retrieval, grounded LLM response, agentic triage, and evidence traceability |
| `src/run_llama3_grounded_rag.py` | Focused grounded RAG demo using Llama 3 with retrieved source evidence |
| `src/rag_agent_tools.py` | Deterministic agent-style tools for risk classification, scoring, priority assignment, and recommended action |
| `src/run_rag_demo.py` | Lightweight custom retrieval demo showing PDF parsing, chunking, embeddings, cosine similarity, and source comparison |
| `src/rag_llamaindex.py` | LlamaIndex workflow for document-specific retrieval and query-engine based routing |
| `src/llm_init.py` | LLM initialization/configuration for Hugging Face and LlamaIndex workflows |
| `src/llm_tasks.py` | Reusable helper functions for completion, chat, document loading, indexing, and retrieval |
| `src/rag_phi3_langchain.py` | LangChain + Phi-3 local RAG workflow with ChromaDB experimentation |
| `src/rag_phi3_demo.py` | Demo runner for the Phi-3 LangChain workflow |
| `sample_docs/` | Sample public documents used for retrieval workflows |
| `outputs/` | Screenshots of successful workflow outputs |

---

## Core Components

### `run_document_risk_intelligence_demo.py`

Main end-to-end showcase workflow.

It ingests sample business documents, chunks PDF content, generates embeddings, retrieves source-specific evidence, sends compact retrieved context to Llama 3, and passes retrieved risk evidence into deterministic triage tools.

This script demonstrates the complete workflow:

```text
document ingestion → semantic retrieval → grounded generation → agent-style tool execution → evidence traceability
```

### `run_llama3_grounded_rag.py`

Focused Llama 3 RAG demo over multiple documents.

This script retrieves compact source evidence from each report and generates a grounded analyst-style answer through Hugging Face Inference API.

The output includes source document names, page numbers, similarity scores, retrieved evidence, and a Llama 3 generated summary.

### `rag_agent_tools.py`

Deterministic agent-style risk triage tools.

These functions classify retrieved evidence into a risk category, calculate a confidence-weighted score, assign review priority, and recommend the next action.

The goal is to avoid asking the LLM to invent scores or operational decisions. The LLM handles grounded summarization, while Python tools handle structured triage logic.

### `run_rag_demo.py`

Lightweight custom retrieval demo.

It shows the basic retrieval mechanics directly: PDF text extraction, chunking, embedding generation, cosine similarity search, and source comparison.

This file is useful because it makes the retrieval layer transparent without depending on a larger framework.

### `rag_llamaindex.py`

LlamaIndex-based document retrieval workflow.

It creates document query engines and exposes them through a tool-style interface, allowing the assistant to retrieve from specific documents and answer questions using document-grounded context.

### `llm_init.py`

Central LLM initialization module.

It configures Hugging Face/LlamaIndex model access and loads the Hugging Face token through the `HF_TOKEN` environment variable instead of hardcoding credentials.

### `llm_tasks.py`

Reusable helper layer for LLM and RAG tasks, including direct completion, chat-style generation, document loading, embedding setup, vector index creation, and simple retrieval workflows.

### `rag_phi3_langchain.py` and `rag_phi3_demo.py`

Secondary LangChain and Phi-3 Mini RAG path.

This workflow explores local LLM experimentation with PDF loading, recursive text splitting, Hugging Face embeddings, ChromaDB vector storage, and retrieved-context question answering.

This path is kept as a supporting implementation branch, while the main showcase uses Llama 3 for cleaner hosted generation and source-grounded outputs.

---

## Architecture

### Document Intelligence Workflow

```text
User Query
   ↓
Document Loader
   ↓
Text Chunking
   ↓
Embedding Model
   ↓
Vector Retrieval
   ↓
Relevant Source Context
   ↓
Grounded LLM Response
   ↓
Agent-Style Tool Execution
   ↓
Evidence Traceability + Final Decision
```

### Agent Tool-Execution Workflow

```text
Retrieved Evidence
   ↓
Risk Signal Classification
   ↓
Risk Score Calculation
   ↓
Priority Assignment
   ↓
Recommended Action
   ↓
Final Review Decision
```

The LLM is used for grounded summarization. Risk scoring and routing decisions are handled by deterministic Python tools so the model does not invent operational scores or priorities.

---

## Output Examples

### Document intelligence workflow

This output shows the complete workflow: documents are ingested, split into retrieval chunks, embedded, searched using semantic retrieval, summarized with Llama 3, and passed into an agent-style risk triage layer.

The output includes source pages, similarity scores, evidence traceability, tool execution, and a final workflow decision.

![Document intelligence workflow](outputs/document_intelligence_workflow.png)

---

### Grounded Llama 3 RAG answer

This output focuses on the grounded RAG generation flow.

The assistant retrieves compact source evidence from multiple documents and sends only that context to Llama 3 through Hugging Face Inference API. The answer includes an executive summary, key findings, source document names, page numbers, similarity scores, and evidence snippets.

![Grounded Llama 3 RAG answer](outputs/llama3_grounded_rag_answer.png)

---

### Agentic risk triage workflow

This output demonstrates agent-style tool execution from retrieved evidence.

The workflow starts with a user task, shows the evidence source, uses the retrieved evidence chunk as tool input, invokes deterministic Python tools, and returns a structured risk category, score, priority, recommended action, and final decision.

![Agentic risk triage workflow](outputs/agentic_risk_triage_workflow.png)

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| LLM | Llama 3 through Hugging Face Inference API |
| Embeddings | SentenceTransformers, Hugging Face embedding models |
| Retrieval | Vector similarity search, cosine similarity, source-specific retrieval |
| RAG frameworks | LlamaIndex, LangChain |
| Vector-store experiment | ChromaDB |
| Local LLM experiment | Microsoft Phi-3 Mini |
| Document processing | PDF parsing, chunking, metadata tracking |
| Agent-style tools | Deterministic Python functions for risk scoring and triage |
| Security | Environment variables for token handling |

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

Run the agent-style risk triage tool workflow:

```bash
python src/rag_agent_tools.py
```

Run the optional Phi-3 LangChain workflow:

```bash
python src/rag_phi3_demo.py
```

---

## Notes on Retrieval and Vector Storage

The main demo uses local PDF parsing, embedding generation, and in-memory vector similarity retrieval. This keeps the workflow transparent and easy to inspect.

The repository also includes a secondary LangChain/Phi-3 path that explores ChromaDB-based persisted vector-store retrieval.

Generated vector indexes are local runtime artifacts and are intentionally excluded from the repository.

---

## Repository Hygiene

The repository avoids committing secrets, generated indexes, local cache files, or IDE-specific files.

Ignored examples include:

```text
.env
__pycache__/
*.pyc
mydb_index/
new_db_index/
chroma_db/
*.user
*.sln
*.pyproj
.vs/
.DS_Store
```

Before pushing changes, check that no real Hugging Face token is present:

```bash
grep -R "hf_" .
```

Only `.env.example` should reference:

```text
HF_TOKEN=your_huggingface_token_here
```

---

## What This Project Shows

This project demonstrates how document intelligence systems can be built around retrieval-first design.

The important parts are:

- converting long documents into searchable chunks
- using embeddings for semantic retrieval
- grounding answers in retrieved source context
- comparing information across multiple documents
- exposing document query engines as tools
- using deterministic helper functions in an agent-style workflow
- preserving source names, page references, and similarity scores
- keeping secrets and generated artifacts out of source control

The result is a practical GenAI workflow that is inspectable, modular, and closer to how real document assistants are built.
