# GenAI Document Intelligence Assistant

A GenAI document intelligence project that uses Retrieval Augmented Generation to search across unstructured documents, retrieve relevant context, and generate grounded answers with LLMs.

The project combines LangChain, LlamaIndex, vector search, document specific query engines, and ReAct based tools to support question answering across mixed document sources such as business reports, policy style documents, operational guides, and knowledge files.

## What This Project Covers

Document loading and preprocessing

Text chunking for retrieval

Embedding generation

Vector search with ChromaDB and FAISS style retrieval patterns

Question answering with retrieved context

LangChain based RAG flow with Phi 3

LlamaIndex based query engine flow with Llama 3

ReAct based tool use for document routing and utility functions

Multi document retrieval across different document types

## Architecture

The workflow follows this pattern:

Documents are loaded from local files

Text is split into smaller chunks

Embeddings are generated for each chunk

Chunks are stored in a vector index

User questions are embedded and matched against stored chunks

Relevant context is passed to the LLM

The LLM generates a grounded answer based on retrieved context

Agentic tools are used for document specific query routing and utility operations

## Project Structure

```text
genai_document_intelligence_assistant/
│
├── src/
│   ├── rag_phi3_langchain.py
│   ├── rag_phi3_demo.py
│   ├── rag_llamaindex.py
│   ├── llm_init.py
│   ├── llm_tasks.py
│   └── rag_math_tools.py
│
├── sample_docs/
│   ├── lyft_2021.pdf
│   ├── uber_2021.pdf
│   └── MSAI.txt
│
├── outputs/
│   └── rag_outputs.pdf
│
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
