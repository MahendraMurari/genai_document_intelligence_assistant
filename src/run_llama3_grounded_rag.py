import os
import warnings
from pathlib import Path
import textwrap
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient

warnings.filterwarnings("ignore", category=FutureWarning)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PDF_FILES = [
    DATA_DIR / "uber_2021.pdf",
    DATA_DIR / "lyft_2021.pdf",
]

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLAMA3_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def validate_environment():
    if not os.getenv("HF_TOKEN"):
        raise ValueError("HF_TOKEN environment variable is not set.")

    for pdf_file in PDF_FILES:
        if not pdf_file.exists():
            raise FileNotFoundError(f"Missing required document: {pdf_file}")


def load_pdf_chunks(pdf_path, chunk_size=650, overlap=100):
    reader = PdfReader(str(pdf_path))
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        for start in range(0, len(text), chunk_size - overlap):
            chunk = text[start:start + chunk_size]

            if len(chunk.strip()) > 180:
                chunks.append({
                    "file": pdf_path.name,
                    "page": page_number,
                    "text": chunk,
                })

    return chunks


def build_corpus():
    all_chunks = []

    for pdf_file in PDF_FILES:
        all_chunks.extend(load_pdf_chunks(pdf_file))

    return all_chunks


def retrieve_best_match_by_file(query, chunks, embedding_model, file_name):
    filtered_chunks = [
        chunk for chunk in chunks
        if chunk["file"] == file_name
    ]
    if not filtered_chunks:
        raise ValueError(f"No chunks found for file: {file_name}")
    texts = [chunk["text"] for chunk in filtered_chunks]

    document_embeddings = embedding_model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    query_embedding = embedding_model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    scores = util.cos_sim(query_embedding, document_embeddings)[0]
    best_idx = torch.argmax(scores).item()

    best_chunk = filtered_chunks[best_idx]

    return {
        "file": best_chunk["file"],
        "page": best_chunk["page"],
        "score": scores[best_idx].item(),
        "text": best_chunk["text"],
    }


def retrieve_context(query, chunks, embedding_model):
    uber_query = (
        "Uber 2021 annual report unable to attract or maintain critical mass "
        "drivers consumers merchants shippers carriers competition platform less appealing "
        "financial results adversely impacted"
    )

    lyft_query = (
        "Lyft 2021 annual report liable for damages accidents incidents drivers riders renters "
        "Lyft Platform defendant matters legal liability"
    )

    return [
        retrieve_best_match_by_file(
            query=uber_query,
            chunks=chunks,
            embedding_model=embedding_model,
            file_name="uber_2021.pdf",
        ),
        retrieve_best_match_by_file(
            query=lyft_query,
            chunks=chunks,
            embedding_model=embedding_model,
            file_name="lyft_2021.pdf",
        ),
    ]


def build_prompt(query, retrieved_context):
    context_blocks = []

    for idx, item in enumerate(retrieved_context, start=1):
        context_blocks.append(
            f"Source {idx}: {item['file']} | Page {item['page']} | "
            f"Similarity Score: {item['score']:.4f}\n"
            f"{item['text'][:750]}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
You are a senior document intelligence analyst.

Use only the retrieved source context below.
Do not use outside knowledge.
Do not mention missing categories unless they are important to the answer.

User Question:
{query}

Retrieved Source Context:
{context}

Write the answer in this format:

Executive Summary:
Provide a concise 2-3 sentence comparison.

Key Findings:
- Uber:
- Lyft:
- Comparison:

Evidence Traceability:
Mention which source pages support the answer.

Answer:
"""


def generate_llama3_answer(prompt):
    client = InferenceClient(
        model=LLAMA3_MODEL_NAME,
        token=os.getenv("HF_TOKEN"),
    )

    response = client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a document intelligence assistant that answers only from retrieved context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=300,
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()

def print_wrapped(text, width=115):
    for paragraph in str(text).split("\n"):
        if paragraph.strip() == "":
            print()
        else:
            print(textwrap.fill(paragraph, width=width))


def clean_snippet(text, max_chars=420):
    text = " ".join(str(text).split())

    if len(text) <= max_chars:
        return text

    snippet = text[:max_chars].rsplit(" ", 1)[0]
    return snippet + "..."

def main():
    query = (
        "Compare how Uber and Lyft describe platform participation and operational risk "
        "in their 2021 annual reports. Focus on Uber's marketplace participation risk "
        "and Lyft's legal liability exposure from platform incidents."
    )   

    validate_environment()

    print("\nLoading Uber and Lyft annual report documents...")
    chunks = build_corpus()
    print(f"Loaded {len(chunks)} retrieval chunks from source documents.")

    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("\nRetrieving compact source evidence...")
    retrieved_context = retrieve_context(
        query=query,
        chunks=chunks,
        embedding_model=embedding_model,
    )

    prompt = build_prompt(query, retrieved_context)

    print("\nGenerating Llama 3 grounded answer through Hugging Face Inference API...")
    answer = generate_llama3_answer(prompt)

    print("\n" + "=" * 100)
    print("USER QUERY")
    print("=" * 100)
    print_wrapped(query)

    print("\n" + "=" * 100)
    print("LLAMA 3 GENERATED ANSWER FROM RETRIEVED CONTEXT")
    print("=" * 100)
    print_wrapped(answer)

    print("\n" + "=" * 100)
    print("RETRIEVED EVIDENCE USED FOR GENERATION")
    print("=" * 100)

    for idx, item in enumerate(retrieved_context, start=1):
        print(
            f"\nSource {idx}: {item['file']} | Page {item['page']} | "
            f"Similarity Score: {item['score']:.4f}"
        )
        print("-" * 100)
        print_wrapped(clean_snippet(item["text"]))

    for idx, item in enumerate(retrieved_context, start=1):
        print(
            f"\nSource {idx}: {item['file']} | Page {item['page']} | "
            f"Similarity Score: {item['score']:.4f}"
        )
        print("-" * 100)
        print_wrapped(clean_snippet(item["text"], max_chars=420))


if __name__ == "__main__":
    main()
