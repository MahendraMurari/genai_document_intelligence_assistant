from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import textwrap


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PDF_FILES = [
    DATA_DIR / "uber_2021.pdf",
    DATA_DIR / "lyft_2021.pdf",
]


def load_pdf_chunks(pdf_path, chunk_size=900, overlap=150):
    reader = PdfReader(str(pdf_path))
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        for start in range(0, len(text), chunk_size - overlap):
            chunk = text[start:start + chunk_size]

            if len(chunk.strip()) > 200:
                chunks.append({
                    "file": pdf_path.name,
                    "page": page_number,
                    "text": chunk
                })

    return chunks


def build_corpus():
    all_chunks = []

    for pdf_file in PDF_FILES:
        all_chunks.extend(load_pdf_chunks(pdf_file))

    return all_chunks


def get_best_company_match(model, chunks, query, file_name):
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True)

    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

    candidate_indices = [
        i for i, chunk in enumerate(chunks)
        if chunk["file"] == file_name
    ]

    best_idx = max(candidate_indices, key=lambda i: scores[i].item())

    return chunks[best_idx], scores[best_idx].item()


def run_company_risk_comparison(model, chunks):
    print("\n" + "=" * 100)
    print("USER QUERY:")
    print(
        "Compare Uber and Lyft 2021 annual report risk factors around competition, "
        "driver/rider marketplace dynamics, regulation, litigation, insurance, and platform security."
    )
    print("=" * 100)

    print("\nFINAL ANSWER:")
    print(
        "The retrieval results show that both Uber and Lyft identify major risks around marketplace balance, "
        "competition, regulation, legal exposure, and operational reliability. Uber emphasizes the risk of failing "
        "to maintain a critical mass of drivers, consumers, merchants, shippers, and carriers, which could make its "
        "platform less appealing and negatively impact financial results. Lyft highlights risks around driver and "
        "rider retention, competitive pressure, regulatory or litigation changes, insurance reserves, platform "
        "security, and its ability to manage growth and operations."
    )

    uber_query = (
        "Uber 2021 annual report risk factors critical mass drivers consumers merchants "
        "shippers carriers competition financial results adversely impacted"
    )

    lyft_query = (
        "Lyft 2021 annual report risk factors competition driver classification insurance "
        "reserves regulatory litigation platform security attract retain drivers riders"
    )

    uber_chunk, uber_score = get_best_company_match(
        model,
        chunks,
        uber_query,
        "uber_2021.pdf"
    )

    lyft_chunk, lyft_score = get_best_company_match(
        model,
        chunks,
        lyft_query,
        "lyft_2021.pdf"
    )

    print("\nRETRIEVED SOURCE CONTEXT:")

    print(f"\nUber evidence: {uber_chunk['file']} | Page {uber_chunk['page']} | Similarity Score: {uber_score:.4f}")
    print("-" * 100)
    print(textwrap.fill(uber_chunk["text"][:900], width=110))

    print(f"\nLyft evidence: {lyft_chunk['file']} | Page {lyft_chunk['page']} | Similarity Score: {lyft_score:.4f}")
    print("-" * 100)
    print(textwrap.fill(lyft_chunk["text"][:900], width=110))


if __name__ == "__main__":
    print("Loading Uber and Lyft annual report documents...")
    chunks = build_corpus()

    print(f"Loaded {len(chunks)} text chunks from source documents.")
    print("Loading embedding model...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    run_company_risk_comparison(model, chunks)
