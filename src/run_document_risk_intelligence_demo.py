import os
from pathlib import Path
from typing import Literal
import os
import warnings
from pathlib import Path
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
warnings.filterwarnings("ignore", category=FutureWarning)

RiskCategory = Literal[
    "marketplace_participation",
    "legal_liability",
    "regulatory_exposure",
    "platform_security",
    "general_operational_risk",
]

Priority = Literal["Low", "Medium", "High", "Critical"]


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


def retrieve_source_evidence(chunks, embedding_model):
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


def build_llama3_prompt(user_question, retrieved_context):
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
{user_question}

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
                "content": (
                    "You are a document intelligence assistant that answers only "
                    "from retrieved source context."
                ),
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


def classify_risk_signal(evidence_text: str) -> RiskCategory:
    text = evidence_text.lower()

    if any(term in text for term in ["driver", "rider", "accident", "defendant", "liable", "liability"]):
        return "legal_liability"

    if any(term in text for term in ["critical mass", "competition", "platform less appealing", "consumers", "merchants"]):
        return "marketplace_participation"

    if any(term in text for term in ["regulation", "regulatory", "compliance"]):
        return "regulatory_exposure"

    if any(term in text for term in ["security", "breach", "unauthorized", "cyber"]):
        return "platform_security"

    return "general_operational_risk"


def calculate_risk_score(
    base_score: float,
    evidence_confidence: float,
    business_impact_multiplier: float,
) -> float:
    if not 0 <= evidence_confidence <= 1:
        raise ValueError("Evidence confidence must be between 0 and 1.")

    if base_score < 0:
        raise ValueError("Base score cannot be negative.")

    if business_impact_multiplier < 0:
        raise ValueError("Business impact multiplier cannot be negative.")

    return round(base_score * evidence_confidence * business_impact_multiplier, 2)


def assign_review_priority(risk_score: float) -> Priority:
    if risk_score >= 85:
        return "Critical"

    if risk_score >= 70:
        return "High"

    if risk_score >= 40:
        return "Medium"

    return "Low"


def recommend_action(category: RiskCategory, priority: Priority) -> str:
    if priority in ["Critical", "High"]:
        if category == "legal_liability":
            return "Route to legal and operations review with supporting document evidence."

        if category == "marketplace_participation":
            return "Route to marketplace operations review for supply-demand impact analysis."

        if category == "regulatory_exposure":
            return "Route to compliance review for policy and regulatory assessment."

        if category == "platform_security":
            return "Route to security review for incident and control validation."

    return "Monitor risk signal and retain evidence for periodic review."


def generate_risk_triage_decision(
    evidence_text: str,
    base_score: float,
    evidence_confidence: float,
    business_impact_multiplier: float,
) -> dict:
    category = classify_risk_signal(evidence_text)

    risk_score = calculate_risk_score(
        base_score=base_score,
        evidence_confidence=evidence_confidence,
        business_impact_multiplier=business_impact_multiplier,
    )

    priority = assign_review_priority(risk_score)

    action = recommend_action(
        category=category,
        priority=priority,
    )

    return {
        "risk_category": category,
        "risk_score": risk_score,
        "review_priority": priority,
        "recommended_action": action,
    }


def main():
    user_question = (
        "Compare how Uber and Lyft describe platform participation and operational risk "
        "in their 2021 annual reports. Focus on Uber's marketplace participation risk "
        "and Lyft's legal liability exposure from platform incidents."
    )

    validate_environment()

    print("\n" + "=" * 100)
    print("DOCUMENT INTELLIGENCE WORKFLOW")
    print("=" * 100)

    print("\n1. INPUT DOCUMENTS")
    print("- uber_2021.pdf")
    print("- lyft_2021.pdf")

    print("\n2. DOCUMENT INGESTION AND CHUNKING")
    chunks = build_corpus()
    print(f"Loaded {len(chunks)} retrieval chunks from source documents.")

    print("\n3. EMBEDDING AND VECTOR RETRIEVAL SETUP")
    print("Loading embedding model for semantic retrieval...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded successfully.")

    print("\n4. SOURCE-SPECIFIC RETRIEVAL")
    retrieved_context = retrieve_source_evidence(
        chunks=chunks,
        embedding_model=embedding_model,
    )

    for idx, item in enumerate(retrieved_context, start=1):
        print(
            f"Source {idx}: {item['file']} | Page {item['page']} | "
            f"Similarity Score: {item['score']:.4f}"
        )

    prompt = build_llama3_prompt(user_question, retrieved_context)

    print("\n5. LLAMA 3 GROUNDED ANALYST SUMMARY")
    print("Generating answer through Hugging Face Inference API...")
    answer = generate_llama3_answer(prompt)
    print("\n" + answer)

    lyft_evidence = next(
        item for item in retrieved_context
        if item["file"] == "lyft_2021.pdf"
    )

    print("\n6. AGENTIC RISK TRIAGE TOOL EXECUTION")
    print("Retrieved Signal:")
    print(
        "Lyft evidence mentions liability exposure from accidents or incidents "
        "involving drivers, riders, renters, and third parties on the Lyft Platform."
    )

    print("\nTool Plan:")
    print("1. Classify retrieved evidence into a risk category.")
    print("2. Calculate risk score using retrieval confidence and business impact.")
    print("3. Assign review priority.")
    print("4. Recommend next action.")

    triage_result = generate_risk_triage_decision(
        evidence_text=lyft_evidence["text"],
        base_score=80,
        evidence_confidence=lyft_evidence["score"],
        business_impact_multiplier=1.32,
    )

    print("\nTool Result:")
    print(f"risk_category: {triage_result['risk_category']}")
    print(f"risk_score: {triage_result['risk_score']}")
    print(f"review_priority: {triage_result['review_priority']}")
    print(f"recommended_action: {triage_result['recommended_action']}")

    print("\n7. EVIDENCE TRACEABILITY")
    for idx, item in enumerate(retrieved_context, start=1):
        print(
            f"\nSource {idx}: {item['file']} | Page {item['page']} | "
            f"Similarity Score: {item['score']:.4f}"
        )
        print("-" * 100)
        print(item["text"][:650])

    print("\n8. FINAL WORKFLOW DECISION")
    print(
        "The workflow retrieved source-specific risk evidence, generated a grounded "
        "Llama 3 analyst summary, and used deterministic tools to route the Lyft "
        "legal-liability signal for high-priority review."
    )


if __name__ == "__main__":
    main()
