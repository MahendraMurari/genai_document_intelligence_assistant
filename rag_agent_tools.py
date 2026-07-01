from typing import Literal


RiskCategory = Literal[
    "marketplace_participation",
    "legal_liability",
    "regulatory_exposure",
    "platform_security",
    "general_operational_risk",
]


Priority = Literal["Low", "Medium", "High", "Critical"]


def classify_risk_signal(evidence_text: str) -> RiskCategory:
    """
    Classify retrieved document evidence into a business risk category.
    This is a deterministic keyword-based helper used by the agent workflow.
    """
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
    """
    Calculate a normalized operational risk score.

    Formula:
    risk_score = base_score * evidence_confidence * business_impact_multiplier
    """
    if not 0 <= evidence_confidence <= 1:
        raise ValueError("Evidence confidence must be between 0 and 1.")

    if base_score < 0:
        raise ValueError("Base score cannot be negative.")

    if business_impact_multiplier < 0:
        raise ValueError("Business impact multiplier cannot be negative.")

    return round(base_score * evidence_confidence * business_impact_multiplier, 2)


def assign_review_priority(risk_score: float) -> Priority:
    """
    Assign review priority based on calculated risk score.
    """
    if risk_score >= 85:
        return "Critical"

    if risk_score >= 70:
        return "High"

    if risk_score >= 40:
        return "Medium"

    return "Low"


def recommend_action(category: RiskCategory, priority: Priority) -> str:
    """
    Recommend the next action based on risk category and priority.
    """
    if priority in ["Critical", "High"]:
        if category == "legal_liability":
            return "Route to legal and operations review with supporting document evidence."

        if category == "marketplace_participation":
            return "Route to marketplace operations review for driver/rider supply-demand impact analysis."

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
    """
    End-to-end deterministic tool used by the agent workflow.

    It classifies retrieved evidence, calculates a risk score,
    assigns priority, and recommends the next review action.
    """
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

if __name__ == "__main__":
    user_query = (
        "Assess the retrieved Lyft annual report evidence for operational or legal risk. "
        "Classify the risk, assign a review priority, and recommend the next action."
    )

    evidence_source = "lyft_2021.pdf"
    evidence_page = 109
    similarity_score = 0.8598

    retrieved_evidence = (
        "claimed, and may claim in the future, that the Company is liable for damages "
        "related to accidents or other incidents involving drivers, riders, or renters "
        "using or who have used services offered on the Lyft Platform, as well as from "
        "third parties. The Company is currently named as a defendant in a number of "
        "matters related to accidents or other incidents involving drivers on the Lyft "
        "Platform, other riders, renters and third parties."
    )

    print("\n" + "=" * 100)
    print("AGENTIC RISK TRIAGE TOOL EXECUTION")
    print("=" * 100)

    print("\nUser Query / Task:")
    print(user_query)

    print("\nEvidence Source:")
    print(f"{evidence_source} | Page {evidence_page} | Similarity Score: {similarity_score}")

    print("\nRetrieved Evidence Chunk:")
    print(retrieved_evidence)

    print("\nAgent Tool Plan:")
    print("1. Read the retrieved evidence chunk.")
    print("2. Classify the risk signal using deterministic tool logic.")
    print("3. Calculate risk score using retrieval confidence and business impact.")
    print("4. Assign review priority.")
    print("5. Recommend the next review action.")

    print("\nTool Invocation:")
    print("generate_risk_triage_decision(evidence_text, base_score=80, evidence_confidence=0.8598, business_impact_multiplier=1.32)")

    result = generate_risk_triage_decision(
        evidence_text=retrieved_evidence,
        base_score=80,
        evidence_confidence=similarity_score,
        business_impact_multiplier=1.32,
    )

    print("\nTool Result:")
    print(f"risk_category: {result['risk_category']}")
    print(f"risk_score: {result['risk_score']}")
    print(f"review_priority: {result['review_priority']}")
    print(f"recommended_action: {result['recommended_action']}")

    print("\nFinal Agent Decision:")
    print(
        "The retrieved Lyft evidence is classified as a legal-liability risk signal. "
        "Because the score falls in the critical range, it should be routed to legal "
        "and operations review with the source document evidence attached."
    )