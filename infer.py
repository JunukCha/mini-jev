"""
Mini-Jev — demo for all three decision types.

Usage:
    python infer.py
"""

from model import MiniJev

jev = MiniJev()
print(f"Model loaded. Temperature = {jev.temperature}\n")


# ── Noul ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("NOUL — Binary Decision")
print("=" * 60)

state = "I would like a refund please."

print(f"State: {state}\n")

result_refund = jev.noul(state=state, question="Is this message a refund request?")
print(f"Q: Is this message a refund request?")
print(result_refund)

result_urgent = jev.noul(state=state, question="Is this request urgent?")
print(f"Q: Is this request urgent?")
print(result_urgent)


# ── Choice ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CHOICE — Categorical Decision")
print("=" * 60)

state = "I cannot log in to my account."
question = "Which category does this inquiry belong to?"
choices = {
    "auth":    "Authentication / login issue",
    "billing": "Billing issue",
    "bug":     "Bug report",
    "general": "General inquiry",
}

result = jev.choice(state=state, question=question, choices=choices)
print(f"State   : {state}")
print(f"Question: {question}")
print(result)


# ── Score ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCORE — Ordinal Decision")
print("=" * 60)

state = "I absolutely love this product! I will definitely buy it again."
question = "What is the sentiment score of this review?"
criteria = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]

result = jev.score(state=state, question=question, criteria=criteria)
print(f"State   : {state}")
print(f"Question: {question}")
print(result)
