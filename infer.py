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

result_refund = jev.noul(state=state, question="Is this message a refund request?")
print(f"State   : {state}")
print(f"Q: Is this message a refund request?")
print(result_refund)

result_urgent = jev.noul(state=state, question="Is this request urgent?")
print(f"\nQ: Is this request urgent?")
print(result_urgent)


# ── Choice ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CHOICE — Categorical Decision")
print("=" * 60)

state = "I cannot log in to my account."
question = "Which category does this inquiry belong to?"
choices = [
    "Authentication / login issue",
    "Billing issue",
    "Bug report",
    "General inquiry",
]

result = jev.choice(state=state, question=question, choices=choices)
print(f"State   : {state}")
print(f"Question: {question}")
print(result)


# ── Score ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCORE — Ordinal Decision")
print("=" * 60)

state = "I absolutely love this product! I will definitely buy it again."
question = "What is the sentiment score of this review? (1=Very negative, 2=Negative, 3=Neutral, 4=Positive, 5=Very positive)"
scale = [1, 2, 3, 4, 5]

result = jev.score(state=state, question=question, scale=scale)
print(f"State   : {state}")
print(f"Question: {question}")
print(result)
