# Mini-Jev

A lightweight NLI-based decision engine inspired by [TypeSafe AI's Jev](https://typesafe.ai).  
Makes typed probabilistic decisions — **Noul**, **Choice**, **Score** — with zero training, using the HuggingFace Inference API.

## Decision Types

| Type | Output | Use Case |
|------|--------|----------|
| `noul` | `{answer: bool, probability, confidence}` | Yes/No questions |
| `choice` | `{answer: str, probabilities, confidence}` | Multi-class classification |
| `score` | `{value: float, distribution, confidence}` | Ordinal scoring |

## How It Works

Each decision is powered by **Natural Language Inference (NLI)** via `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` on the HuggingFace Inference API — no local model, no GPU, no training required.

- **Noul**: classifies input into `["yes", "no"]`, returns calibrated P(yes)
- **Choice**: classifies into arbitrary user-defined labels, softmax over entailment scores
- **Score**: classifies into numeric scale values, computes expected value E[score]

## Setup

**1. Clone and create a virtual environment**

```bash
git clone https://github.com/JunukCha/mini-jev.git
cd mini-jev
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

**2. Get a HuggingFace token**

Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a token (read-only is sufficient).

**3. Create `.env.local`**

```
HF_TOKEN=hf_your_token_here
```

See [.env.example](.env.example) for the format.

## Usage

```bash
python infer.py
```

```python
from model import MiniJev

jev = MiniJev()

# Noul — binary decision
result = jev.noul(
    state="I would like a refund please.",
    question="Is this message a refund request?",
)
# NoulResult(answer=True, probability=0.9231, confidence=0.8462)

# Choice — categorical decision
result = jev.choice(
    state="I cannot log in to my account.",
    question="Which category does this inquiry belong to?",
    choices=["Authentication / login issue", "Billing issue", "Bug report", "General inquiry"],
)
# ChoiceResult(answer='Authentication / login issue', ...)

# Score — ordinal decision
result = jev.score(
    state="I absolutely love this product!",
    question="Sentiment score (1=Very negative … 5=Very positive)",
    scale=[1, 2, 3, 4, 5],
)
# ScoreResult(value=4.7, ...)
```

## Files

```
model.py          — MiniJev class (Noul, Choice, Score)
infer.py          — example usage
requirements.txt  — dependencies
.env.example      — token setup guide
```

## Requirements

- Python 3.10+
- HuggingFace account (free tier)
- Internet connection (API calls)

No GPU, no local model download, no training.