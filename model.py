"""
Mini-Jev — HuggingFace Inference API, no local GPU or model required.

Uses the HF zero-shot-classification endpoint:
  Noul:   classify into ["yes", "no"]
  Choice: classify into arbitrary user-defined choices
  Score:  classify into scale values, then compute E[score]

Token is loaded from .env.local (HF_TOKEN) or passed directly to the constructor.
"""

from __future__ import annotations

import math
import os
import time

import requests
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(".env.local")


HF_MODEL   = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class NoulResult:
    answer: bool
    probability: float   # P(yes)
    confidence: float    # 0=uncertain, 1=certain

    def __repr__(self):
        return (
            f"NoulResult(\n"
            f"  answer      = {self.answer}\n"
            f"  probability = {self.probability:.4f}   # P(yes)\n"
            f"  confidence  = {self.confidence:.4f}\n"
            f")"
        )


@dataclass
class ChoiceResult:
    answer: str
    probabilities: dict[str, float]
    confidence: float

    def __repr__(self):
        rows = "\n".join(f"    {k}: {v:.4f}" for k, v in self.probabilities.items())
        return (
            f"ChoiceResult(\n"
            f"  answer        = {self.answer!r}\n"
            f"  probabilities = {{\n{rows}\n  }}\n"
            f"  confidence    = {self.confidence:.4f}\n"
            f")"
        )


@dataclass
class ScoreResult:
    value: float
    distribution: dict[int | float, float]
    confidence: float

    def __repr__(self):
        rows = "\n".join(f"    {k}: {v:.4f}" for k, v in self.distribution.items())
        return (
            f"ScoreResult(\n"
            f"  value        = {self.value:.4f}   # E[score]\n"
            f"  distribution = {{\n{rows}\n  }}\n"
            f"  confidence   = {self.confidence:.4f}\n"
            f")"
        )


# ── Core model ────────────────────────────────────────────────────────────────

def _apply_temperature(probs: list[float], temperature: float) -> list[float]:
    """Temperature scaling via logit space. τ>1 → softer, τ<1 → sharper."""
    if temperature == 1.0:
        return probs
    logits = [math.log(max(p, 1e-9) / max(1 - p, 1e-9)) for p in probs]
    scaled = [l / temperature for l in logits]
    exp_s  = [math.exp(s) for s in scaled]
    total  = sum(exp_s)
    return [e / total for e in exp_s]


class MiniJev:
    """
    Zero-shot typed decisions via HuggingFace Inference API.

    Token: https://huggingface.co/settings/tokens (free, read-only is enough)
    Set HF_TOKEN in .env.local, or pass hf_token= directly to the constructor.
    """

    def __init__(
        self,
        hf_token: str | None = None,
        temperature: float = 1.0,
        timeout: int = 60,
    ):
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "HF_TOKEN is required.\n"
                "  Add HF_TOKEN=hf_xxxx to .env.local, or\n"
                "  pass MiniJev(hf_token='hf_xxxx')"
            )
        self.headers     = {"Authorization": f"Bearer {token}"}
        self.temperature = temperature
        self.timeout     = timeout

    def _call(
        self,
        text: str,
        candidate_labels: list[str],
        hypothesis_template: str = "The answer is {}.",
    ) -> dict[str, float]:
        """
        Call HF zero-shot-classification endpoint.
        Returns {label: probability} normalised to sum to 1.
        Retries automatically on model cold-start (HTTP 503).
        """
        payload = {
            "inputs": text,
            "parameters": {
                "candidate_labels": candidate_labels,
                "hypothesis_template": hypothesis_template,
            },
        }
        for attempt in range(3):
            resp = requests.post(
                HF_API_URL, headers=self.headers, json=payload, timeout=self.timeout
            )
            if resp.status_code == 503:
                # Model not loaded yet (cold start) — wait and retry
                wait = resp.json().get("estimated_time", 20)
                print(f"  Model loading, waiting {wait:.0f}s...")
                time.sleep(min(wait, 30))
                continue
            resp.raise_for_status()
            data = resp.json()
            raw_probs = data["scores"]  # already softmaxed by HF API
            scaled    = _apply_temperature(raw_probs, self.temperature)
            return {label: round(p, 4) for label, p in zip(data["labels"], scaled)}

        raise RuntimeError("HF API call failed after 3 attempts")

    # ── Noul ──────────────────────────────────────────────────────────────────

    def noul(self, state: str, question: str) -> NoulResult:
        text   = f"{state.strip()}\n\n{question.strip()}"
        probs  = self._call(text, ["yes", "no"], hypothesis_template="The answer to the question is {}.")
        p_yes  = probs["yes"]
        conf   = abs(p_yes - 0.5) * 2

        return NoulResult(
            answer=p_yes >= 0.5,
            probability=round(p_yes, 4),
            confidence=round(conf, 4),
        )

    # ── Choice ────────────────────────────────────────────────────────────────

    def choice(self, state: str, question: str, choices: list[str]) -> ChoiceResult:
        text  = f"{state.strip()}\n\nQuestion: {question.strip()}"
        probs = self._call(text, choices, hypothesis_template="The answer is {}.")
        best  = max(probs, key=probs.get)

        return ChoiceResult(
            answer=best,
            probabilities=probs,
            confidence=round(probs[best], 4),
        )

    # ── Score ─────────────────────────────────────────────────────────────────

    def score(
        self,
        state: str,
        question: str,
        scale: list[int | float],
    ) -> ScoreResult:
        lo, hi  = scale[0], scale[-1]
        labels  = [str(v) for v in scale]
        text    = f"{state.strip()}\n\nQuestion: {question.strip()}"
        probs   = self._call(
            text, labels,
            hypothesis_template=f"The score is {{}} on a scale of {lo} to {hi}.",
        )
        dist     = {v: probs[str(v)] for v in scale}
        expected = sum(v * dist[v] for v in scale)

        return ScoreResult(
            value=round(expected, 4),
            distribution=dist,
            confidence=round(max(dist.values()), 4),
        )
