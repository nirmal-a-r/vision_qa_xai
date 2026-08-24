"""
llm_narrator.py
===============
Local-LLM reporting layer over the certified control loop (Ollama / Qwen).

Placement is the whole design decision here. The LLM does **not** decide
anything. Every accept / review / reject call, every recalibration trigger and
every acquisition choice is made by `InspectionAgent` under a conformal
certificate, and the certificate is what the guarantee rests on. A language
model cannot carry that guarantee: it will produce fluent, confident prose for a
decision it has no bound on, which is precisely the failure mode this project
argues against.

What a local LLM *is* good for, and what it does here, is turning certified
machine state into something a shift supervisor can act on:

  * a plain-language shift report from the episode's counters
  * root-cause hypotheses from the observed defect-class mix
  * a per-part justification of why the policy routed it to a human

Every one of those is a *description of a decision already made*. If the model
is unavailable, hallucinates, or returns nonsense, the inspection outcome is
byte-identical - only the prose is missing. That is the property that makes it
safe to put a 7B model in an industrial loop at all, and it is why the fallback
chain below degrades silently to a deterministic template rather than blocking.

Model preference order (first available wins):
    qwen2.5:7b        - best instruction following of the local options
    qwen2.5:3b        - lighter, still adequate for structured summarisation
    phi4-mini         - already present on this machine, works as a fallback
    <deterministic>   - no LLM at all; templated text, always available

Nothing in the notebook fails if Ollama is not running.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error

OLLAMA_URL = "http://127.0.0.1:11434"
PREFERRED_MODELS = ["qwen2.5:7b", "qwen2.5:7b-instruct", "qwen2.5:3b", "phi4-mini:latest", "phi4-mini"]


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def ollama_available(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def installed_models(timeout: float = 3.0) -> list:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return []


def pick_model(preferred=None) -> str | None:
    """First preferred model that is actually installed, else None."""
    have = installed_models()
    for want in (preferred or PREFERRED_MODELS):
        for h in have:
            if h == want or h.split(":")[0] == want.split(":")[0]:
                return h
    return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(prompt: str, model: str = None, timeout: float = 120.0,
             temperature: float = 0.2, num_predict: int = 400) -> str | None:
    """One-shot completion. Returns None on any failure - never raises.

    Low temperature on purpose: this is summarisation of numeric state, where
    creative variation is a defect rather than a feature.
    """
    model = model or pick_model()
    if model is None:
        return None
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response", "").strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Deterministic fallbacks (always available)
# ---------------------------------------------------------------------------

def _template_shift_report(stats: dict) -> str:
    n = stats.get("n_parts", 0)
    acc = stats.get("auto_accept", 0)
    rev = stats.get("human_review", 0)
    rej = stats.get("auto_reject", 0)
    esc = stats.get("escapes", 0)
    alpha = stats.get("alpha")
    lines = [
        f"Shift summary: {n} parts inspected.",
        f"  auto-accepted {acc} ({100*acc/max(n,1):.1f}%), "
        f"routed to review {rev} ({100*rev/max(n,1):.1f}%), "
        f"auto-rejected {rej} ({100*rej/max(n,1):.1f}%).",
    ]
    if alpha is not None:
        lines.append(f"  escape budget alpha={alpha}; observed escapes {esc} "
                     f"({100*esc/max(n,1):.2f}% of parts).")
    if stats.get("recalibrations"):
        lines.append(f"  {stats['recalibrations']} recalibration(s) triggered by drift.")
    if stats.get("drift_alarms"):
        lines.append(f"  {stats['drift_alarms']} drift alarm(s) raised.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shift_report(stats: dict, model: str = None) -> dict:
    """Operator-facing shift report. Always returns text; flags its source."""
    prompt = f"""You are a quality-engineering assistant writing a factory shift report.

Below is machine-verified inspection state. Every decision was made by a
statistically certified policy, NOT by you. Your job is only to summarise and
flag what a supervisor should look at.

Rules:
- Use ONLY the numbers given. Never invent a figure.
- Be concise: at most 8 short bullet points.
- If the escape budget was respected, say so plainly; do not editorialise.
- End with one line: "ACTION:" and the single most useful next step.

Inspection state (JSON):
{json.dumps(stats, indent=2)}
"""
    txt = generate(prompt, model=model)
    if txt:
        return {"text": txt, "source": model or pick_model(), "llm": True}
    return {"text": _template_shift_report(stats), "source": "deterministic-template",
            "llm": False}


def explain_routing(record: dict, policy: dict, model: str = None) -> dict:
    """Why did the certified policy route THIS part to a human?"""
    prompt = f"""A certified inspection policy routed one part for human review.
Explain to the operator, in 3 sentences maximum, why - using only these numbers.

Do not second-guess the decision and do not suggest overriding it; the policy
carries a statistical guarantee and you do not.

Part: {json.dumps(record, indent=2)}
Policy thresholds: {json.dumps(policy, indent=2)}
"""
    txt = generate(prompt, model=model, num_predict=200)
    if txt:
        return {"text": txt, "source": model or pick_model(), "llm": True}
    s = record.get("max_score")
    f = record.get("faithfulness")
    return {"text": (f"Routed to review: top detection confidence {s} falls between the "
                     f"auto-accept threshold {policy.get('lambda_lo')} and the auto-reject "
                     f"threshold {policy.get('lambda_hi')}"
                     + (f", and explanation faithfulness {f} is below the gate "
                        f"{policy.get('phi')}." if f is not None else ".")),
            "source": "deterministic-template", "llm": False}


def root_cause_hypotheses(class_counts: dict, dataset: str, model: str = None) -> dict:
    """Candidate process causes for the observed defect mix.

    Explicitly framed as hypotheses to check, never as conclusions - the model
    has no access to the line and cannot verify anything it proposes.
    """
    prompt = f"""You are advising a process engineer on a {dataset} inspection line.

Observed defect class counts this shift:
{json.dumps(class_counts, indent=2)}

Give at most 4 candidate PROCESS causes for this particular mix, each with the
one measurement that would confirm or rule it out. These are hypotheses to
check, not conclusions - you cannot see the line. Be specific to the defect
types listed. One line each, no preamble.
"""
    txt = generate(prompt, model=model, num_predict=350)
    if txt:
        return {"text": txt, "source": model or pick_model(), "llm": True}
    top = sorted(class_counts.items(), key=lambda kv: -kv[1])[:3]
    return {"text": "Most frequent classes this shift: "
                    + ", ".join(f"{k} ({v})" for k, v in top)
                    + ". (LLM unavailable - no hypotheses generated.)",
            "source": "deterministic-template", "llm": False}


def status() -> dict:
    """Environment probe for the notebook to print."""
    up = ollama_available()
    have = installed_models() if up else []
    chosen = pick_model() if up else None
    return {
        "ollama_running": up,
        "installed_models": have,
        "selected_model": chosen,
        "mode": "llm" if chosen else "deterministic-template",
        "note": ("LLM narrates certified decisions; it never makes them, so "
                 "inspection outcomes are identical either way."),
    }
