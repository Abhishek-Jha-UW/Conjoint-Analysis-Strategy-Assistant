from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

try:
    # Optional dependency: used for estimation if present.
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    LogisticRegression = None  # type: ignore[assignment]


# =========================
# Public configuration types
# =========================


@dataclass(frozen=True)
class StudyConfig:
    """
    Defaults chosen to keep end-to-end runs interactive (~30-45s)
    for typical attribute counts, while producing enough signal to demo.
    """

    n_respondents: int = 40
    n_tasks_per_respondent: int = 8
    n_alternatives: int = 3
    include_none_option: bool = True

    # Concurrency controls for synthetic generation.
    max_concurrent_requests: int = 8

    # Model selection. Keep as a parameter so app can override.
    # (Names evolve; set in app/config rather than hard-coding.)
    model_name: str = "gpt-4.1-mini"

    # Retry behavior for occasional JSON formatting issues.
    max_retries_per_respondent: int = 2

    # Seed for reproducibility (design + sampling). If None, random.
    seed: Optional[int] = 42


class AttributeSpec(BaseModel):
    name: str = Field(..., min_length=1)
    levels: List[str] = Field(..., min_length=2)


class ChoiceTask(BaseModel):
    task_id: int
    alternatives: List[Dict[str, str]]  # list of profiles (attribute -> level)
    none_option: bool = False


class RespondentChoices(BaseModel):
    respondent_id: int
    # For each task: chosen alternative index in [0..n_alts-1], or -1 for None.
    choices: List[int]


# =========================
# Core: design generation
# =========================


def normalize_attribute_specs(attributes: Sequence[Dict[str, Any]] | Sequence[AttributeSpec]) -> List[AttributeSpec]:
    parsed: List[AttributeSpec] = []
    for a in attributes:
        parsed.append(a if isinstance(a, AttributeSpec) else AttributeSpec.model_validate(a))
    return parsed


def build_random_cbc_design(
    *,
    attributes: Sequence[Dict[str, Any]] | Sequence[AttributeSpec],
    config: StudyConfig,
) -> List[ChoiceTask]:
    """
    Builds a simple randomized CBC design.
    For portfolio/demo purposes we prefer speed and simplicity over
    statistically optimized designs (D-efficient designs can be a later upgrade).
    """
    attrs = normalize_attribute_specs(attributes)
    rng = random.Random(config.seed)

    tasks: List[ChoiceTask] = []
    for t in range(config.n_tasks_per_respondent):
        alts: List[Dict[str, str]] = []
        for _ in range(config.n_alternatives):
            profile = {a.name: rng.choice(a.levels) for a in attrs}
            alts.append(profile)
        tasks.append(
            ChoiceTask(
                task_id=t,
                alternatives=alts,
                none_option=config.include_none_option,
            )
        )
    return tasks


# =========================
# Core: synthetic generation
# =========================


def _default_api_key() -> str:
    # Streamlit Cloud typically provides secrets via env vars.
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _generation_system_prompt() -> str:
    # Keep the role tight: choose like a plausible buyer; no extra text.
    return (
        "You are simulating a single realistic consumer completing a conjoint choice survey. "
        "Choose the option you would most likely buy in each task. "
        "Be consistent: prefer trade-offs that make sense for a real person. "
        "Return ONLY valid JSON that matches the schema exactly."
    )


def _generation_user_prompt(product: str, tasks: Sequence[ChoiceTask]) -> str:
    # Give the model concrete options to pick from, making it easy to respond in JSON.
    # One request per respondent keeps API calls low and is faster than per-task calls.
    tasks_payload = [
        {
            "task_id": t.task_id,
            "alternatives": t.alternatives,
            "none_option": t.none_option,
        }
        for t in tasks
    ]
    return (
        f"Product context:\n{product}\n\n"
        "Survey instructions:\n"
        "- For each task, pick exactly one option.\n"
        "- If none_option is true and all options are bad value, you MAY pick None.\n"
        "- Output JSON only.\n\n"
        "Tasks:\n"
        f"{json.dumps(tasks_payload, ensure_ascii=False)}\n\n"
        "JSON schema:\n"
        "{\n"
        '  "respondent_id": <int>,\n'
        '  "choices": [<int>, ...]  // one per task; -1 means None; otherwise index into alternatives\n'
        "}\n"
    )


async def _call_openai_json(
    *,
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: int = 45,
) -> Dict[str, Any]:
    """
    Minimal wrapper around OpenAI Responses API.
    Kept isolated so `app.py` can swap providers if desired.
    """
    from openai import AsyncOpenAI  # imported here to keep module import light

    client = AsyncOpenAI(api_key=api_key)
    resp = await client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Encourage structured output.
        temperature=0.4,
        timeout=timeout_s,
    )

    # The SDK returns text in output_text; keep it simple.
    text = getattr(resp, "output_text", None)
    if not text:
        # Fallback: try to reconstruct from outputs if needed.
        raise RuntimeError("Empty model response.")

    # Some models occasionally wrap JSON in whitespace; load strictly.
    return json.loads(text)


async def generate_synthetic_dataset(
    *,
    product: str,
    attributes: Sequence[Dict[str, Any]] | Sequence[AttributeSpec],
    config: StudyConfig = StudyConfig(),
    api_key: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[ChoiceTask]]:
    """
    Generates a full synthetic CBC dataset.

    Returns:
      - `df_long`: one row per (respondent, task, alternative), with a `chosen` flag
      - `tasks`: the design used (same tasks for all respondents in v1)
    """
    key = (api_key or _default_api_key()).strip()
    if not key:
        raise ValueError("Missing OPENAI_API_KEY (set env var or pass api_key).")

    attrs = normalize_attribute_specs(attributes)
    tasks = build_random_cbc_design(attributes=attrs, config=config)

    sem = asyncio.Semaphore(max(1, int(config.max_concurrent_requests)))
    rng = random.Random(config.seed)

    async def one_respondent(respondent_id: int) -> RespondentChoices:
        # Add light per-respondent variation without changing the tasks.
        persona_hint = rng.choice(
            [
                "Value-conscious, dislikes overpaying.",
                "Quality-focused, willing to pay more for clear benefits.",
                "Practical, prefers simple and reliable options.",
                "Feature-seeking, likes premium specs when justified.",
            ]
        )

        system_prompt = _generation_system_prompt() + f" Persona: {persona_hint}"
        user_prompt = _generation_user_prompt(product, tasks)

        last_err: Optional[Exception] = None
        for attempt in range(config.max_retries_per_respondent + 1):
            try:
                async with sem:
                    payload = await _call_openai_json(
                        api_key=key,
                        model_name=config.model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                payload["respondent_id"] = respondent_id
                parsed = RespondentChoices.model_validate(payload)
                if len(parsed.choices) != len(tasks):
                    raise ValueError("choices length mismatch")
                return parsed
            except (json.JSONDecodeError, ValidationError, ValueError, RuntimeError) as e:
                last_err = e
                # Tight repair attempt: ask for JSON again with stricter instruction.
                system_prompt = _generation_system_prompt() + " IMPORTANT: Output JSON only, no markdown."
                await asyncio.sleep(0.2 * (attempt + 1))
        raise RuntimeError(f"Failed to generate respondent {respondent_id}: {last_err}")

    started = time.time()
    respondents = await asyncio.gather(*(one_respondent(i) for i in range(config.n_respondents)))

    # Build long-format dataset.
    rows: List[Dict[str, Any]] = []
    for r in respondents:
        for task in tasks:
            chosen_idx = r.choices[task.task_id]
            for alt_idx, profile in enumerate(task.alternatives):
                row: Dict[str, Any] = {
                    "respondent_id": r.respondent_id,
                    "task_id": task.task_id,
                    "alt_id": alt_idx,
                    "chosen": 1 if chosen_idx == alt_idx else 0,
                }
                row.update({f"attr__{k}": v for k, v in profile.items()})
                rows.append(row)
            if task.none_option:
                rows.append(
                    {
                        "respondent_id": r.respondent_id,
                        "task_id": task.task_id,
                        "alt_id": -1,
                        "chosen": 1 if chosen_idx == -1 else 0,
                        **{f"attr__{a.name}": "__NONE__" for a in attrs},
                    }
                )

    df_long = pd.DataFrame(rows)
    df_long.attrs["generation_seconds"] = round(time.time() - started, 3)
    return df_long, tasks


# =========================
# Core: estimation & summaries
# =========================


def _one_hot_encode(
    df_long: pd.DataFrame, attribute_names: Sequence[str], drop_none_level: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    # One-hot encode each attribute column; keep stable column ordering.
    feature_cols: List[str] = []
    X_parts: List[pd.DataFrame] = []
    for name in attribute_names:
        col = f"attr__{name}"
        d = pd.get_dummies(df_long[col], prefix=col, dtype=float)
        if drop_none_level and f"{col}___NONE__" in d.columns:
            d = d.drop(columns=[f"{col}___NONE__"])
        X_parts.append(d)
        feature_cols.extend(list(d.columns))
    X = pd.concat(X_parts, axis=1)
    return X, feature_cols


def estimate_utilities(
    df_long: pd.DataFrame,
    *,
    attributes: Sequence[Dict[str, Any]] | Sequence[AttributeSpec],
    l2_c: float = 2.0,
) -> Dict[str, Any]:
    """
    Estimates aggregate part-worth utilities from the long-format dataset.

    Method (v1):
      - Treat each (task, alternative) row as an observation, with `chosen` as target.
      - Fit a regularized logistic regression on one-hot encoded attribute levels.

    This is fast and stable for interactive apps; you can swap in a true
    conditional logit / HB later without changing the app surface.
    """
    if LogisticRegression is None:
        raise ImportError("scikit-learn is required for estimation (add it to requirements.txt).")

    attrs = normalize_attribute_specs(attributes)
    attribute_names = [a.name for a in attrs]

    X, feature_cols = _one_hot_encode(df_long, attribute_names)
    y = df_long["chosen"].astype(int).values

    model = LogisticRegression(
        penalty="l2",
        C=float(l2_c),
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(X.values, y)

    coefs = model.coef_.reshape(-1)  # binary logistic => shape (1, n_features)
    utilities = dict(zip(feature_cols, coefs))

    # Attribute importance: range of level utilities within each attribute.
    importance: Dict[str, float] = {}
    for a in attrs:
        prefix = f"attr__{a.name}_"
        levels = [k for k in utilities.keys() if k.startswith(prefix)]
        if not levels:
            importance[a.name] = 0.0
            continue
        vals = [utilities[k] for k in levels]
        importance[a.name] = float(max(vals) - min(vals))

    # Normalize importances to percentages.
    total = sum(importance.values()) or 1.0
    importance_pct = {k: 100.0 * v / total for k, v in importance.items()}

    return {
        "utilities": utilities,
        "attribute_importance": importance_pct,
        "n_rows": int(df_long.shape[0]),
    }


def summarize_for_llm(
    *,
    product: str,
    attributes: Sequence[Dict[str, Any]] | Sequence[AttributeSpec],
    estimation: Dict[str, Any],
    top_k_levels: int = 12,
) -> Dict[str, Any]:
    """
    Creates a compact, structured payload suitable for an interpretation call.
    Keep it small and numeric; the model should explain, not recalculate.
    """
    attrs = normalize_attribute_specs(attributes)
    importance = estimation.get("attribute_importance", {})
    utilities: Dict[str, float] = estimation.get("utilities", {})

    # Top +/- levels by absolute utility.
    sorted_levels = sorted(utilities.items(), key=lambda kv: abs(kv[1]), reverse=True)[: int(top_k_levels)]

    return {
        "product": product,
        "attributes": [a.model_dump() for a in attrs],
        "attribute_importance_pct": importance,
        "top_level_effects": [{"feature": k, "utility": float(v)} for k, v in sorted_levels],
        "estimation_notes": {
            "method": "regularized_logistic_regression_on_long_format",
            "warning": "Directional: synthetic preferences; validate with real survey data for decisions.",
        },
    }

