from __future__ import annotations

import asyncio
import json
import os
import textwrap
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, ValidationError

from model import AttributeSpec, StudyConfig, estimate_utilities, generate_synthetic_dataset, summarize_for_llm


APP_TITLE = "AI-Driven Conjoint Analysis"


# =========================
# LLM helpers (attributes + interpretation)
# =========================


class AttributesProposal(BaseModel):
    attributes: List[AttributeSpec] = Field(..., min_length=2, max_length=8)


def _get_api_key() -> str:
    # Prefer Streamlit secrets if present.
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        key = str(st.secrets["OPENAI_API_KEY"]).strip()
        if key:
            return key
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _openai_client(api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def propose_attributes_with_ai(*, api_key: str, model_name: str, product: str) -> List[AttributeSpec]:
    """
    Uses a cheap model to propose a compact conjoint-ready attribute set
    with levels, including price points as one attribute.
    """
    client = _openai_client(api_key)

    system = (
        "You are a market research analyst designing a conjoint study.\n"
        "Return ONLY valid JSON that matches the schema exactly.\n"
        "Do not include markdown, commentary, or extra keys."
    )

    user = f"""
Product:
{product}

Task:
- Propose 4 to 6 attributes with 3 to 5 levels each.
- One attribute MUST be Price with realistic price points.
- Attribute names should be short.
- Levels must be mutually exclusive and easy to understand.
- Avoid repeating the same idea across attributes.

JSON schema:
{{
  "attributes": [
    {{"name": "Price", "levels": ["...", "...", "..."]}},
    {{"name": "<attribute>", "levels": ["...", "..."]}}
  ]
}}
""".strip()

    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
    )

    text = getattr(resp, "output_text", None)
    if not text:
        raise RuntimeError("Empty response while generating attributes.")

    payload = json.loads(text)
    parsed = AttributesProposal.model_validate(payload)
    return parsed.attributes


def interpret_results_with_ai(*, api_key: str, model_name: str, summary_payload: Dict[str, Any]) -> str:
    """
    Produces a narrative interpretation from structured numeric outputs.
    """
    client = _openai_client(api_key)
    system = (
        "You are a conjoint analyst.\n"
        "Explain the results clearly for a non-technical product manager.\n"
        "Do not fabricate numbers; only reference the provided JSON.\n"
        "Give concrete next steps and what to validate with a real survey."
    )
    user = "Here are the computed conjoint results in JSON:\n\n" + json.dumps(summary_payload, indent=2)
    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.5,
    )
    text = getattr(resp, "output_text", None)
    if not text:
        raise RuntimeError("Empty response while interpreting results.")
    return text.strip()


# =========================
# UI helpers
# =========================


def _parse_manual_attributes(raw: str) -> List[AttributeSpec]:
    """
    Parses a simple, human-friendly format:
      Attribute: level1, level2, level3
    """
    attrs: List[AttributeSpec] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        name, levels_str = line.split(":", 1)
        levels = [x.strip() for x in levels_str.split(",") if x.strip()]
        if name.strip() and len(levels) >= 2:
            attrs.append(AttributeSpec(name=name.strip(), levels=levels))
    if len(attrs) < 2:
        raise ValueError("Please provide at least 2 attributes (each with 2+ levels).")
    return attrs


def _attributes_to_text(attrs: Sequence[AttributeSpec]) -> str:
    lines = []
    for a in attrs:
        lines.append(f"{a.name}: {', '.join(a.levels)}")
    return "\n".join(lines)


def _run_async(coro):
    """
    Streamlit runs in a sync context. This helper executes async code reliably.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run(coro)
    except RuntimeError:
        pass
    return asyncio.run(coro)


def _importance_df(importance_pct: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(
        [{"attribute": k, "importance_pct": float(v)} for k, v in importance_pct.items()]
    ).sort_values("importance_pct", ascending=False)
    return df


# =========================
# App
# =========================


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

st.caption(
    "Direction-first conjoint using AI-generated synthetic preference data. "
    "Great for early exploration; validate with real respondents for decisions."
)

with st.sidebar:
    st.subheader("Settings")
    model_name = st.text_input("OpenAI model (cheap)", value="gpt-4.1-mini")

    st.markdown("**Speed / quality trade-off**")
    n_respondents = st.slider("Synthetic respondents", min_value=15, max_value=120, value=40, step=5)
    n_tasks = st.slider("Tasks per respondent", min_value=5, max_value=14, value=8, step=1)
    max_conc = st.slider("Parallel requests", min_value=2, max_value=20, value=8, step=1)
    include_none = st.toggle("Include 'None' option", value=True)

    st.divider()
    st.markdown("**AI strategy**")
    auto_strategy = st.toggle("Auto-generate strategy after results", value=True)
    st.caption("Uses your `OPENAI_API_KEY` from Streamlit secrets / environment variables.")


product = st.text_area(
    "Describe the product",
    placeholder="Example: A meal kit subscription for busy professionals in Seattle. Emphasize healthy options and delivery flexibility.",
    height=120,
)

tab_ai, tab_manual = st.tabs(["AI-assisted setup (recommended)", "Manual setup"])

attrs: Optional[List[AttributeSpec]] = None
attrs_source: str = "ai"

with tab_ai:
    st.markdown("Generate attributes and price points automatically from your product description.")
    if st.button("Generate attributes", type="primary", disabled=not product.strip()):
        key = _get_api_key()
        if not key:
            st.error("Missing API key. Add `OPENAI_API_KEY` as a Streamlit secret or environment variable.")
        else:
            with st.spinner("Generating attributes…"):
                try:
                    generated = propose_attributes_with_ai(api_key=key, model_name=model_name, product=product.strip())
                    st.session_state["attrs_text"] = _attributes_to_text(generated)
                except Exception as e:
                    st.error(f"Failed to generate attributes: {e}")

    attrs_text = st.text_area(
        "Attributes (editable)",
        value=st.session_state.get(
            "attrs_text",
            "Price: $9, $12, $15\n"
            "Delivery window: 1-2 days, 3-4 days, 5-7 days\n"
            "Meal variety: 10 meals/week, 20 meals/week, 30 meals/week\n"
            "Diet options: None, Vegetarian, Vegan\n"
            "Packaging: Standard, Recyclable, Compostable",
        ),
        height=180,
        help="Format: Attribute: level1, level2, level3",
    )

with tab_manual:
    st.markdown("Type attributes directly. Use the same format as above.")
    manual_text = st.text_area(
        "Attributes",
        value="",
        height=180,
        placeholder="Price: $9, $12, $15\nBrand: New, Known, Premium\nBattery: 10h, 15h, 20h",
        help="Format: Attribute: level1, level2, level3",
    )


def _current_attributes() -> Tuple[List[AttributeSpec], str]:
    if manual_text.strip():
        return _parse_manual_attributes(manual_text), "manual"
    return _parse_manual_attributes(attrs_text), "ai"


st.divider()
run_col1, run_col2 = st.columns([1, 1])

with run_col1:
    run = st.button("Run conjoint", type="primary", disabled=not product.strip())

with run_col2:
    st.caption("Tip: start small (40×8) for speed; increase respondents for smoother importances.")


if run:
    key = _get_api_key()
    if not key:
        st.error("Missing API key. Add `OPENAI_API_KEY` as a Streamlit secret or environment variable.")
        st.stop()

    try:
        attrs, attrs_source = _current_attributes()
    except Exception as e:
        st.error(str(e))
        st.stop()

    config = StudyConfig(
        n_respondents=int(n_respondents),
        n_tasks_per_respondent=int(n_tasks),
        n_alternatives=3,
        include_none_option=bool(include_none),
        max_concurrent_requests=int(max_conc),
        model_name=str(model_name).strip() or "gpt-4.1-mini",
    )

    st.subheader("1) Synthetic data generation")
    with st.spinner("Generating synthetic survey responses…"):
        try:
            df_long, tasks = _run_async(
                generate_synthetic_dataset(
                    product=product.strip(),
                    attributes=attrs,
                    config=config,
                    api_key=key,
                )
            )
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

    gen_s = float(df_long.attrs.get("generation_seconds", 0.0))
    st.success(f"Generated {len(df_long):,} rows in ~{gen_s:.1f}s")
    with st.expander("Preview synthetic dataset"):
        st.dataframe(df_long.head(50), use_container_width=True)

    st.subheader("2) Estimate utilities + importance")
    with st.spinner("Estimating…"):
        try:
            est = estimate_utilities(df_long, attributes=attrs)
        except Exception as e:
            st.error(f"Estimation failed: {e}")
            st.stop()

    imp_df = _importance_df(est["attribute_importance"])
    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Attribute importance (%)**")
        st.bar_chart(imp_df.set_index("attribute")["importance_pct"])
        st.dataframe(imp_df, use_container_width=True)

    with right:
        st.markdown("**Top level effects (directional)**")
        util_items = list(est["utilities"].items())
        util_df = (
            pd.DataFrame(util_items, columns=["feature", "utility"])
            .assign(abs_utility=lambda d: d["utility"].abs())
            .sort_values("abs_utility", ascending=False)
            .head(20)
            .drop(columns=["abs_utility"])
        )
        st.dataframe(util_df, use_container_width=True)

    st.subheader("3) Export")
    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        st.download_button(
            "Download synthetic data (CSV)",
            data=df_long.to_csv(index=False).encode("utf-8"),
            file_name="synthetic_conjoint_long.csv",
            mime="text/csv",
        )
    with export_col2:
        st.download_button(
            "Download estimation (JSON)",
            data=json.dumps(est, indent=2).encode("utf-8"),
            file_name="conjoint_estimation.json",
            mime="application/json",
        )
    with export_col3:
        st.download_button(
            "Download attributes (JSON)",
            data=json.dumps([a.model_dump() for a in attrs], indent=2).encode("utf-8"),
            file_name="attributes.json",
            mime="application/json",
        )

    st.subheader("4) AI interpretation (optional)")
    summary_payload = summarize_for_llm(product=product.strip(), attributes=attrs, estimation=est)
    with st.expander("Show payload sent for interpretation"):
        st.code(json.dumps(summary_payload, indent=2), language="json")

    def _strategy_cache_key() -> str:
        # Stable key so Streamlit re-runs still show the strategy.
        return json.dumps(summary_payload, sort_keys=True)

    cache_key = _strategy_cache_key()
    if auto_strategy:
        if st.session_state.get("strategy_cache_key") != cache_key:
            st.session_state["strategy_cache_key"] = cache_key
            st.session_state["strategy_text"] = None

        if not st.session_state.get("strategy_text"):
            with st.spinner("Generating strategy…"):
                try:
                    st.session_state["strategy_text"] = interpret_results_with_ai(
                        api_key=key,
                        model_name=str(model_name).strip() or "gpt-4.1-mini",
                        summary_payload=summary_payload,
                    )
                except Exception as e:
                    st.error(f"Strategy generation failed: {e}")
                    st.stop()

        st.markdown(st.session_state["strategy_text"])
    else:
        st.info("Enable “Auto-generate strategy after results” in the sidebar to generate strategy text.")

