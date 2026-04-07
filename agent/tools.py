"""Phase 4 LangChain tools: three dual-model tools with Role-Play LLM arbitration.

Architecture
────────────
Each of the three domain tools (longitudinal / safety_molecule / ehr_centric)
runs its two complementary ML models, then classifies every candidate drug into:

  AUTO_ACCEPT  — both models score >= accept_threshold  → passes directly
  AUTO_REJECT  — both models score <  reject_threshold  → silently dropped
  UNCERTAIN    — anything else                          → sent to a domain-specific
                                                          Role-Play LLM

A summarisation tool (p4_summarize_tool) then aggregates all three outputs,
computes vote counts, DDI analysis, and ATC class distribution — producing the
structured brief reviewed by the consultant-physician ReAct agent.
"""
from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import Optional

import yaml
from langchain_core.tools import tool

from .data_loader import (
    get_data_bundle,
    get_patient_visits,
    predict_two_models,
)
from .ddi     import check_ddi, ddi_check_tool  # noqa: F401 (re-exported)
from .codebook import drug_label, atc_desc, diag_label, proc_label
from .primekg  import get_primekg_context, get_drug_summary, is_available as primekg_available
from .config   import get_config

# ──────────────────────────── uncertainty config ──────────────────────────────

_UNCERTAIN_YAML = Path(__file__).parent.parent / "uncertain_config.yaml"


@functools.lru_cache(maxsize=1)
def _load_thresholds() -> dict:
    with open(_UNCERTAIN_YAML) as f:
        raw = yaml.safe_load(f)
    return {k: v for k, v in raw["models"].items()}


def reload_thresholds() -> dict:
    """Force-reload thresholds from YAML (call after editing uncertain_config.yaml)."""
    _load_thresholds.cache_clear()
    return _load_thresholds()


def _zone(score: float, model_name: str) -> str:
    t = _load_thresholds().get(model_name, {"accept_threshold": 0.5, "reject_threshold": 0.2})
    if score >= t["accept_threshold"]:
        return "ACCEPT"
    if score < t["reject_threshold"]:
        return "REJECT"
    return "UNCERTAIN"


def _combined_zone(za: str, zb: str) -> str:
    if za == "ACCEPT" and zb == "ACCEPT":
        return "AUTO_ACCEPT"
    if za == "REJECT" and zb == "REJECT":
        return "AUTO_REJECT"
    return "UNCERTAIN"


# ──────────────────────────── Role-Play LLM personas ─────────────────────────

_PERSONA_LONGITUDINAL = """\
You are a Temporal Historian — a specialist in longitudinal patient care and
multi-visit treatment continuity.

Your clinical focus:
- Visit history and medication continuity across admissions
- Temporal medication patterns (initiation, continuation, discontinuation)
- Prior treatment responses and treatment-history evidence

When evaluating uncertain drug candidates, you weight:
1. Whether the drug was prescribed in earlier admissions (continuity signal)
2. Whether the temporal pattern supports introduction or re-introduction now
3. Whether the longitudinal evidence is consistent across both models"""

_PERSONA_SAFETY_MOLECULE = """\
You are a Chemical Safety Specialist — an expert in drug molecular structure,
substructure pharmacology, and drug-drug interaction (DDI) risk.

Your clinical focus:
- Drug-drug interaction risk at the molecular level
- Substructure clashes and overlapping pharmacophores
- Molecular safety profiles relative to the patient's existing regimen

When evaluating uncertain drug candidates, you weight:
1. Known or predicted DDI risk with other drugs in the candidate set
2. Substructure overlap that may cause additive or antagonistic effects
3. Molecular similarity to safer alternatives"""

_PERSONA_EHR_CENTRIC = """\
You are an EHR Integration Specialist — an expert in holistic electronic health
record analysis, disease progression, and multi-modal clinical evidence.

Your clinical focus:
- Disease progression signals from visit-to-visit EHR deltas
- Alignment of clinical codes, drug descriptions, and therapeutic class
- Evidence from procedures, diagnoses, and structured EHR data

When evaluating uncertain drug candidates, you weight:
1. Whether the drug matches the patient's current disease stage and progression
2. Alignment between the drug's therapeutic class and the active diagnosis profile
3. Whether EHR-derived signals (procedures, comorbidities) support this drug"""


# ──────────────────────────── patient context builder ────────────────────────

def _build_patient_context(visits: list) -> tuple[str, list[str]]:
    """Return (context_text, current_diag_names) for the Role-Play LLM."""
    data_bundle, _, med_voc = get_data_bundle()
    voc = data_bundle.get("voc", {})
    diag_idx2word = getattr(voc.get("diag_voc"), "idx2word", {}) if voc.get("diag_voc") else {}
    proc_idx2word = getattr(voc.get("pro_voc"),  "idx2word", {}) if voc.get("pro_voc")  else {}

    lines = [f"Patient history ({len(visits)} visit(s)):"]
    current_diag_names: list[str] = []

    for i, adm in enumerate(visits):
        diag_codes = [diag_idx2word.get(d, str(d)) for d in adm[0][:5]]
        proc_codes = [proc_idx2word.get(p, str(p)) for p in adm[1][:3]]
        diag_str   = "; ".join(diag_label(c) for c in diag_codes) or "none"
        proc_str   = "; ".join(proc_label(c) for c in proc_codes) or "none"

        if i < len(visits) - 1:
            med_names = [med_voc.get(m, f"drug_{m}") for m in adm[2][:6]]
            med_str   = ", ".join(drug_label(n) for n in med_names) or "none"
            lines.append(
                f"\nVisit {i+1}:\n"
                f"  Diagnoses:  {diag_str}\n"
                f"  Procedures: {proc_str}\n"
                f"  Medications: {med_str}"
            )
        else:
            current_diag_names = [diag_label(c) for c in diag_codes]
            lines.append(
                f"\nCurrent visit (medications to determine):\n"
                f"  Diagnoses:  {diag_str}\n"
                f"  Procedures: {proc_str}"
            )

    return "\n".join(lines), current_diag_names


# ──────────────────────────── Role-Play LLM call ─────────────────────────────

def _call_roleplay_llm(
    persona: str,
    uncertain_items: list,       # [(drug, sa, sb, model_a, model_b, za, zb)]
    patient_context: str,
    llm_model: str,
    patient_diag_names: Optional[list[str]] = None,
) -> dict[str, str]:             # {drug: "ACCEPT" | "REJECT"}
    if not uncertain_items:
        return {}

    from langchain_openai import ChatOpenAI

    uncertain_codes = [drug for drug, *_ in uncertain_items]
    model_a = uncertain_items[0][3]
    model_b = uncertain_items[0][4]

    # Optional PrimeKG context
    kg_context = ""
    if primekg_available():
        kg_context = get_primekg_context(uncertain_codes, patient_diag_names)
        if kg_context:
            print(f"  [roleplay] PrimeKG context for {len(uncertain_codes)} drugs.", flush=True)

    def _kg_hint(drug: str) -> str:
        s = get_drug_summary(drug)
        return f"    KG: {s}" if s else ""

    header = (
        f"{'#':<4} {'Drug (ATC3 code)':<44} "
        f"{model_a:>10} {model_b:>10}  Zone_A    Zone_B"
    )
    rows = []
    for idx, (drug, sa, sb, ma, mb, za, zb) in enumerate(uncertain_items, 1):
        row  = (f"{idx:<4} {drug_label(drug):<44} "
                f"{sa:>10.3f} {sb:>10.3f}  {za:<9} {zb}")
        hint = _kg_hint(drug)
        rows.append(row + ("\n" + hint if hint else ""))
    table = "\n".join([header, "─" * 90] + rows)

    kg_section = f"\n{kg_context}\n" if kg_context else ""

    user_msg = (
        f"{patient_context}\n"
        f"{kg_section}"
        "── Uncertain Drug Candidates ──────────────────────────────────────────────────\n"
        f"{table}\n\n"
        "For EACH numbered drug above, respond on ONE line:\n"
        "  <number>. ACCEPT | <brief clinical reason>\n"
        "  <number>. REJECT | <brief clinical reason>\n\n"
        "Respond for ALL drugs. Use only ACCEPT or REJECT."
    )

    llm = ChatOpenAI(
        model=llm_model,
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    response = llm.invoke([
        {"role": "system", "content": persona},
        {"role": "user",   "content": user_msg},
    ])

    decisions: dict[str, str] = {}
    for line in response.content.splitlines():
        m = re.match(r"^(\d+)\.\s+(ACCEPT|REJECT)", line.strip(), re.IGNORECASE)
        if m:
            i_zero   = int(m.group(1)) - 1
            decision = m.group(2).upper()
            if 0 <= i_zero < len(uncertain_items):
                decisions[uncertain_items[i_zero][0]] = decision

    for drug, *_ in uncertain_items:
        decisions.setdefault(drug, "REJECT")   # conservative default

    return decisions


# ──────────────────────────── prior meds helper ───────────────────────────────

def _prior_meds(visits: list) -> tuple[set, set]:
    """Return (ever_prescribed, most_recent_prior_visit_meds)."""
    _, _, med_voc = get_data_bundle()
    if len(visits) < 2:
        return set(), set()
    ever: set = set()
    for adm in visits[:-1]:
        for idx in adm[2]:
            name = med_voc.get(idx)
            if name:
                ever.add(name)
    recent: set = {med_voc[idx] for idx in visits[-2][2] if med_voc.get(idx)}
    return ever, recent


# ──────────────────────────── per-patient accumulator ─────────────────────────

# {patient_id: {tool_name: {drug: mean_score}}}
_phase4_results: dict[str, dict[str, dict[str, float]]] = {}


# ──────────────────────────── shared tool implementation ──────────────────────

def _phase4_impl(
    tool_name: str,
    model_a: str,
    model_b: str,
    persona: str,
    patient_id: str,
) -> str:
    cfg    = get_config()
    visits = get_patient_visits(patient_id)
    ever, recent = _prior_meds(visits)

    # 1. Run both models at threshold=0.0 to get full score distributions
    print(f"  [phase4/{tool_name}] running {model_a} + {model_b}...", flush=True)
    per_model = predict_two_models(model_a, model_b, visits)

    scores_a: dict = per_model.get(model_a, {}).get("scores", {})
    scores_b: dict = per_model.get(model_b, {}).get("scores", {})
    all_drugs = set(scores_a) | set(scores_b)

    # 2. Classify each drug into AUTO_ACCEPT / AUTO_REJECT / UNCERTAIN
    auto_accept: list = []
    auto_reject: list = []
    uncertain:   list = []   # (drug, sa, sb, model_a, model_b, zone_a, zone_b)

    for drug in sorted(all_drugs):
        sa = scores_a.get(drug, 0.0)
        sb = scores_b.get(drug, 0.0)
        za = _zone(sa, model_a)
        zb = _zone(sb, model_b)
        cz = _combined_zone(za, zb)

        if cz == "AUTO_ACCEPT":
            auto_accept.append((drug, sa, sb))
        elif cz == "AUTO_REJECT":
            auto_reject.append(drug)
        else:
            uncertain.append((drug, sa, sb, model_a, model_b, za, zb))

    print(
        f"  [phase4/{tool_name}] auto_accept={len(auto_accept)} "
        f"uncertain={len(uncertain)} auto_reject={len(auto_reject)}",
        flush=True,
    )

    # 3. Role-Play LLM arbitrates uncertain drugs
    llm_decisions: dict[str, str] = {}
    if uncertain:
        print(
            f"  [phase4/{tool_name}] calling Role-Play LLM ({cfg.llm_model}) "
            f"for {len(uncertain)} uncertain drugs...",
            flush=True,
        )
        patient_ctx, diag_names = _build_patient_context(visits)
        llm_decisions = _call_roleplay_llm(
            persona, uncertain, patient_ctx, cfg.llm_model,
            patient_diag_names=diag_names,
        )

    llm_accepted = [drug for drug, *_ in uncertain if llm_decisions.get(drug) == "ACCEPT"]
    llm_rejected = [drug for drug, *_ in uncertain if llm_decisions.get(drug) != "ACCEPT"]

    print(
        f"  [phase4/{tool_name}] LLM accepted={len(llm_accepted)} "
        f"rejected={len(llm_rejected)}",
        flush=True,
    )

    # 4. Build final accepted set (mean score)
    final: dict[str, float] = {}
    for drug, sa, sb in auto_accept:
        final[drug] = round((sa + sb) / 2, 4)
    for drug, sa, sb, *_ in uncertain:
        if llm_decisions.get(drug) == "ACCEPT":
            final[drug] = round((sa + sb) / 2, 4)

    predicted = sorted(final, key=lambda d: -final[d])

    # 5. Store for summarisation
    _phase4_results.setdefault(patient_id, {})[tool_name] = final

    # 6. Format output
    thresholds = _load_thresholds()
    ta = thresholds.get(model_a, {})
    tb = thresholds.get(model_b, {})

    lines = [
        "─" * 72,
        f"Phase 4 — {tool_name.replace('_', ' ').title()} — patient {patient_id}",
        f"Models: {model_a} + {model_b}",
        f"Uncertainty windows: "
        f"{model_a} [{ta.get('reject_threshold','?')}, {ta.get('accept_threshold','?')})  "
        f"{model_b} [{tb.get('reject_threshold','?')}, {tb.get('accept_threshold','?')})",
        "",
        "Zone summary:",
        f"  AUTO_ACCEPT (both confident):     {len(auto_accept)}",
        f"  UNCERTAIN   (sent to LLM):        {len(uncertain)}",
        f"  AUTO_REJECT (both reject):        {len(auto_reject)}",
        f"  LLM accepted from uncertain:      {len(llm_accepted)}",
        f"  LLM rejected from uncertain:      {len(llm_rejected)}",
        f"  Final accepted drugs:             {len(predicted)}",
        "",
    ]

    if predicted:
        ca = f"{model_a[:7]:<7}"
        cb = f"{model_b[:7]:<7}"
        lines.append(f"{'Drug':<44} {'Mean':>6}  {ca}  {cb}  {'Source':12}")
        lines.append("─" * 90)
        for drug in predicted:
            sa = scores_a.get(drug, 0.0)
            sb = scores_b.get(drug, 0.0)
            za = _zone(sa, model_a)
            zb = _zone(sb, model_b)
            source = "AUTO_ACCEPT" if (za == "ACCEPT" and zb == "ACCEPT") else "LLM_ACCEPT"
            lines.append(
                f"  {drug_label(drug):<42} {final[drug]:>6.3f}  "
                f"{sa:>7.3f}  {sb:>7.3f}  {source:<12}"
            )
        lines.append("")

    if uncertain:
        lines.append("Role-Play LLM arbitration:")
        for drug, sa, sb, ma, mb, za, zb in uncertain:
            d = llm_decisions.get(drug, "REJECT")
            lines.append(
                f"  {drug_label(drug):<42}  {ma}={sa:.3f}({za})  "
                f"{mb}={sb:.3f}({zb})  -> {d}"
            )
        lines.append("")

    lines.append("Prior medication context:")
    if not ever:
        lines.append("  (no prior visit medications — first admission)")
    else:
        lines.append(
            f"  Most-recent-visit meds ({len(recent)}): "
            + (", ".join(sorted(recent)) if recent else "none")
        )
        ever_only = ever - recent
        if ever_only:
            lines.append(
                f"  Earlier visits only ({len(ever_only)}): "
                + ", ".join(sorted(ever_only))
            )
    lines.append("")
    lines.append("─" * 72)
    return "\n".join(lines)


# ──────────────────────────── Phase 4 LangChain tools ────────────────────────

@tool
def p4_longitudinal_tool(patient_id: str) -> str:
    """Analyse multi-visit patient history using RETAIN and GAMENet (Phase 4).

    Runs RETAIN (visit-level attention) and GAMENet (graph memory bank) in
    parallel.  Per-drug scores are classified into three zones:
      - AUTO_ACCEPT : both models confident  → drug passes directly
      - AUTO_REJECT : both models reject     → drug dropped
      - UNCERTAIN   : mixed signals          → sent to a Temporal Historian
                                               Role-Play LLM for arbitration

    Args:
        patient_id: patient identifier

    Returns:
        Enriched drug candidates with zone breakdown and LLM arbitration log.
    """
    return _phase4_impl(
        "longitudinal", "retain", "gamenet", _PERSONA_LONGITUDINAL, patient_id
    )


@tool
def p4_safety_molecule_tool(patient_id: str) -> str:
    """Recommend safe drugs via molecular structure using SafeDrug and MoleRec (Phase 4).

    Uncertain drugs are arbitrated by a Chemical Safety Specialist Role-Play
    LLM that weighs DDI risk, substructure clashes, and molecular safety.

    Args:
        patient_id: patient identifier

    Returns:
        Molecularly-grounded drug candidates with zone breakdown and LLM log.
    """
    return _phase4_impl(
        "safety_molecule", "safedrug", "molerec", _PERSONA_SAFETY_MOLECULE, patient_id
    )


@tool
def p4_ehr_centric_tool(patient_id: str) -> str:
    """Recommend drugs via full EHR integration using DEPOT and MedAlign (Phase 4).

    Uncertain drugs are arbitrated by an EHR Integration Specialist Role-Play
    LLM that weighs disease progression, modality evidence, and therapy signals.

    Args:
        patient_id: patient identifier

    Returns:
        EHR-aligned drug candidates with zone breakdown and LLM arbitration log.
    """
    return _phase4_impl(
        "ehr_centric", "depot", "medalign", _PERSONA_EHR_CENTRIC, patient_id
    )


@tool
def p4_summarize_tool(patient_id: str) -> str:
    """Summarise all three Phase 4 tool results for the final consultant-physician.

    Aggregates accepted drug sets from longitudinal, safety_molecule, and
    ehr_centric tools, computing:
      - Vote counts (how many tools accepted each drug)
      - Mean confidence score across accepting tools
      - DDI analysis for the full candidate set
      - ATC class distribution
      - Continuation / new / re-introduction status

    Call AFTER p4_longitudinal_tool, p4_safety_molecule_tool, p4_ehr_centric_tool.

    Args:
        patient_id: patient identifier

    Returns:
        Structured multi-tool summary with vote counts, DDI, and ATC breakdown.
    """
    tool_scores = _phase4_results.get(patient_id, {})
    if not tool_scores:
        return (
            "ERROR: No Phase 4 results found for this patient. "
            "Call p4_longitudinal_tool, p4_safety_molecule_tool, and "
            "p4_ehr_centric_tool first."
        )

    visits = get_patient_visits(patient_id)
    ever, recent = _prior_meds(visits)

    tool_order = ["longitudinal", "safety_molecule", "ehr_centric"]
    n_tools = len([t for t in tool_order if t in tool_scores])

    # Vote count and mean score per drug
    all_drugs: set = set()
    for scores in tool_scores.values():
        all_drugs.update(scores)

    vote_count: dict[str, int]   = {}
    mean_score: dict[str, float] = {}
    for drug in all_drugs:
        votes = sum(1 for t in tool_order if drug in tool_scores.get(t, {}))
        vals  = [tool_scores[t][drug] for t in tool_order if drug in tool_scores.get(t, {})]
        vote_count[drug] = votes
        mean_score[drug] = round(sum(vals) / len(vals), 4) if vals else 0.0

    # Tool-level zoning: >=2 votes → AUTO_ACCEPT, 1 vote → UNCERTAIN
    auto_accepted = sorted(
        [d for d in all_drugs if vote_count[d] >= 2],
        key=lambda d: (-vote_count[d], -mean_score[d]),
    )
    uncertain = sorted(
        [d for d in all_drugs if vote_count[d] == 1],
        key=lambda d: -mean_score[d],
    )
    predicted = auto_accepted

    # DDI check on auto-accepted drugs
    ddi = check_ddi(predicted)

    lines = [
        "=" * 72,
        f"Phase 4 Summary — patient {patient_id}",
        f"Tools with results: {', '.join(t for t in tool_order if t in tool_scores)}",
        "",
        "Tool-level zone summary:",
        f"  AUTO_ACCEPT (>=2 tools agreed): {len(auto_accepted)} drugs",
        f"  UNCERTAIN   (1 tool only):      {len(uncertain)} drugs",
        "",
        f"{'Drug':<44} {'Votes':>5}  {'Mean':>6}  {'Long':>6}  {'Sfty':>6}  {'EHR':>6}  Zone",
        "─" * 96,
    ]

    for drug in auto_accepted:
        l = tool_scores.get("longitudinal",    {}).get(drug, 0.0)
        s = tool_scores.get("safety_molecule", {}).get(drug, 0.0)
        e = tool_scores.get("ehr_centric",     {}).get(drug, 0.0)
        lines.append(
            f"  {drug_label(drug):<42} {vote_count[drug]:>5}  {mean_score[drug]:>6.3f}"
            f"  {l:>6.3f}  {s:>6.3f}  {e:>6.3f}  AUTO_ACCEPT"
        )

    if uncertain:
        lines.append("")
        for drug in uncertain:
            l = tool_scores.get("longitudinal",    {}).get(drug, 0.0)
            s = tool_scores.get("safety_molecule", {}).get(drug, 0.0)
            e = tool_scores.get("ehr_centric",     {}).get(drug, 0.0)
            lines.append(
                f"  {drug_label(drug):<42} {vote_count[drug]:>5}  {mean_score[drug]:>6.3f}"
                f"  {l:>6.3f}  {s:>6.3f}  {e:>6.3f}  UNCERTAIN"
            )

    lines.append("")

    # Breakdown by vote count
    unanimous = [d for d in auto_accepted if vote_count[d] == n_tools]
    majority  = [d for d in auto_accepted if vote_count[d] < n_tools]
    lines.append(f"Unanimous ({n_tools}/{n_tools} tools): {len(unanimous)} drugs — AUTO_ACCEPT")
    if unanimous:
        lines.append("  " + ", ".join(drug_label(d) for d in unanimous))
    lines.append(f"Majority  (2/{n_tools} tools): {len(majority)} drugs — AUTO_ACCEPT")
    if majority:
        lines.append("  " + ", ".join(drug_label(d) for d in majority))
    lines.append(f"Uncertain (1/{n_tools} tools): {len(uncertain)} drugs — agent decides")
    if uncertain:
        lines.append("  " + ", ".join(
            f"{drug_label(d)} ({mean_score[d]:.3f})" for d in uncertain))
    lines.append("")

    # ATC class distribution (auto-accepted)
    from collections import defaultdict
    atc_classes: dict[str, list] = defaultdict(list)
    for drug in predicted:
        cls = drug[0].upper() if drug else "?"
        atc_classes[cls].append(drug)

    lines.append("ATC class distribution (AUTO_ACCEPT drugs):")
    for cls in sorted(atc_classes):
        drugs_str = ", ".join(drug_label(d) for d in atc_classes[cls])
        lines.append(f"  [{cls}] {atc_desc(cls)}: {drugs_str}")
    lines.append("")

    # DDI summary
    lines.append(f"DDI summary ({len(predicted)} AUTO_ACCEPT drugs):")
    if ddi["is_safe"]:
        lines.append("  No drug-drug interactions detected.")
    else:
        lines.append(
            f"  {ddi['n_interactions']} interaction(s) "
            f"(DDI rate {ddi['ddi_rate']:.1%}):"
        )
        for pair in ddi["ddi_pairs"][:8]:
            lines.append(f"    - {pair[0]} x {pair[1]}")
        removed = [d for d in predicted if d not in ddi["safe_drugs"]]
        if removed:
            lines.append(f"  Flagged by DDI checker: {', '.join(removed)}")
        lines.append(
            f"  Safe subset ({len(ddi['safe_drugs'])}): "
            + ", ".join(ddi["safe_drugs"])
        )
    lines.append("")

    # Prior meds context
    lines.append("Prior medication context:")
    if not ever:
        lines.append("  (no prior visit medications — first admission)")
    else:
        lines.append(
            f"  Most-recent-visit meds ({len(recent)}): "
            + (", ".join(sorted(recent)) if recent else "none")
        )
        dropped = recent - set(predicted)
        if dropped:
            lines.append(
                f"  Dropped from prior regimen ({len(dropped)}): "
                + ", ".join(drug_label(d) for d in sorted(dropped))
            )
    lines.append("")
    lines.append("AUTO_ACCEPT ATC3 codes (include directly): " + ", ".join(predicted))
    if uncertain:
        lines.append("UNCERTAIN ATC3 codes (agent decides):     " + ", ".join(uncertain))
    lines.append("=" * 72)

    # Clear accumulator so re-runs don't stale-contaminate
    _phase4_results.pop(patient_id, None)

    return "\n".join(lines)


# ──────────────────────────── exported tool list ──────────────────────────────

PHASE4_TOOLS = [
    p4_longitudinal_tool,
    p4_safety_molecule_tool,
    p4_ehr_centric_tool,
    p4_summarize_tool,
]
