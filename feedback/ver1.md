# Pipeline Revision Plan — Version 1

---

## Current Pipeline — Critical Weaknesses

**1. Uncalibrated thresholds**
All models use the same fixed `accept=0.60 / reject=0.40` despite having very different score distributions. RETAIN and GAMENet produce very different sigmoid output ranges than SafeDrug or DEPOT. The thresholds are not principled — there is no validation-set evidence that 0.60 means "confident" for any specific model.

**2. Vote counting discards confidence**
The summarizer treats a drug accepted with mean score 0.95 the same as one accepted with 0.62 — both count as one vote. The aggregation loses the information that was just computed.

**3. Domain-isolated arbitration**
The Temporal Historian, Chemical Safety Specialist, and EHR Integration Specialist each arbitrate in complete isolation. A drug that one specialist rejects for molecular reasons is unknown to the specialist reasoning about temporal patterns. Arbitration is locally informed but globally blind.

**4. DDI is purely post-hoc**
DDI is checked after the recommendation is formed and only flagged to the consultant. A drug that creates a DDI pair should ideally influence the arbitration step, not just appear in a warning the agent may or may not act on.

**5. Sequential tool execution**
The three domain tools run one after another. There is no architectural dependency between them — they could run in parallel.

**6. Unconstrained consultant output**
The final recommendation is extracted by regex from free text. The consultant physician can hallucinate ATC codes, omit drugs, or produce malformed output, and the pipeline has no mechanism to detect or correct this.

**7. Global mutable state**
`_phase4_results` in `agent/tools.py:255` is a process-level dict. It is not designed for concurrent patient evaluation and is cleared only at summarize time — a crash between tool calls leaves stale data.

---

## Proposed Revised Architecture

```
Patient EHR
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│              PARALLEL DOMAIN TOOLS                       │
│  (run simultaneously — no sequential dependency)         │
│                                                          │
│  [Longitudinal]  [Safety Molecule]  [EHR-Centric]        │
│                                                          │
│  Each tool:                                              │
│  1. Calibrated model scores (Platt scaling)              │
│  2. Adaptive uncertainty windows (per-model, data-       │
│     driven from validation set score distribution)       │
│  3. Role-Play LLM arbitration with DDI context           │
│     injected per uncertain drug                          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │  CROSS-DOMAIN SYNTHESIS  │
        │                          │
        │  Confidence-weighted     │
        │  aggregation             │
        │  (replace vote counting) │
        │                          │
        │  Split-vote drugs →      │
        │  Cross-domain arbitration│
        │  LLM (sees all 3 domain  │
        │  signals simultaneously) │
        └──────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │   CONSULTANT PHYSICIAN   │
        │   (LangGraph ReAct)      │
        │                          │
        │   Structured output →    │
        │   constrained to valid   │
        │   ATC vocabulary         │
        └──────────────────────────┘
```

---

## Improvements — Prioritized Plan

### Priority 1 — Score Calibration
**What:** Apply Platt scaling (logistic regression on validation set logits) per model to convert raw sigmoid outputs into true probabilities.

**Why it matters:** Makes the thresholds meaningful and comparable across models. A threshold of 0.60 should mean "60% empirical confidence this drug is correct."

**Where:** `agent/data_loader.py` — add a `calibrate_model()` function per model using the eval split from `split_data()`.

---

### Priority 2 — Adaptive Uncertainty Windows
**What:** Replace fixed `0.60 / 0.40` with per-model windows derived from the validation score distribution (e.g., upper/lower quartile of positive-class scores).

**Why it matters:** Each model has a different score dynamic. This makes `uncertain_config.yaml` data-driven instead of hand-tuned.

**Where:** `uncertain_config.yaml` values become outputs of a calibration script, not manual entries.

---

### Priority 3 — Confidence-Weighted Aggregation
**What:** Replace vote counting in `p4_summarize_tool` with a weighted score:

```
final_score(drug) = Σ(calibrated_score × tool_weight) / Σ(tool_weight)
```

where tool weights can be set by per-tool validation Jaccard.

**Why it matters:** Directly addresses the information loss in the current binary aggregation.

**Where:** `agent/tools.py:503` — rewrite the aggregation block in `p4_summarize_tool`.

---

### Priority 4 — DDI-Aware Arbitration
**What:** At the Role-Play LLM arbitration step, inject the DDI profile of each uncertain drug relative to the current AUTO_ACCEPT set.

**Why it matters:** The specialist arbitrating a drug should know it creates a DDI pair with an already-accepted drug — that is directly relevant clinical information the current prompt omits.

**Where:** `agent/tools.py:198` — extend `_call_roleplay_llm()` to accept the current accepted set and compute per-drug DDI flags from `agent/ddi.py`.

---

### Priority 5 — Cross-Domain Synthesis Layer
**What:** Add a new arbitration step between the tool outputs and the consultant. Drugs with split votes (1 or 2 out of 3 tools) get a dedicated cross-domain review by an LLM that receives all three domain signals simultaneously.

**Why it matters:** Currently the consultant sees a summary table — this layer does the hard arbitration before the consultant, letting the consultant focus on holistic clinical judgment rather than resolving disagreements. This is the most architecturally novel addition.

**Where:** New function in `agent/tools.py`, called inside `p4_summarize_tool` before returning.

---

### Priority 6 — Parallel Tool Execution
**What:** Execute the three domain tools as parallel LangGraph nodes rather than sequential tool calls.

**Why it matters:** Reduces wall-clock latency by ~3×. Also forces a cleaner architectural separation — tools cannot implicitly depend on each other's execution order.

**Where:** `agent/graph.py` — restructure with a parallel fan-out node using LangGraph's `Send` API.

---

### Priority 7 — Structured Consultant Output
**What:** Replace regex extraction in `benchmark.py:80` with LangChain structured output (`.with_structured_output()`) that constrains the agent to return a `list[str]` of valid ATC codes from the medication vocabulary.

**Why it matters:** Eliminates hallucinated drug codes and makes evaluation deterministic.

**Where:** `agent/graph.py` and `benchmark.py`.

---

## Paper Contribution Priority

For a NeurIPS submission, the most impactful contributions in order:

| Priority | Improvement | Measurable Impact |
|----------|-------------|-------------------|
| 1 | Calibration + adaptive windows | Makes zone mechanism principled, not heuristic |
| 2 | Cross-domain synthesis layer | Most architecturally novel addition |
| 3 | Confidence-weighted aggregation | Directly measurable improvement on Jaccard |
| 4 | DDI-aware arbitration | Directly measurable on DDI rate |
| 5 | Parallel execution | Engineering — latency improvement |
| 6 | Structured output | Engineering — evaluation reliability |
