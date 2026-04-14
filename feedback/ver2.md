# Pipeline Update Summary — Version 2
**Commit:** `bc56eb2` — "update new pipeline"
**Based on:** ver1 revision plan (`2a829b6`)

---

## What Was Implemented

Three of the seven improvements from ver1 were shipped in this commit.

---

### 1. Per-Model Calibrated Uncertainty Thresholds (`uncertain_config.yaml`)

**ver1 plan:** Priority 2 — replace fixed `0.60 / 0.40` with data-driven per-model windows.

**What changed:** All six models now have independent thresholds derived from validation
set score distributions, replacing the uniform hand-tuned values.

| Model     | Old accept / reject | New accept / reject | Uncertainty zone |
|-----------|---------------------|---------------------|------------------|
| RETAIN    | 0.60 / 0.40         | 0.023 / 0.018       | [0.018, 0.023)   |
| GAMENet   | 0.60 / 0.40         | 0.453 / 0.046       | [0.046, 0.453)   |
| SafeDrug  | 0.60 / 0.40         | 0.356 / 0.054       | [0.054, 0.356)   |
| MoleRec   | 0.60 / 0.40         | 0.306 / 0.050       | [0.050, 0.306)   |
| DEPOT     | 0.60 / 0.40         | 0.314 / 0.048       | [0.048, 0.314)   |
| MedAlign  | 0.60 / 0.40         | 0.392 / 0.046       | [0.046, 0.392)   |

Each model now has a much narrower and model-specific confidence signal. RETAIN, for
instance, produces broadly distributed probability mass and therefore has a very tight
uncertain window, while GAMENet is more selective with a wider window.

---

### 2. DDI-Aware Role-Play LLM Arbitration (`agent/tools.py`)

**ver1 plan:** Priority 4 — inject DDI conflict context into per-tool arbitration prompts.

**What changed:** `_call_roleplay_llm()` now accepts an `accepted_drugs` parameter.
Before calling the LLM, it computes DDI conflicts between each uncertain drug and the
already AUTO_ACCEPT set. The arbitration table gains a `DDI_conflicts` column and a
warning header is injected into the prompt when conflicts are found.

```
# Before
Drug          score_A  score_B  Zone_A  Zone_B

# After
Drug          score_A  score_B  Zone_A  Zone_B    DDI_conflicts
...           0.231    0.189    UNCERTAIN UNCERTAIN  [DDI: warfarin aspirin]
```

The `_phase4_impl()` caller passes `accepted_names` (the AUTO_ACCEPT set) into the
Role-Play LLM so the specialist can weigh DDI risk at arbitration time, not just as
a post-hoc flag.

---

### 3. Cross-Domain Synthesis Layer (`agent/tools.py`)

**ver1 plan:** Priority 5 — add a cross-domain arbitration step for split-vote drugs
before the consultant physician.

**What changed:** A new `_cross_domain_arbitration()` function was added and wired into
`p4_summarize_tool()`. Split-vote drugs (accepted by exactly 1 of 3 tools) are no
longer passed as `UNCERTAIN` to the consultant agent — instead they are resolved here
by a cross-domain LLM that sees all three domain signals simultaneously.

The LLM receives a table with longitudinal, safety, and EHR scores side-by-side and
reasons about whether the single accepting signal is clinically sufficient. Output is
`CROSS_ACCEPT` or `CROSS_REJECT`, and cross-accepted drugs join the final predicted
set alongside AUTO_ACCEPT drugs.

The summarize output now reports:
```
AUTO_ACCEPT (>=2 tools agreed):       N drugs
UNCERTAIN   (1 tool only, pre-synth): N drugs
Cross-domain ACCEPT:                  N drugs
Cross-domain REJECT:                  N drugs
Final predicted:                      N drugs
```

---

## What Remains From the ver1 Plan

| Priority | Improvement | Status |
|----------|-------------|--------|
| 1 | Score calibration (Platt scaling on logits) | **Not yet** — thresholds are updated but raw sigmoid outputs are still uncalibrated |
| 2 | Adaptive uncertainty windows | **Done** (`bc56eb2`) |
| 3 | Confidence-weighted aggregation | **Not yet** — vote counting still used in `p4_summarize_tool` |
| 4 | DDI-aware arbitration | **Done** (`bc56eb2`) |
| 5 | Cross-domain synthesis layer | **Done** (`bc56eb2`) |
| 6 | Parallel tool execution | **Not yet** |
| 7 | Structured consultant output | **Not yet** |

---

## Open Questions

- **Threshold calibration source:** The new thresholds in `uncertain_config.yaml` are
  labelled "AUTO-CALIBRATED" but the calibration script is not in the repo. How were
  these values derived — validation set percentile, Platt scaling, or manual inspection?
  This matters for the paper's methodology section.

- **RETAIN uncertainty window is very narrow** (`[0.018, 0.023)`). A 0.005 gap means
  almost all RETAIN predictions will be AUTO_ACCEPT or AUTO_REJECT with very few
  uncertain drugs escalated. This may reduce the Role-Play LLM's opportunity to
  contribute for the longitudinal tool.

- **Cross-domain synthesis is now a hard gate** — split-vote drugs resolved to
  `CROSS_REJECT` are completely removed from the final predicted set and are not
  shown to the consultant agent at all. Consider whether borderline `CROSS_REJECT`
  drugs should still be surfaced to the consultant as low-confidence candidates.
