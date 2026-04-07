# agentic_drug_rec

Agentic Phase 4 medication recommendation pipeline for the **RecSync 2026** paper submission.  
Built on MIMIC-III, this system combines six trained ML models with a multi-tier LLM
arbitration strategy to produce safe, clinically justified drug recommendations.

---

## Overview

Phase 4 introduces **dual-model tools with Role-Play LLM arbitration** — a three-layer
decision architecture where drug candidates pass through ML model agreement zones before
reaching a final consultant-physician LLM agent.  The pipeline is fully self-contained:
all model architectures, data loading, DDI checking, and agent logic are implemented
from scratch in this repository.  The only external dependency on `agentic_pipeline/`
is the pre-trained model checkpoint files and preprocessed PKL data.

---

## Pipeline Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     PHASE 4 AGENTIC DRUG RECOMMENDATION                        ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  Patient EHR Input
  (diagnoses · procedures · prior medications · visit history)
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH REACT AGENT (Consultant Physician)               │
│                         LLM: gpt-4o-mini / gpt-4o                            │
│  Orchestrates the four tools below. Reviews the structured summary and makes  │
│  the final holistic prescription decision based on vote counts and DDI risk.  │
└─────────┬──────────────────┬──────────────────┬──────────────────┬────────────┘
          │                  │                  │                  │
    TOOL CALL 1        TOOL CALL 2        TOOL CALL 3        TOOL CALL 4
          │                  │                  │                  │
          ▼                  ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  LONGITUDINAL   │ │ SAFETY MOLECULE │ │   EHR-CENTRIC   │ │   SUMMARIZE     │
│     TOOL        │ │     TOOL        │ │     TOOL        │ │     TOOL        │
│                 │ │                 │ │                 │ │  (called last)  │
│  RETAIN         │ │  SafeDrug       │ │  DEPOT          │ │                 │
│    +            │ │    +            │ │    +            │ │ Merges outputs  │
│  GAMENet        │ │  MoleRec        │ │  MedAlign       │ │ from all 3      │
│                 │ │                 │ │                 │ │ domain tools    │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │                   │
         └───────────────────┴───────────────────┘                   │
                             │                                       │
              Per-drug zone classification                           │
              (for each of the two models):                          │
                             │                                       │
         ┌───────────────────┼───────────────────┐                   │
         ▼                   ▼                   ▼                   │
  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐            │
  │ AUTO_ACCEPT │    │  UNCERTAIN   │    │ AUTO_REJECT │            │
  │             │    │              │    │             │            │
  │ Both models │    │ Mixed signal │    │ Both models │            │
  │ score >=    │    │ (one accepts,│    │ score <     │            │
  │ accept_thr  │    │  one rejects,│    │ reject_thr  │            │
  │             │    │  or both in  │    │             │            │
  │ → Passes    │    │  grey zone)  │    │ → Dropped   │            │
  │   directly  │    │              │    │   silently  │            │
  └──────┬──────┘    └──────┬───────┘    └─────────────┘            │
         │                  │                                        │
         │                  ▼                                        │
         │      ┌───────────────────────┐                           │
         │      │   ROLE-PLAY LLM       │                           │
         │      │   ARBITRATION         │                           │
         │      │                       │                           │
         │      │  Domain-specific      │                           │
         │      │  clinical persona:    │                           │
         │      │                       │                           │
         │      │  Temporal Historian   │                           │
         │      │                       │                           │
         │      │  Tool 2 →             │                           │
         │      │  Chemical Safety      │                           │
         │      │  Specialist           │                           │
         │      │                       │                           │
         │      │  Tool 3 →             │                           │
         │      │  EHR Integration      │                           │
         │      │  Specialist           │                           │
         │      │                       │                           │
         │      │  PrimeKG              │                           │
         │      │  KG context injected  │                           │
         │      │  per uncertain drug   │                           │
         │      │                       │                           │
         │      │  → ACCEPT / REJECT    │                           │
         │      │    per drug           │                           │
         │      └──────────┬────────────┘                           │
         │                 │                                        │
         ▼                 ▼                                        │
  ┌──────────────────────────────┐                                  │
  │   ACCEPTED DRUGS (per tool)  │ ─────────────────────────────────┘
  │   AUTO_ACCEPT + LLM_ACCEPT   │          ▲
  │   ranked by mean score       │          │ All three tool outputs
  └──────────────────────────────┘          │ collected here
                                            │
                              ┌─────────────┴──────────────────┐
                              │       p4_summarize_tool         │
                              │                                 │
                              │  Per-drug vote count (0–3)      │
                              │  Mean confidence across tools   │
                              │  Tool-level zone:               │
                              │    AUTO_ACCEPT : votes >= 2     │
                              │    UNCERTAIN   : votes == 1     │
                              │  DDI analysis (greedy safe set) │
                              │  ATC class distribution         │
                              │  Prior medication context       │
                              └─────────────────┬───────────────┘
                                                │
                                                ▼
                              ┌─────────────────────────────────┐
                              │  CONSULTANT PHYSICIAN (ReAct)   │
                              │                                 │
                              │  Reviews structured brief:      │
                              │  · Unanimous drugs (3/3 tools)  │
                              │    → include directly           │
                              │  · Majority drugs (2/3 tools)   │
                              │    → include with review        │
                              │  · Uncertain drugs (1/3 tools)  │
                              │    → agent decides              │
                              │  · DDI-flagged drugs            │
                              │    → agent decides removal      │
                              │                                 │
                              │  FINAL RECOMMENDATION           │
                              │  Recommended drugs: <ATC3 list> │
                              └─────────────────────────────────┘
```

---

## Decision Zone Details

Each dual-model tool classifies every candidate drug through two tiers of zoning:

### Tier 1 — Per-Model Score Classification

| Zone      | Condition                                  |
|-----------|--------------------------------------------|
| ACCEPT    | score >= `accept_threshold` (default 0.60) |
| UNCERTAIN | `reject_threshold` <= score < `accept_threshold` |
| REJECT    | score < `reject_threshold` (default 0.40)  |

### Tier 2 — Combined Zone (two models per tool)

| Zone A   | Zone B   | Combined Result | Action                              |
|----------|----------|-----------------|-------------------------------------|
| ACCEPT   | ACCEPT   | AUTO_ACCEPT     | Drug passes directly                |
| REJECT   | REJECT   | AUTO_REJECT     | Drug dropped silently               |
| anything else      | — | UNCERTAIN    | Sent to Role-Play LLM               |

### Tier 3 — Tool-Level Summary Zone (across three tools)

| Votes | Result      | Action                                              |
|-------|-------------|-----------------------------------------------------|
| 3/3   | UNANIMOUS   | AUTO_ACCEPT — include directly                      |
| 2/3   | MAJORITY    | AUTO_ACCEPT — include directly                      |
| 1/3   | UNCERTAIN   | Agent decides based on clinical context             |
| 0/3   | —           | Drug does not appear in final set                   |

---

## ML Models

Six models are loaded as the three dual-model tools:

| Tool              | Model A    | Model B   | Clinical Signal                       |
|-------------------|------------|-----------|---------------------------------------|
| Longitudinal      | **RETAIN** | **GAMENet** | Multi-visit temporal history        |
| Safety Molecule   | **SafeDrug** | **MoleRec** | Molecular structure + DDI safety  |
| EHR-Centric       | **DEPOT**  | **MedAlign** | Full EHR integration              |

### Model Architecture Summary

| Model    | Architecture                                          | Key Feature                        |
|----------|-------------------------------------------------------|------------------------------------|
| RETAIN   | Bidirectional GRU + visit/code-level attention        | Temporal attention over visit history |
| GAMENet  | GRU encoders + dual GCN (EHR adj + DDI adj)           | Graph-augmented memory bank        |
| SafeDrug | GRU + bipartite DDI mask + optional MPNN              | DDI-controlled molecular encoding |
| MoleRec  | GRU + substructure SAB + masked attention aggregator  | Substructure-weighted relevance    |
| DEPOT    | GRU + SubGraphNetwork (Transformer over substructures)| Disease-progression motif matching |
| MedAlign | GRU + SubGraphNet + OT-based multi-modal alignment    | Code + text + structure alignment  |

All model architectures are implemented in `agent/models/` with exact parameter
layouts matching the pre-trained checkpoints from `agentic_pipeline/medrec_pipeline/`.

---

## Repository Structure

```
agentic_drug_rec/
│
├── benchmark.py              # CLI benchmark runner
├── config.yaml               # Default benchmark configuration
├── uncertain_config.yaml     # Per-model uncertainty thresholds
│
└── agent/
    ├── __init__.py
    ├── config.py             # Config dataclass + YAML loader
    ├── codebook.py           # ICD-9 / ATC code → human-readable labels
    ├── data_loader.py        # PKL data loading, model build, inference
    ├── ddi.py                # DDI checking + ddi_check_tool
    ├── primekg.py            # PrimeKG KG context (optional)
    ├── prompts.py            # Phase 4 system prompt + message formatter
    ├── tools.py              # Phase 4 LangChain tools (4 tools total)
    ├── graph.py              # build_phase4_agent() — LangGraph ReAct
    │
    └── models/               # Self-contained nn.Module implementations
        ├── __init__.py
        ├── retain.py         # Retain (GRU + dual attention)
        ├── gamenet.py        # GAMENet + GCN
        ├── safedrug.py       # SafeDrug + MPNN (rdkit or fallback)
        ├── molerec.py        # MoleRec + AdjAttenAggr
        ├── depot.py          # DrugRecNet + SubGraphNetwork
        └── medalign.py       # MedAlignNet + OT alignment + fallback
```

---

## Setup

### Prerequisites

```bash
pip install torch numpy dill pyyaml scikit-learn
pip install langchain langgraph langchain-openai
pip install python-dotenv
# Optional — enables MPNN branch in SafeDrug:
pip install rdkit
# Optional — enables OT alignment in MedAlign:
pip install POT
# Optional — enables PrimeKG KG context injection:
# Build primekg_index.pkl.gz with agentic_pipeline/agent/tools/build_primekg_index.py
```

### Environment

Create a `.env` file in `agentic_drug_rec/`:

```
OPENAI_API_KEY=sk-...
```

### Data and Checkpoints

This pipeline reads from `agentic_pipeline/medrec_pipeline/`:

| Resource          | Default path                                              |
|-------------------|-----------------------------------------------------------|
| PKL data files    | `../agentic_pipeline/medrec_pipeline/data/mimic3/`        |
| Model checkpoints | `../agentic_pipeline/medrec_pipeline/outputs/checkpoints/`|

Override via `config.yaml` (`data_dir` / `ckpt_dir` keys) or environment variables:

```bash
export MEDREC_DATA_DIR=/path/to/data
export MEDREC_CKPT_DIR=/path/to/checkpoints
```

Required PKL files per dataset:

```
records_final.pkl       # patient visit sequences
voc_final.pkl           # diagnosis / procedure / medication vocabularies
ddi_A_final.pkl         # DDI adjacency matrix
ddi_mask_H.pkl          # DDI mask for SafeDrug / MoleRec
atc3toSMILES.pkl        # ATC3 → SMILES mapping (for SafeDrug MPNN)
ehr_adj_final.pkl       # EHR co-occurrence adjacency (for GAMENet)
drug_smile.pkl          # drug × SMILES binary matrix (DEPOT / MedAlign)
smile_sub_b.pkl         # SMILES × substructure binary matrix
smile_sub_degree_b.pkl  # substructure degree features
smile_sub_recency_b.pkl # substructure recency features
```

---

## Running the Benchmark

```bash
# Run with defaults (10 patients, gpt-4o-mini, no DDI tool)
python benchmark.py

# Custom run
python benchmark.py --n_patients 50 --model gpt-4o --dataset mimic3

# Load all settings from YAML
python benchmark.py --config config.yaml

# Disable DDI checker tool
python benchmark.py --no_ddi
```

### Output

Results are saved to `outputs/benchmark_YYYYMMDD_HHMMSS.json` with:

- Per-patient: predicted drugs, ground truth, tool trace, latency
- Aggregate: Jaccard, PRAUC, F1, Precision, Recall, DDI rate, avg tools called

---

## Configuration Reference

### `config.yaml`

| Key          | Default       | Description                                    |
|--------------|---------------|------------------------------------------------|
| `llm_model`  | `gpt-4o-mini` | OpenAI model for agent + Role-Play LLMs        |
| `n_patients` | `10`          | Number of test patients to evaluate            |
| `dataset`    | `mimic3`      | `mimic3` or `mimic4`                           |
| `use_ddi`    | `false`       | Include `ddi_check_tool` in agent tool set     |
| `models`     | all 6 models  | ML models to load (must cover all three tools) |
| `data_dir`   | `""`          | Override data PKL directory                    |
| `ckpt_dir`   | `""`          | Override checkpoint directory                  |
| `output_dir` | `outputs`     | Directory for result JSON files                |

### `uncertain_config.yaml`

Per-model uncertainty thresholds used for zone classification.  
Default `accept_threshold: 0.60` / `reject_threshold: 0.40` for all models.  
Calibrate these using the model's score distribution on a held-out validation set.

---

## Evaluation Metrics

| Metric     | Description                                                      |
|------------|------------------------------------------------------------------|
| Jaccard    | |pred ∩ gt| / |pred ∪ gt| — primary recommendation quality metric |
| PRAUC      | Area under precision-recall curve over the full drug vocabulary  |
| Precision  | Fraction of predicted drugs that are correct                     |
| Recall     | Fraction of ground-truth drugs that are predicted                |
| F1         | Harmonic mean of precision and recall                            |
| DDI Rate   | Fraction of predicted drug pairs with a known interaction        |

---

## Extending the Pipeline

### Adding a new dual-model tool

1. Add the model `nn.Module` to `agent/models/`
2. Add a build function to `agent/data_loader.py` (`_BUILDERS` dict)
3. Add a persona string and `@tool` function in `agent/tools.py`
4. Add the tool to `PHASE4_TOOLS` and update the system prompt in `agent/prompts.py`

### Adjusting uncertainty thresholds

Edit `uncertain_config.yaml`.  To force a reload without restarting Python:

```python
from agent.tools import reload_thresholds
reload_thresholds()
```

### Using a different LLM provider

Replace `ChatOpenAI` in `agent/tools.py` (`_call_roleplay_llm`) and
`agent/graph.py` (`build_phase4_agent`) with any LangChain-compatible chat model.

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@inproceedings{recsync2026,
  title     = {RecSync: Agentic Multi-Model Drug Recommendation with Role-Play LLM Arbitration},
  booktitle = {Proceedings of RecSync 2026},
  year      = {2026},
}
```

---

## License

See [LICENSE](LICENSE) for details.
