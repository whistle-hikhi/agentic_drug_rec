"""Phase 4 benchmark runner.

Usage:
    # Run on 10 test patients (default config)
    python benchmark.py

    # Run on 20 patients with a specific LLM
    python benchmark.py --n_patients 20 --model gpt-4o

    # Load settings from a YAML file
    python benchmark.py --config config.yaml

    # Disable the DDI checker tool
    python benchmark.py --no_ddi
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.messages import ToolMessage as LCToolMessage

# ── project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from agent.config      import load_config, get_config, Config
from agent.data_loader import (
    get_data_bundle, split_data, preload_all,
    register_patient, get_patient_visits,
)
from agent.graph       import build_phase4_agent
from agent.prompts     import format_patient_message
from agent.codebook    import diag_label, proc_label, drug_label
from agent.ddi         import check_ddi


# ──────────────────────────── visit summary builder ───────────────────────────

def build_visit_summary(patient_visits: list, med_voc: dict,
                        diag_voc: dict, proc_voc: dict) -> str:
    lines = [f"Patient clinical history ({len(patient_visits)} visit(s)):"]
    for i, adm in enumerate(patient_visits):
        diag_raw = [diag_voc.get(d, str(d)) for d in adm[0]]
        proc_raw = [proc_voc.get(p, str(p)) for p in adm[1]]
        diag_labels = [diag_label(c) for c in diag_raw[:6]]
        if len(diag_raw) > 6:
            diag_labels.append(f"(+{len(diag_raw)-6} more)")
        proc_labels = [proc_label(c) for c in proc_raw[:4]]
        if len(proc_raw) > 4:
            proc_labels.append(f"(+{len(proc_raw)-4} more)")

        lines.append(f"\nVisit {i+1}:")
        lines.append(f"  Diagnoses:  {'; '.join(diag_labels) or 'none'}")
        lines.append(f"  Procedures: {'; '.join(proc_labels) or 'none'}")

        if i < len(patient_visits) - 1:
            med_labels = [drug_label(med_voc.get(m, f"drug_{m}")) for m in adm[2][:6]]
            if len(adm[2]) > 6:
                med_labels.append(f"(+{len(adm[2])-6} more)")
            lines.append(f"  Medications prescribed: {', '.join(med_labels) or 'none'}")
        else:
            lines.append("  Medications to recommend: [to be determined]")
    return "\n".join(lines)


# ──────────────────────────── recommendation extraction ───────────────────────

def extract_recommendation(text: str) -> list[str]:
    m = re.search(r"Recommended drugs?:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if not m:
        return []
    drug_str = m.group(1).strip().rstrip(".")
    return [re.split(r"[\s(]", d.strip())[0]
            for d in re.split(r"[,;]", drug_str) if d.strip()]


# ──────────────────────────── metrics ─────────────────────────────────────────

def _jaccard(pred: set, gt: set) -> float:
    if not pred and not gt:
        return 1.0
    inter = len(pred & gt)
    union = len(pred | gt)
    return inter / union if union else 0.0


def _prf1(pred: set, gt: set) -> tuple[float, float, float]:
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(pred & gt)
    p  = tp / len(pred)
    r  = tp / len(gt) if gt else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def compute_metrics(pred_drugs: list, gt_drugs: list, ddi_adj) -> dict:
    from sklearn.metrics import average_precision_score

    data_bundle, _, med_voc = get_data_bundle()
    med_voc_inv = {v: k for k, v in med_voc.items()}

    pred_set = set(pred_drugs); gt_set = set(gt_drugs)
    ja = _jaccard(pred_set, gt_set)
    p, r, f1 = _prf1(pred_set, gt_set)

    # DDI rate
    pred_idx = [med_voc_inv[d] for d in pred_drugs if d in med_voc_inv]
    ddi_rate = 0.0
    if len(pred_idx) >= 2:
        cnt = tot = 0
        for i in range(len(pred_idx)):
            for j in range(i + 1, len(pred_idx)):
                a, b = pred_idx[i], pred_idx[j]
                tot += 1
                if ddi_adj[a, b] == 1 or ddi_adj[b, a] == 1:
                    cnt += 1
        ddi_rate = cnt / tot if tot else 0.0

    # PRAUC
    n_drugs  = ddi_adj.shape[0]
    gt_vec   = np.zeros(n_drugs); pred_vec = np.zeros(n_drugs)
    for d in gt_drugs:
        idx = med_voc_inv.get(d)
        if idx is not None: gt_vec[idx] = 1
    for d in pred_drugs:
        idx = med_voc_inv.get(d)
        if idx is not None: pred_vec[idx] = 1
    prauc = float(average_precision_score(gt_vec, pred_vec)) if gt_vec.sum() > 0 else 0.0

    return {
        "jaccard":   round(ja, 4),
        "prauc":     round(prauc, 4),
        "precision": p,
        "recall":    r,
        "f1":        f1,
        "ddi_rate":  round(float(ddi_rate), 4),
        "n_pred":    len(pred_drugs),
        "n_gt":      len(gt_drugs),
    }


# ──────────────────────────── single-patient run ──────────────────────────────

def run_patient(agent, patient_id: str, patient_visits: list, visit_summary: str) -> dict:
    input_msg  = format_patient_message(patient_id, visit_summary)
    state_input = {
        "messages": [HumanMessage(content=input_msg)],
    }

    t0          = time.time()
    output_text = ""
    tool_calls  = 0
    tool_trace  = []

    try:
        active_calls   = {}   # index -> {name, args}
        llm_turn_active = False
        current_buf    = []
        llm_turns      = []
        pending_results = 0
        received        = 0

        print(f"\n  {'─'*60}")
        print(f"  AGENT TRACE — streaming:", flush=True)
        print(f"  Waiting for LLM...", flush=True)

        for chunk, _meta in agent.stream(state_input, stream_mode="messages"):
            if isinstance(chunk, AIMessageChunk):
                for tc in (chunk.tool_call_chunks or []):
                    idx  = tc.get("index", 0)
                    name = tc.get("name") or ""
                    args = tc.get("args") or ""
                    if idx not in active_calls:
                        active_calls[idx] = {"name": name, "args": args}
                        if name:
                            if llm_turn_active:
                                print(); llm_turn_active = False
                            print(f"\n  [TOOL CALL -> {name}] {args[:120]}", flush=True)
                            tool_calls += 1; pending_results += 1
                            tool_trace.append({
                                "type": "tool_call", "name": name,
                                "args": {}, "timestamp": round(time.time() - t0, 3)
                            })
                    else:
                        active_calls[idx]["args"] += args

                if chunk.content and isinstance(chunk.content, str):
                    if not llm_turn_active:
                        print(f"\n  [LLM +{round(time.time()-t0,1)}s] ", end="", flush=True)
                        llm_turn_active = True
                    print(chunk.content, end="", flush=True)
                    current_buf.append(chunk.content)

            elif isinstance(chunk, LCToolMessage):
                if llm_turn_active:
                    print(); llm_turn_active = False
                if current_buf:
                    llm_turns.append("".join(current_buf)); current_buf = []

                # Finalise args in trace
                for entry in reversed(tool_trace):
                    if entry["type"] == "tool_call" and entry["name"] == chunk.name \
                            and not entry["args"]:
                        tc_data = next(
                            (v for v in active_calls.values() if v["name"] == chunk.name), {})
                        try:
                            entry["args"] = json.loads(tc_data.get("args", "{}"))
                        except Exception:
                            entry["args"] = tc_data.get("args", {})
                        break

                elapsed = round(time.time() - t0, 1)
                content = chunk.content or ""
                print(f"\n  [TOOL OUTPUT <- {chunk.name} +{elapsed}s]:", flush=True)
                for line in (content.splitlines())[:20]:
                    print(f"    {line}", flush=True)
                tool_trace.append({
                    "type": "tool_output", "name": chunk.name,
                    "content": content, "timestamp": elapsed
                })

                received += 1
                if received >= pending_results:
                    active_calls.clear(); pending_results = 0; received = 0
                    print(f"\n  Waiting for LLM... (+{round(time.time()-t0,1)}s)", flush=True)

        if llm_turn_active:
            print()
        if current_buf:
            llm_turns.append("".join(current_buf))

        output_text = llm_turns[-1] if llm_turns else ""
        print(f"\n  {'─'*60}")

        return {
            "input_message":   input_msg,
            "output_text":     output_text,
            "predicted_drugs": extract_recommendation(output_text),
            "latency_s":       round(time.time() - t0, 2),
            "tool_calls":      tool_calls,
            "tool_trace":      tool_trace,
            "error":           None,
        }

    except Exception as e:
        return {
            "input_message":   input_msg,
            "output_text":     "",
            "predicted_drugs": [],
            "latency_s":       round(time.time() - t0, 2),
            "tool_calls":      0,
            "tool_trace":      tool_trace,
            "error":           str(e),
        }


# ──────────────────────────── benchmark loop ─────────────────────────────────

def run_benchmark(
    n_patients: int  = 10,
    dataset:    str  = "mimic3",
    llm_model:  str  = "gpt-4o-mini",
    use_ddi:    bool = True,
    output_dir: str  = None,
    cfg:        Config = None,
):
    if cfg is not None:
        n_patients = cfg.n_patients
        dataset    = cfg.dataset
        llm_model  = cfg.llm_model
        use_ddi    = cfg.use_ddi
        output_dir = output_dir or cfg.output_dir

    print(f"\n{'='*70}")
    print("PHASE 4 AGENTIC MEDICATION RECOMMENDATION BENCHMARK")
    print(f"Dataset: {dataset} | LLM: {llm_model} | Patients: {n_patients}")
    print(f"DDI tool: {'ON' if use_ddi else 'OFF'}")
    print(f"{'='*70}\n")

    output_dir = output_dir or str(_ROOT / "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading dataset...")
    data_bundle, voc_size, med_voc = get_data_bundle()
    ddi_adj  = data_bundle["ddi_adj"]
    records  = data_bundle["records"]
    voc      = data_bundle["voc"]
    diag_voc = voc["diag_voc"].idx2word
    proc_voc = voc["pro_voc"].idx2word

    data_train, data_eval, data_test = split_data(records)
    test_patients = [p for p in data_test if len(p) >= 2][:n_patients]
    print(f"Using {len(test_patients)} test patients (>=2 visits)\n")

    # Register patients
    for i, patient in enumerate(test_patients):
        register_patient(f"patient_{i}", patient)

    # Preload models
    print("Preloading models (may take a few minutes)...")
    preload_all()
    print("Models loaded.\n")

    # Build agent
    print("Building Phase 4 agent...")
    agent = build_phase4_agent(model=llm_model, use_ddi=use_ddi)

    # Run
    results = []
    for i, patient in enumerate(test_patients):
        pid = f"patient_{i}"
        print(f"\n--- Patient {i+1}/{len(test_patients)} ({pid}) ---")

        gt_drugs = [med_voc.get(m, f"drug_{m}") for m in patient[-1][2]]
        summary  = build_visit_summary(patient, med_voc, diag_voc, proc_voc)

        print(f"\nINPUT:\n{format_patient_message(pid, summary)}\n")

        run = run_patient(agent, pid, patient, summary)

        if run["error"]:
            print(f"  ERROR: {run['error'][:100]}")
            metrics = {"jaccard": 0, "prauc": 0, "precision": 0, "recall": 0,
                       "f1": 0, "ddi_rate": 0, "n_pred": 0, "n_gt": len(gt_drugs)}
        else:
            metrics = compute_metrics(run["predicted_drugs"], gt_drugs, ddi_adj)
            print(
                f"  Done: {run['latency_s']}s | {run['tool_calls']} tool calls | "
                f"Jaccard={metrics['jaccard']:.3f} | F1={metrics['f1']:.3f}"
            )

        results.append({
            "patient_id":      pid,
            "ground_truth":    gt_drugs,
            "predicted_drugs": run["predicted_drugs"],
            "metrics":         metrics,
            "latency_s":       run["latency_s"],
            "tool_calls":      run["tool_calls"],
            "tool_trace":      run["tool_trace"],
            "output_text":     run["output_text"],
            "error":           run["error"],
        })

    # Aggregate
    valid = [r for r in results if not r["error"]]

    def _mean(key):
        vals = [r["metrics"][key] for r in valid]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary_row = {
        "jaccard":    _mean("jaccard"),
        "prauc":      _mean("prauc"),
        "precision":  _mean("precision"),
        "recall":     _mean("recall"),
        "f1":         _mean("f1"),
        "ddi_rate":   _mean("ddi_rate"),
        "avg_pred":   round(sum(r["metrics"]["n_pred"] for r in valid) / max(len(valid), 1), 1),
        "avg_tools":  round(sum(r["tool_calls"] for r in valid) / max(len(valid), 1), 1),
        "avg_latency":round(sum(r["latency_s"] for r in valid) / max(len(valid), 1), 1),
        "n_patients": len(valid),
    }

    print(f"\n\n{'='*80}")
    print("PHASE 4 BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'Metric':<16} {'Value':>10}")
    print("-" * 28)
    for k, v in summary_row.items():
        print(f"  {k:<14} {str(v):>10}")
    print(f"{'='*80}\n")

    # Save
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    full = {
        "timestamp":   timestamp,
        "dataset":     dataset,
        "llm_model":   llm_model,
        "n_patients":  len(test_patients),
        "summary":     summary_row,
        "per_patient": results,
    }
    with open(result_path, "w") as f:
        json.dump(full, f, indent=2, default=str)
    print(f"Results saved to: {result_path}")
    return full


# ──────────────────────────── CLI ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 agentic medication recommendation benchmark"
    )
    parser.add_argument("--n_patients", type=int, default=10)
    parser.add_argument("--dataset",    choices=["mimic3", "mimic4"], default="mimic3")
    parser.add_argument("--model",      default="gpt-4o-mini",
                        help="OpenAI model for agent and Role-Play LLMs")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_ddi",     action="store_true",
                        help="Disable the DDI checker tool")
    parser.add_argument("--config",     default=None,
                        help="Path to YAML config file (overrides all other args)")
    args = parser.parse_args()

    cfg = None
    if args.config:
        cfg = load_config(args.config)
        print(f"Loaded config from: {args.config}")

    run_benchmark(
        n_patients  = args.n_patients,
        dataset     = args.dataset,
        llm_model   = args.model,
        use_ddi     = not args.no_ddi,
        output_dir  = args.output_dir,
        cfg         = cfg,
    )


if __name__ == "__main__":
    main()
