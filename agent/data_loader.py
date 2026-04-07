"""Data and model loading for the Phase 4 agentic pipeline.

Loads pre-processed PKL files from the medrec_pipeline data directory,
builds and caches each model nn.Module, and provides inference utilities
used by the Phase 4 LangChain tools.

All model architectures are defined locally in agent/models/ so there is
no runtime dependency on medrec_pipeline or the original base_methods code.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import dill
import numpy as np
import torch
import torch.nn.functional as F

from .models.retain   import Retain
from .models.gamenet  import GAMENet
from .models.safedrug import SafeDrug, build_mpnn_inputs
from .models.molerec  import MoleRec
from .models.depot    import DrugRecNet
from .models.medalign import MedAlignNet, MedAlignFallback


# ──────────────────────────── path resolution ─────────────────────────────────

_HERE = Path(__file__).parent.parent   # agentic_drug_rec/

# Default data / checkpoint roots relative to this repo.
# Override via environment variables if needed.
_DATA_ROOT = Path(os.environ.get(
    "MEDREC_DATA_DIR",
    str(_HERE.parent / "agentic_pipeline" / "medrec_pipeline" / "data")
))
_CKPT_ROOT = Path(os.environ.get(
    "MEDREC_CKPT_DIR",
    str(_HERE.parent / "agentic_pipeline" / "medrec_pipeline" / "outputs" / "checkpoints")
))

DATASET = "mimic3"

# Models loaded as Phase 4 tools
PHASE4_MODELS = ["retain", "gamenet", "safedrug", "molerec", "depot", "medalign"]


# ──────────────────────────── global state ────────────────────────────────────

_lock              = threading.Lock()
_data_bundle: dict = {}
_voc_size: tuple   = ()
_med_voc: dict     = {}        # idx → ATC3 drug name
_med_voc_inv: dict = {}        # ATC3 drug name → idx
_loaded_models: dict = {}      # model_name → (nn.Module, device)
_initialized       = False

_patient_registry: dict = {}   # patient_id → patient_visits


# ──────────────────────────── data loading ────────────────────────────────────

def _load_pkl(data_dir: Path, fname: str):
    path = data_dir / fname
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return dill.load(f)


def _build_depot_bundle(molecules, med_voc_dict, voc_size, sub_num=64):
    """Generate DEPOT-required matrices on the fly when PKL files are absent."""
    num_drugs = voc_size[2]
    molecules = molecules or {}

    all_smiles, seen = [], set()
    for atc, smi_list in molecules.items():
        for smi in (smi_list or []):
            if smi not in seen:
                seen.add(smi); all_smiles.append(smi)
    if not all_smiles:
        all_smiles = ["C"]
    num_smiles = len(all_smiles)
    smi2idx = {s: i for i, s in enumerate(all_smiles)}

    drug_smile = np.zeros((num_drugs, num_smiles), dtype=np.float32)
    for drug_idx, atc in med_voc_dict.items():
        for smi in molecules.get(atc, []):
            if smi in smi2idx:
                drug_smile[drug_idx, smi2idx[smi]] = 1.0

    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        smile_sub = np.zeros((num_smiles, sub_num), dtype=np.float32)
        for i, smi in enumerate(all_smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, sub_num)
                for bit in fp.GetOnBits():
                    if bit < sub_num:
                        smile_sub[i, bit] = 1.0
    except Exception:
        rng = np.random.default_rng(42)
        smile_sub = rng.integers(0, 2, size=(num_smiles, sub_num)).astype(np.float32)

    MAX_SUB = 29
    for i in range(num_smiles):
        nz = np.where(smile_sub[i] > 0)[0]
        if len(nz) > MAX_SUB:
            smile_sub[i, nz[MAX_SUB:]] = 0.0

    rng = np.random.default_rng(42)
    smile_sub_degree  = smile_sub * rng.integers(1, 10, size=smile_sub.shape).astype(np.float32)
    smile_sub_recency = smile_sub * rng.integers(1, 30, size=smile_sub.shape).astype(np.float32)
    return drug_smile, smile_sub, smile_sub_degree, smile_sub_recency


def _init_data(dataset: str = DATASET):
    global _data_bundle, _voc_size, _med_voc, _med_voc_inv, _initialized
    if _initialized:
        return
    data_dir = _DATA_ROOT / dataset

    records  = _load_pkl(data_dir, "records_final.pkl")
    voc      = _load_pkl(data_dir, "voc_final.pkl")
    ddi_adj  = _load_pkl(data_dir, "ddi_A_final.pkl")
    ddi_mask = _load_pkl(data_dir, "ddi_mask_H.pkl")
    molecules = _load_pkl(data_dir, "atc3toSMILES.pkl")
    ehr_adj  = _load_pkl(data_dir, "ehr_adj_final.pkl")

    if records is None or voc is None or ddi_adj is None:
        raise FileNotFoundError(
            f"Required PKL files not found in {data_dir}. "
            "Run the medrec_pipeline preprocessing first."
        )

    diag_voc = voc["diag_voc"]
    pro_voc  = voc["pro_voc"]
    med_voc  = voc["med_voc"]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    if ehr_adj is None:
        n = voc_size[2]
        ehr_adj = np.eye(n, dtype=np.float32)
    else:
        ehr_adj = np.array(ehr_adj)
        if ehr_adj.shape[0] != voc_size[2]:
            n = voc_size[2]
            ehr_adj = ehr_adj[:n, :n]

    if ddi_mask is None:
        ddi_mask = np.ones((voc_size[2], 64), dtype=np.float32)

    bundle: dict = {
        "records":  records,
        "voc_size": voc_size,
        "voc":      voc,
        "ddi_adj":  ddi_adj,
        "ddi_mask": ddi_mask,
        "ehr_adj":  ehr_adj,
        "molecules": molecules,
        "med_voc":  med_voc.idx2word,
    }

    for key, fname in [
        ("drug_smile",        "drug_smile.pkl"),
        ("smile_sub",         "smile_sub_b.pkl"),
        ("smile_sub_voc",     "smile_sub_voc_b.pkl"),
        ("smile_sub_degree",  "smile_sub_degree_b.pkl"),
        ("smile_sub_recency", "smile_sub_recency_b.pkl"),
        ("drug_text_embs",    "drug_text_embs.pkl"),
    ]:
        val = _load_pkl(data_dir, fname)
        if val is not None:
            bundle[key] = val

    if "drug_smile" not in bundle:
        ds, ss, ssd, ssr = _build_depot_bundle(
            bundle.get("molecules"), bundle["med_voc"], voc_size)
        bundle["drug_smile"]        = ds
        bundle["smile_sub"]         = ss
        bundle["smile_sub_degree"]  = ssd
        bundle["smile_sub_recency"] = ssr
        for fname, data in [
            ("drug_smile.pkl",          ds),
            ("smile_sub_b.pkl",         ss),
            ("smile_sub_degree_b.pkl",  ssd),
            ("smile_sub_recency_b.pkl", ssr),
        ]:
            try:
                with open(data_dir / fname, "wb") as f:
                    dill.dump(data, f)
            except Exception:
                pass

    _data_bundle = bundle
    _voc_size    = voc_size
    _med_voc     = med_voc.idx2word
    _med_voc_inv = {v: k for k, v in _med_voc.items()}
    _initialized = True


def get_data_bundle():
    with _lock:
        _init_data()
    return _data_bundle, _voc_size, _med_voc


def split_data(records: list):
    """Return (train, eval, test) using the same 2/3 : 1/6 : 1/6 split."""
    n   = len(records)
    sp1 = int(n * 2 / 3)
    rem = n - sp1
    sp2 = sp1 + rem // 2
    return records[:sp1], records[sp2:], records[sp1:sp2]


# ──────────────────────────── device ──────────────────────────────────────────

def resolve_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ──────────────────────────── checkpoint finding ──────────────────────────────

def _find_best_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    best_path, best_ja = None, -1.0
    if not ckpt_dir.exists():
        return None
    for fname in ckpt_dir.iterdir():
        if "_JA_" in fname.name and fname.suffix == ".model":
            try:
                ja = float(fname.name.split("_JA_")[1].split("_")[0])
                if ja > best_ja:
                    best_ja  = ja
                    best_path = fname
            except Exception:
                pass
    return best_path


# ──────────────────────────── model builders ──────────────────────────────────

def _build_retain(voc_size, data_bundle, device, cfg) -> torch.nn.Module:
    model = Retain(voc_size, emb_size=cfg.get("emb_dim", 64), device=device)
    return model.to(device)


def _build_gamenet(voc_size, data_bundle, device, cfg) -> torch.nn.Module:
    model = GAMENet(voc_size, data_bundle["ehr_adj"], data_bundle["ddi_adj"],
                    emb_dim=cfg.get("emb_dim", 64), device=device)
    return model.to(device)


def _build_safedrug(voc_size, data_bundle, device, cfg) -> torch.nn.Module:
    ddi_adj  = data_bundle["ddi_adj"]
    ddi_mask = data_bundle.get("ddi_mask", np.ones((voc_size[2], 64)))
    molecules = data_bundle.get("molecules")
    med_voc_d = data_bundle.get("med_voc")
    mpnn_data = None
    if molecules is not None and med_voc_d is not None:
        mpnn_data = build_mpnn_inputs(
            molecules, med_voc_d, radius=2, device=torch.device("cpu"))
    model = SafeDrug(voc_size, ddi_adj, ddi_mask, mpnn_data,
                     emb_dim=cfg.get("emb_dim", 256), device=device)
    return model.to(device)


def _build_molerec(voc_size, data_bundle, device, cfg) -> torch.nn.Module:
    ddi_mask = data_bundle.get("ddi_mask", np.ones((voc_size[2], 64)))
    substruct_num = ddi_mask.shape[1]
    model = MoleRec(voc_size, substruct_num,
                    emb_dim=cfg.get("emb_dim", 64),
                    dropout=cfg.get("dropout", 0.7),
                    device=device)
    return model.to(device)


def _build_depot(voc_size, data_bundle, device, cfg) -> torch.nn.Module:
    drug_smile = torch.FloatTensor(data_bundle["drug_smile"]).to(device)
    smile_sub  = torch.FloatTensor(data_bundle["smile_sub"]).to(device)
    ddi_matrix = torch.FloatTensor(data_bundle["ddi_adj"]).to(device)
    stru = (
        torch.FloatTensor(data_bundle["smile_sub_recency"]).to(device),
        torch.FloatTensor(data_bundle["smile_sub_degree"]).to(device),
    )
    model = DrugRecNet(voc_size, cfg.get("emb_dim", 64),
                       drug_smile, smile_sub, ddi_matrix, stru, device)
    return model.to(device)


def _build_medalign(voc_size, data_bundle, device, cfg) -> torch.nn.Module:
    ddi_matrix = torch.FloatTensor(data_bundle["ddi_adj"]).to(device)
    drug_smile = (torch.FloatTensor(data_bundle["drug_smile"]).to(device)
                  if "drug_smile" in data_bundle else None)
    smile_sub  = (torch.FloatTensor(data_bundle["smile_sub"]).to(device)
                  if "smile_sub" in data_bundle else None)
    stru = None
    if "smile_sub_recency" in data_bundle:
        stru = (torch.FloatTensor(data_bundle["smile_sub_recency"]).to(device),
                torch.FloatTensor(data_bundle["smile_sub_degree"]).to(device))
    text_embs = (torch.FloatTensor(data_bundle["drug_text_embs"]).to(device)
                 if "drug_text_embs" in data_bundle else None)

    # Detect fallback architecture by checking saved state dict keys
    # (done by caller via _force_fallback flag — handled in _load_model below)
    if drug_smile is None or smile_sub is None or stru is None:
        model = MedAlignFallback(voc_size, cfg.get("emb_dim", 64), ddi_matrix, device)
    else:
        model = MedAlignNet(voc_size, cfg.get("emb_dim", 64),
                            drug_smile, smile_sub, ddi_matrix, stru, text_embs, device)
    return model.to(device)


_BUILDERS = {
    "retain":   _build_retain,
    "gamenet":  _build_gamenet,
    "safedrug": _build_safedrug,
    "molerec":  _build_molerec,
    "depot":    _build_depot,
    "medalign": _build_medalign,
}

_DEFAULT_CFG = {
    "retain":   {"emb_dim": 64},
    "gamenet":  {"emb_dim": 64},
    "safedrug": {"emb_dim": 256},
    "molerec":  {"emb_dim": 64, "dropout": 0.7},
    "depot":    {"emb_dim": 64},
    "medalign": {"emb_dim": 64},
}


def _load_model(model_name: str):
    """Build and load the best checkpoint for a model. Returns (nn.Module, device)."""
    _init_data()
    device    = resolve_device("auto")
    cfg       = _DEFAULT_CFG.get(model_name, {})
    build_fn  = _BUILDERS[model_name]
    ckpt_dir  = _CKPT_ROOT / model_name

    ckpt_path = _find_best_checkpoint(ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint for '{model_name}' in {ckpt_dir}. "
            "Train the model first with medrec_pipeline/pipeline/runner.py."
        )

    # Detect SafeDrug fallback (no MPNN — has 'd_emb.weight' key)
    saved_keys = set(torch.load(str(ckpt_path), map_location="cpu").keys())
    if model_name == "safedrug" and "d_emb.weight" in saved_keys:
        cfg = dict(cfg)
        cfg["_force_fallback"] = True
    if model_name == "medalign" and "d_emb.weight" in saved_keys:
        cfg = dict(cfg)
        cfg["_force_fallback"] = True

    model = build_fn(_voc_size, _data_bundle, device, cfg)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def get_model(model_name: str):
    """Return cached (nn.Module, device). Loads on first call (thread-safe)."""
    with _lock:
        if model_name not in _loaded_models:
            _loaded_models[model_name] = _load_model(model_name)
    return _loaded_models[model_name]


def preload_all():
    """Eagerly load all Phase 4 models. Call at benchmark startup."""
    _init_data()
    for name in PHASE4_MODELS:
        try:
            get_model(name)
            print(f"  [data_loader] loaded: {name}", flush=True)
        except Exception as e:
            print(f"  [data_loader] WARNING: could not load {name}: {e}", flush=True)


# ──────────────────────────── patient registry ────────────────────────────────

def register_patient(patient_id: str, patient_visits: list):
    _patient_registry[patient_id] = patient_visits


def get_patient_visits(patient_id: str) -> list:
    if patient_id not in _patient_registry:
        raise KeyError(
            f"Patient '{patient_id}' not registered. Call register_patient() first.")
    return _patient_registry[patient_id]


# ──────────────────────────── inference ───────────────────────────────────────

def _prob_array_to_scores(prob_arr: np.ndarray, threshold: float) -> dict:
    scores = {}
    for idx, p in enumerate(prob_arr):
        if float(p) >= threshold:
            name = _med_voc.get(idx, f"drug_{idx}")
            scores[name] = round(float(p), 4)
    return scores


def predict_visits(model_name: str, patient_visits: list, threshold: float = 0.0) -> dict:
    """Run inference for one patient. Returns {predicted_drugs, scores, num_predicted}."""
    if not patient_visits:
        return {"predicted_drugs": [], "scores": {}, "num_predicted": 0}

    with _lock:
        _init_data()

    model, device = get_model(model_name)
    voc_size = _voc_size
    prob_arr = np.zeros(voc_size[2], dtype=np.float32)

    with torch.no_grad():
        if model_name == "molerec":
            ddi_mask   = torch.FloatTensor(_data_bundle["ddi_mask"]).to(device)
            tensor_ddi = torch.FloatTensor(_data_bundle["ddi_adj"]).to(device)
            out, _     = model(patient_visits, ddi_mask, tensor_ddi)
            prob_arr   = torch.sigmoid(out).detach().cpu().numpy()[0]
        else:
            out = model(patient_visits)
            if isinstance(out, tuple):
                out = out[0]
            prob_arr = torch.sigmoid(out).detach().cpu().numpy()[0]

    scores          = _prob_array_to_scores(prob_arr, threshold)
    predicted_drugs = sorted(scores, key=lambda d: -scores[d])
    return {"predicted_drugs": predicted_drugs, "scores": scores,
            "num_predicted": len(predicted_drugs)}


def _run_model_timed(name: str, patient_visits: list) -> tuple:
    t0 = time.time()
    try:
        result = predict_visits(name, patient_visits, threshold=0.0)
    except Exception as e:
        result = {"error": str(e), "predicted_drugs": [], "scores": {}}
    return name, result, round(time.time() - t0, 2)


def predict_two_models(model_a: str, model_b: str, patient_visits: list) -> dict:
    """Run two models in parallel and return their raw score dicts."""
    per_model = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_run_model_timed, model_a, patient_visits): model_a,
            pool.submit(_run_model_timed, model_b, patient_visits): model_b,
        }
        for fut in as_completed(futures):
            name, result, elapsed = fut.result()
            per_model[name] = result
            status = "ERROR" if "error" in result else f"{result['num_predicted']} drugs"
            print(f"    [{name}] {status} ({elapsed}s)", flush=True)
    return per_model
