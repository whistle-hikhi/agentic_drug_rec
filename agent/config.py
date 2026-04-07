"""Configuration for the Phase 4 agentic pipeline."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

_HERE = Path(__file__).parent.parent   # agentic_drug_rec/


@dataclass
class Config:
    # LLM used for ReAct agent and Role-Play LLM arbitration
    llm_model: str = "gpt-4o-mini"

    # Benchmark scope
    n_patients: int = 10
    dataset: str = "mimic3"
    use_ddi: bool = True

    # ML models loaded as Phase 4 tools
    models: List[str] = field(default_factory=lambda: [
        "retain", "gamenet", "safedrug", "molerec", "depot", "medalign",
    ])

    # Path overrides (empty → use defaults in data_loader.py)
    data_dir: str = ""
    ckpt_dir: str = ""

    # Output directory for benchmark results
    output_dir: str = str(_HERE / "outputs")


# ──────────────────────────── global singleton ────────────────────────────────

_active: Config = Config()


def get_config() -> Config:
    return _active


def set_config(cfg: Config) -> None:
    global _active
    _active = cfg


def load_config(path: str) -> Config:
    """Load YAML config and return a Config. Unknown keys are ignored."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    if not _YAML_OK:
        raise ImportError("PyYAML required: pip install pyyaml")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config(
        llm_model   = raw.get("llm_model",  Config.llm_model),
        n_patients  = int(raw.get("n_patients", Config.n_patients)),
        dataset     = raw.get("dataset",    Config.dataset),
        use_ddi     = bool(raw.get("use_ddi", Config.use_ddi)),
        models      = raw.get("models",     Config.__dataclass_fields__["models"].default_factory()),
        data_dir    = raw.get("data_dir",   ""),
        ckpt_dir    = raw.get("ckpt_dir",   ""),
        output_dir  = raw.get("output_dir", Config.output_dir),
    )
    set_config(cfg)

    # Propagate path overrides to data_loader environment
    import agent.data_loader as dl
    if cfg.data_dir:
        dl._DATA_ROOT = Path(cfg.data_dir)
    if cfg.ckpt_dir:
        dl._CKPT_ROOT = Path(cfg.ckpt_dir)

    return cfg
