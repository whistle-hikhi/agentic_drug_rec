"""PrimeKG knowledge-graph context for Role-Play LLM arbitration.

Loads a pre-built index file (primekg_index.pkl.gz) and injects KG facts
(indications, contraindications, targets, side effects) into the LLM prompt.
If the index does not exist the module degrades gracefully — all functions
return empty strings and the rest of the pipeline is unaffected.
"""
from __future__ import annotations

import functools
import gzip
import pickle
from pathlib import Path
from typing import Iterable

_INDEX_PATH = Path(__file__).parent / "primekg_index.pkl.gz"
_MAX_SHOW   = 8


@functools.lru_cache(maxsize=1)
def _load_index() -> dict | None:
    if not _INDEX_PATH.exists():
        return None
    with gzip.open(_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def is_available() -> bool:
    return _load_index() is not None


def _fmt_list(items: list[str], n: int = _MAX_SHOW) -> str:
    if not items:
        return "—"
    suffix = f"  (+{len(items) - n} more)" if len(items) > n else ""
    return "; ".join(items[:n]) + suffix


def get_primekg_context(
    atc3_codes: Iterable[str],
    patient_diag_names: list[str] | None = None,
) -> str:
    """Return a formatted KG context block for injection into a Role-Play LLM prompt.

    Returns an empty string if the index is unavailable.
    """
    index = _load_index()
    if index is None:
        return ""

    codes = [c.upper() for c in atc3_codes]
    patient_diags_lower = {d.lower() for d in (patient_diag_names or [])}

    lines: list[str] = [
        "── PrimeKG Knowledge Graph Context ──────────────────────────────────────────",
        "Facts retrieved from the PrimeKG biomedical KG. Use to inform ACCEPT/REJECT.",
        "",
    ]

    found_any = False
    for atc3 in codes:
        entry = index.get(atc3)
        if not entry:
            continue
        found_any = True

        atc_name    = entry.get("atc_name", atc3)
        drug_names  = entry.get("drug_names", [])
        indications = entry.get("indications", [])
        contras     = entry.get("contraindications", [])
        targets     = entry.get("targets", [])
        side_efx    = entry.get("side_effects", [])

        lines.append(f"Drug class: {atc3}  —  {atc_name}")
        if drug_names:
            lines.append(f"  Includes drugs: {_fmt_list(drug_names)}")

        if indications:
            marked = []
            for ind in indications[:_MAX_SHOW]:
                marker = " v" if any(ind.lower() in d or d in ind.lower()
                                     for d in patient_diags_lower) else ""
                marked.append(f"{ind}{marker}")
            suffix = (f"  (+{len(indications) - _MAX_SHOW} more)"
                      if len(indications) > _MAX_SHOW else "")
            lines.append(f"  Indications:        {'; '.join(marked)}{suffix}")
        else:
            lines.append("  Indications:        —")

        lines.append(f"  Contraindications:  {_fmt_list(contras)}")
        lines.append(f"  Molecular targets:  {_fmt_list(targets)}")
        lines.append(f"  Side effects:       {_fmt_list(side_efx)}")
        lines.append("")

    if not found_any:
        return ""

    lines.append("─" * 76)
    return "\n".join(lines)


def get_drug_summary(atc3: str) -> str:
    """Return a one-line KG summary for a single ATC3 code (for table rows)."""
    index = _load_index()
    if index is None:
        return ""
    entry = index.get(atc3.upper())
    if not entry:
        return ""
    parts = []
    indications = entry.get("indications", [])[:3]
    targets     = entry.get("targets", [])[:2]
    if indications:
        parts.append(f"treats: {', '.join(indications)}")
    if targets:
        parts.append(f"targets: {', '.join(targets)}")
    return " | ".join(parts) if parts else ""
