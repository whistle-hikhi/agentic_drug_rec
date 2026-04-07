"""Drug-Drug Interaction (DDI) checking utilities."""
from __future__ import annotations

from langchain_core.tools import tool

from .data_loader import get_data_bundle


def check_ddi(drug_names: list[str]) -> dict:
    """Check DDI for a list of drug names (ATC3 strings).

    Returns:
        {
            "ddi_pairs":      [[drug_a, drug_b], ...],
            "ddi_rate":       float,
            "n_interactions": int,
            "safe_drugs":     [str],
            "is_safe":        bool,
        }
    """
    data_bundle, _, med_voc = get_data_bundle()
    ddi_adj     = data_bundle["ddi_adj"]
    med_voc_inv = {v: k for k, v in med_voc.items()}

    indices = [med_voc_inv[n] for n in drug_names
               if med_voc_inv.get(n) is not None
               and med_voc_inv[n] < ddi_adj.shape[0]]

    if len(indices) < 2:
        return {"ddi_pairs": [], "ddi_rate": 0.0, "n_interactions": 0,
                "safe_drugs": list(drug_names), "is_safe": True}

    ddi_pairs = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            a, b = indices[i], indices[j]
            if ddi_adj[a, b] > 0 or ddi_adj[b, a] > 0:
                ddi_pairs.append([med_voc.get(a, f"drug_{a}"),
                                  med_voc.get(b, f"drug_{b}")])

    total_pairs = len(indices) * (len(indices) - 1) / 2
    ddi_rate    = len(ddi_pairs) / max(total_pairs, 1)

    # Greedy safe subset: iteratively remove the drug with most DDIs
    safe = list(indices)
    while True:
        involvement = {idx: 0 for idx in safe}
        has_ddi = False
        for i in range(len(safe)):
            for j in range(i + 1, len(safe)):
                a, b = safe[i], safe[j]
                if ddi_adj[a, b] > 0 or ddi_adj[b, a] > 0:
                    involvement[a] += 1
                    involvement[b] += 1
                    has_ddi = True
        if not has_ddi:
            break
        worst = max(involvement, key=lambda x: involvement[x])
        safe.remove(worst)

    safe_drugs = [med_voc.get(i, f"drug_{i}") for i in safe]
    return {
        "ddi_pairs":      ddi_pairs,
        "ddi_rate":       round(float(ddi_rate), 4),
        "n_interactions": len(ddi_pairs),
        "safe_drugs":     safe_drugs,
        "is_safe":        len(ddi_pairs) == 0,
    }


@tool
def ddi_check_tool(patient_id: str, candidate_drugs: str) -> str:
    """Screen a proposed medication set for drug-drug interactions (DDI).

    Args:
        patient_id:      patient identifier (for logging only)
        candidate_drugs: comma-separated ATC3 drug codes to screen

    Returns:
        Plain-text DDI report including interacting pairs and a safe subset.
    """
    import json
    drugs  = [d.strip() for d in candidate_drugs.split(",") if d.strip()]
    result = check_ddi(drugs)

    lines = [f"DDI check for {len(drugs)} candidate drugs:"]
    if result["is_safe"]:
        lines.append("No drug-drug interactions detected. Regimen is safe.")
    else:
        lines.append(
            f"Found {result['n_interactions']} DDI pair(s) "
            f"(rate {result['ddi_rate']:.1%}):"
        )
        for pair in result["ddi_pairs"][:10]:
            lines.append(f"  - {pair[0]} x {pair[1]}")
        removed = [d for d in drugs if d not in result["safe_drugs"]]
        lines.append(f"Remove due to DDI: {', '.join(removed)}")
        lines.append(
            f"Safe subset ({len(result['safe_drugs'])}): "
            + ", ".join(result["safe_drugs"])
        )
    lines.append(f"\nFull result: {json.dumps(result)}")
    return "\n".join(lines)
