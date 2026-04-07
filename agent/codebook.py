"""Human-readable descriptions for ICD-9 diagnosis/procedure codes and ATC3 drug codes."""
import csv
import functools
import os
from pathlib import Path

# Path to WHO ATC-DDD CSV (relative to this repo's parent)
_ATC_CSV = Path(__file__).parent.parent.parent / "data" / "WHO ATC-DDD 2021-12-03.csv"


# ──────────────────────────── ATC drug codes ──────────────────────────────────

@functools.lru_cache(maxsize=1)
def _load_atc() -> dict:
    path = _ATC_CSV.resolve()
    lookup = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get("atc_code", "").strip()
                name = row.get("atc_name", "").strip()
                if code and name:
                    lookup[code] = name.title()
    except FileNotFoundError:
        pass
    return lookup


def atc_desc(code: str) -> str:
    """Return human-readable ATC name; falls back to the code itself."""
    names = _load_atc()
    for c in [code, code[:4], code[:3], code[:1]]:
        if c in names:
            return names[c]
    return code


def drug_label(code: str) -> str:
    """Format as 'CODE (Description)' or just 'CODE' if unknown."""
    desc = atc_desc(code)
    return f"{code} ({desc})" if desc != code else code


# ──────────────────────────── ICD-9 diagnosis ─────────────────────────────────

_ICD9_DIAG_RANGES = [
    ((1,   139),  "Infectious & Parasitic Diseases"),
    ((140, 239),  "Neoplasms"),
    ((240, 279),  "Endocrine & Metabolic Diseases"),
    ((280, 289),  "Blood Diseases"),
    ((290, 319),  "Mental Disorders"),
    ((320, 389),  "Nervous System & Sense Organs"),
    ((390, 459),  "Circulatory System Diseases"),
    ((460, 519),  "Respiratory System Diseases"),
    ((520, 579),  "Digestive System Diseases"),
    ((580, 629),  "Genitourinary System Diseases"),
    ((630, 679),  "Pregnancy & Childbirth Complications"),
    ((680, 709),  "Skin & Subcutaneous Tissue Diseases"),
    ((710, 739),  "Musculoskeletal & Connective Tissue"),
    ((740, 759),  "Congenital Anomalies"),
    ((760, 779),  "Perinatal Conditions"),
    ((780, 799),  "Symptoms & Ill-defined Conditions"),
    ((800, 999),  "Injury & Poisoning"),
]

_ICD9_3DIGIT = {
    "250": "Diabetes Mellitus",
    "272": "Disorders of Lipoid Metabolism",
    "276": "Fluid/Electrolyte/Acid-Base Disorders",
    "285": "Other Anemias",
    "311": "Depressive Disorder",
    "401": "Essential Hypertension",
    "410": "Acute Myocardial Infarction",
    "414": "Chronic Ischemic Heart Disease",
    "427": "Cardiac Dysrhythmias",
    "428": "Heart Failure",
    "434": "Occlusion of Cerebral Arteries",
    "440": "Atherosclerosis",
    "480": "Viral Pneumonia",
    "486": "Pneumonia, Unspecified",
    "491": "Chronic Bronchitis",
    "496": "Chronic Airway Obstruction (COPD)",
    "571": "Chronic Liver Disease and Cirrhosis",
    "574": "Cholelithiasis",
    "585": "Chronic Kidney Disease",
    "714": "Rheumatoid Arthritis",
    "715": "Osteoarthrosis",
}


def icd9_diag_desc(code: str) -> str:
    code = str(code).strip().lstrip("0") or "0"
    if code.startswith("V"):
        return f"Supplementary Health Factor ({code})"
    if code.startswith("E"):
        return f"External Cause of Injury ({code})"
    prefix3 = code[:3]
    if prefix3 in _ICD9_3DIGIT:
        return _ICD9_3DIGIT[prefix3]
    try:
        num = int(prefix3)
        for (lo, hi), chapter in _ICD9_DIAG_RANGES:
            if lo <= num <= hi:
                return chapter
    except ValueError:
        pass
    return f"Diagnosis {code}"


def diag_label(code: str) -> str:
    return f"{code} ({icd9_diag_desc(code)})"


# ──────────────────────────── ICD-9 procedure ─────────────────────────────────

_ICD9_PROC_RANGES = [
    ((0,   5),   "Nervous System Operations"),
    ((6,   7),   "Endocrine System Operations"),
    ((8,  16),   "Eye Operations"),
    ((18, 20),   "Ear Operations"),
    ((21, 29),   "Nose, Mouth & Pharynx Operations"),
    ((30, 34),   "Respiratory System Operations"),
    ((35, 39),   "Cardiovascular Operations"),
    ((40, 41),   "Hemic & Lymphatic Operations"),
    ((42, 54),   "Digestive System Operations"),
    ((55, 59),   "Urinary System Operations"),
    ((60, 64),   "Male Genital Operations"),
    ((65, 71),   "Female Genital Operations"),
    ((72, 75),   "Obstetrical Procedures"),
    ((76, 84),   "Musculoskeletal Operations"),
    ((85, 86),   "Integumentary Operations"),
    ((87, 99),   "Diagnostic & Therapeutic Procedures"),
]


def icd9_proc_desc(code: str) -> str:
    code = str(code).strip()
    try:
        prefix2 = int(code[:2])
        for (lo, hi), category in _ICD9_PROC_RANGES:
            if lo <= prefix2 <= hi:
                return category
    except (ValueError, IndexError):
        pass
    return f"Procedure {code}"


def proc_label(code: str) -> str:
    return f"{code} ({icd9_proc_desc(code)})"
