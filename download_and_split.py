"""
download_and_split.py
Downloads TCGA clinical data from the GDC (Genomic Data Commons) REST API,
maps columns to the FL Platform schema, and splits into 4 equal client datasets.

Usage:
    pip install requests pandas scikit-learn
    python download_and_split.py

Output:
    data/client_1.csv
    data/client_2.csv
    data/client_3.csv
    data/client_4.csv
    data/full_dataset.csv
"""

import os
import json  # noqa: F401
import math
import requests
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split  # noqa: F401

# ── GDC API ──────────────────────────────────────────────────────────────────
GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"
GDC_CLINICAL_FIELDS = [
    # Demographics
    "demographic.age_at_index",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.days_to_birth",
    "demographic.cause_of_death",
    # Diagnosis
    "diagnoses.age_at_diagnosis",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.cancer_type",
    "diagnoses.tumor_stage",
    "diagnoses.tumor_grade",
    "diagnoses.primary_diagnosis",
    "diagnoses.morphology",
    "diagnoses.tissue_or_organ_of_origin",
    "diagnoses.site_of_resection_or_biopsy",
    "diagnoses.classification_of_primary_diagnosis",
    "diagnoses.tumor_focality",
    "diagnoses.tumor_largest_dimension",
    "diagnoses.laterality",
    "diagnoses.year_of_diagnosis",
    "diagnoses.days_to_diagnosis",
    "diagnoses.metastasis_at_diagnosis",
    "diagnoses.synchronous_malignancy",
    "diagnoses.prior_malignancy",
    "diagnoses.days_to_new_event",
    "diagnoses.new_event_type",
    "diagnoses.new_event_anatomic_site",
    "diagnoses.ann_arbor_clinical_stage",
    "diagnoses.ann_arbor_b_symptoms",
    "diagnoses.burkitt_lymphoma_clinical_variant",
    "diagnoses.figo_stage",
    "diagnoses.iss_stage",
    "diagnoses.masaoka_stage",
    "diagnoses.primary_gleason_grade",
    "diagnoses.secondary_gleason_grade",
    "diagnoses.circumferential_resection_margin",
    "diagnoses.perineural_invasion_present",
    "diagnoses.colon_polyps_history",
    "diagnoses.hpv_positive_type",
    "diagnoses.hpv_status_by_p16_testing",
    "diagnoses.ldh_level_at_diagnosis",
    "diagnoses.ldh_normal_range_upper",
    # Exposures
    "exposures.alcohol_history",
    "exposures.alcohol_intensity",
    "exposures.bmi",
    "exposures.height",
    "exposures.weight",
    "exposures.weeks_gestation_at_birth",
    "exposures.tobacco_smoking_onset_year",
    "exposures.tobacco_smoking_quit_year",
    "exposures.type_of_tobacco_used",
    # HIV
    "diagnoses.hiv_positive",
    "diagnoses.days_to_hiv_diagnosis",
    # Treatments
    "treatments.treatment_type",
    "treatments.treatment_outcome",
    "treatments.treatment_or_therapy",
    "treatments.days_to_treatment_start",
    "treatments.days_to_treatment_end",
    "treatments.initial_disease_status",
    "treatments.regimen_or_line_of_therapy",
    "treatments.therapeutic_agents",
    "treatments.number_of_cycles",
    # Case info
    "case_id",
    "primary_site",
]

# ── Column rename map: GDC field → FL Platform schema column name ─────────────
RENAME_MAP = {
    "demographic.age_at_index":               "age_at_diagnosis",
    "demographic.gender":                     "gender",
    "demographic.race":                       "race",
    "demographic.ethnicity":                  "ethnicity",
    "demographic.vital_status":               "vital_status",
    "demographic.days_to_death":              "days_to_death",
    "demographic.days_to_birth":              "days_to_birth",
    "demographic.cause_of_death":             "cause_of_death",
    "diagnoses.age_at_diagnosis":             "age_at_diagnosis",
    "diagnoses.days_to_last_follow_up":       "days_to_last_follow_up",
    "diagnoses.primary_diagnosis":            "primary_diagnosis",
    "diagnoses.morphology":                   "morphology",
    "diagnoses.tissue_or_organ_of_origin":    "tissue_or_organ_of_origin",
    "diagnoses.site_of_resection_or_biopsy":  "site_of_resection_or_biopsy",
    "diagnoses.classification_of_primary_diagnosis": "classification_of_primary_diagnosis",
    "diagnoses.tumor_focality":               "tumor_focality",
    "diagnoses.tumor_largest_dimension":      "tumor_largest_dimension",
    "diagnoses.laterality":                   "laterality",
    "diagnoses.year_of_diagnosis":            "year_of_diagnosis",
    "diagnoses.days_to_diagnosis":            "days_to_diagnosis",
    "diagnoses.metastasis_at_diagnosis":      "metastasis_at_diagnosis",
    "diagnoses.synchronous_malignancy":       "synchronous_malignancy",
    "diagnoses.prior_malignancy":             "prior_malignancy",
    "diagnoses.prior_treatment":              "prior_treatment",
    "diagnoses.days_to_new_event":            "days_to_new_event",
    "diagnoses.new_event_type":               "new_event_type",
    "diagnoses.new_event_anatomic_site":      "new_event_anatomic_site",
    "diagnoses.ann_arbor_clinical_stage":     "ann_arbor_clinical_stage",
    "diagnoses.ann_arbor_b_symptoms":         "ann_arbor_b_symptoms",
    "diagnoses.burkitt_lymphoma_clinical_variant": "burkitt_lymphoma_clinical_variant",
    "diagnoses.figo_stage":                   "figo_stage",
    "diagnoses.iss_stage":                    "iss_stage",
    "diagnoses.masaoka_stage":                "masaoka_stage",
    "diagnoses.primary_gleason_grade":        "primary_gleason_grade",
    "diagnoses.secondary_gleason_grade":      "secondary_gleason_grade",
    "diagnoses.circumferential_resection_margin": "circumferential_resection_margin",
    "diagnoses.perineural_invasion_present":  "perineural_invasion_present",
    "diagnoses.colon_polyps_history":         "colon_polyps_history",
    "diagnoses.hpv_positive_type":            "hpv_positive_type",
    "diagnoses.hpv_status_by_p16_testing":    "hpv_status_by_p16_testing",
    "diagnoses.ldh_level_at_diagnosis":       "ldh_level_at_diagnosis",
    "diagnoses.ldh_normal_range_upper":       "ldh_normal_range_upper",
    "diagnoses.tumor_stage":                  "tumor_stage",
    "diagnoses.tumor_grade":                  "tumor_grade",
    "diagnoses.hiv_positive":                 "hiv_positive",
    "diagnoses.days_to_hiv_diagnosis":        "days_to_hiv_diagnosis",
    "exposures.alcohol_history":              "alcohol_history",
    "exposures.alcohol_intensity":            "alcohol_intensity",
    "exposures.bmi":                          "bmi",
    "exposures.height":                       "height",
    "exposures.weight":                       "weight",
    "exposures.weeks_gestation_at_birth":     "weeks_gestation_at_birth",
    "exposures.tobacco_smoking_onset_year":   "tobacco_smoking_onset_year",
    "exposures.tobacco_smoking_quit_year":    "tobacco_smoking_quit_year",
    "exposures.type_of_tobacco_used":         "type_of_tobacco_used",
    "treatments.treatment_type":              "treatment_type",
    "treatments.treatment_outcome":           "treatment_outcome",
    "treatments.treatment_or_therapy":        "treatment_or_therapy",
    "treatments.days_to_treatment_start":     "days_to_treatment_start",
    "treatments.days_to_treatment_end":       "days_to_treatment_end",
    "treatments.initial_disease_status":      "initial_disease_status",
    "treatments.regimen_or_line_of_therapy":  "regimen_or_line_of_therapy",
    "treatments.therapeutic_agents":          "therapeutic_agents",
    "treatments.number_of_cycles":            "number_of_cycles",
    "primary_site":                           "cancer_type",
}

# ── All columns required by FL Platform schema ────────────────────────────────
REQUIRED_COLS = [
    "age_at_diagnosis", "gender", "race", "ethnicity", "vital_status",
    "days_to_death", "days_to_last_follow_up", "cancer_type", "tumor_stage",
    "tumor_grade", "treatment_type", "prior_malignancy", "prior_treatment",
    "tissue_or_organ_of_origin", "morphology", "site_of_resection_or_biopsy",
    "classification_of_primary_diagnosis", "tumor_focality", "tumor_largest_dimension",
    "lymph_node_involved_site", "metastasis_at_diagnosis", "synchronous_malignancy",
    "year_of_diagnosis", "ann_arbor_b_symptoms", "ann_arbor_clinical_stage",
    "burkitt_lymphoma_clinical_variant", "cause_of_death",
    "circumferential_resection_margin", "colon_polyps_history", "days_to_birth",
    "days_to_diagnosis", "days_to_hiv_diagnosis", "days_to_new_event", "figo_stage",
    "hiv_positive", "hpv_positive_type", "hpv_status_by_p16_testing", "iss_stage",
    "laterality", "ldh_level_at_diagnosis", "ldh_normal_range_upper", "masaoka_stage",
    "new_event_anatomic_site", "new_event_type", "overall_survival",
    "perineural_invasion_present", "primary_diagnosis", "primary_gleason_grade",
    "secondary_gleason_grade", "weight", "tobacco_smoking_onset_year",
    "tobacco_smoking_quit_year", "type_of_tobacco_used", "alcohol_history",
    "alcohol_intensity", "bmi", "height", "weeks_gestation_at_birth",
    "treatment_outcome", "treatment_or_therapy", "days_to_treatment_start",
    "days_to_treatment_end", "initial_disease_status", "regimen_or_line_of_therapy",
    "therapeutic_agents", "number_of_cycles",
]


# =============================================================================
# Download
# =============================================================================

def _flatten_case(case: dict) -> dict:
    """Flatten a nested GDC case record into a single-level dict."""
    row = {}

    # Demographics
    demo = case.get("demographic", {}) or {}
    for k, v in demo.items():
        row[f"demographic.{k}"] = v

    # Diagnoses — take first entry
    diagnoses = case.get("diagnoses") or []
    diag = diagnoses[0] if diagnoses else {}
    for k, v in diag.items():
        row[f"diagnoses.{k}"] = v

    # Exposures — take first entry
    exposures = case.get("exposures") or []
    exp = exposures[0] if exposures else {}
    for k, v in exp.items():
        row[f"exposures.{k}"] = v

    # Treatments — take first entry
    treatments = case.get("treatments") or []
    treat = treatments[0] if treatments else {}
    for k, v in treat.items():
        row[f"treatments.{k}"] = v

    row["case_id"] = case.get("case_id", "")
    row["primary_site"] = case.get("primary_site", "")

    return row


def download_tcga(max_cases: int = 10000) -> pd.DataFrame:
    """
    Download TCGA clinical cases from the GDC API in pages of 500.
    Returns a raw DataFrame with dot-notation column names.
    """
    print(f"[*] Downloading up to {max_cases:,} TCGA cases from GDC API…")
    records = []
    page_size = 500
    pages = math.ceil(max_cases / page_size)

    for page in range(pages):
        offset = page * page_size
        size = min(page_size, max_cases - offset)

        params = {
            "fields": ",".join(GDC_CLINICAL_FIELDS),
            "expand": "demographic,diagnoses,exposures,treatments",
            "size":   size,
            "from":   offset,
            "format": "JSON",
        }

        try:
            resp = requests.get(GDC_CASES_URL, params=params, timeout=60)
            resp.raise_for_status()
            hits = resp.json().get("data", {}).get("hits", [])
        except Exception as e:
            print(f"    [!] Page {page + 1} failed: {e} — stopping early.")
            break

        if not hits:
            print(f"    [*] No more records at offset {offset}.")
            break

        for case in hits:
            records.append(_flatten_case(case))

        fetched = offset + len(hits)
        print(f"    [{fetched:>6}/{max_cases}] cases downloaded…", end="\r")

    print(f"\n[+] Total raw records: {len(records):,}")
    return pd.DataFrame(records)


# =============================================================================
# Clean & align to schema
# =============================================================================

def clean_and_align(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename GDC columns to FL Platform schema names,
    add missing required columns as NaN, and derive `overall_survival`.
    """
    df = raw_df.rename(columns=RENAME_MAP)

    # Derive overall_survival from days_to_death or days_to_last_follow_up
    if "days_to_death" in df.columns and "days_to_last_follow_up" in df.columns:
        df["overall_survival"] = df["days_to_death"].combine_first(
            df["days_to_last_follow_up"]
        )
    elif "days_to_last_follow_up" in df.columns:
        df["overall_survival"] = df["days_to_last_follow_up"]
    else:
        df["overall_survival"] = np.nan

    # lymph_node_involved_site — not in GDC; fill with placeholder
    if "lymph_node_involved_site" not in df.columns:
        df["lymph_node_involved_site"] = "not reported"

    # Ensure all required columns exist (fill missing with "not reported" / NaN)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = "not reported"

    # Keep only required columns
    df = df[REQUIRED_COLS].copy()

    # Encode treatment_outcome as int (0=complete, 1=partial, 2=stable, 3=progressive)
    outcome_map = {
        "complete response":  0,
        "partial response":   1,
        "stable disease":     2,
        "progressive disease": 3,
    }
    if "treatment_outcome" in df.columns:
        df["treatment_outcome"] = (
            df["treatment_outcome"]
            .str.lower()
            .str.strip()
            .map(outcome_map)
            .fillna(2)
            .astype(int)
        )

    # Normalise vital_status to lowercase
    if "vital_status" in df.columns:
        df["vital_status"] = df["vital_status"].str.lower().str.strip().fillna("not reported")

    # Drop rows with no overall_survival (critical target)
    df = df[df["overall_survival"].notna()]
    df["overall_survival"] = pd.to_numeric(df["overall_survival"], errors="coerce")
    df = df[df["overall_survival"].notna()]

    print(f"[+] Clean records after filtering: {len(df):,}")
    return df


# =============================================================================
# Split into 4 client datasets
# =============================================================================

def split_into_clients(df: pd.DataFrame, n_clients: int = 4, output_dir: str = "data"):
    """
    Split the cleaned DataFrame into `n_clients` roughly equal parts
    stratified by vital_status, and save each as a CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save full dataset
    full_path = os.path.join(output_dir, "full_dataset.csv")
    df.to_csv(full_path, index=False)
    print(f"[+] Full dataset saved → {full_path}  ({len(df):,} rows)")

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into n_clients equal parts
    splits = np.array_split(df, n_clients)

    paths = []
    for i, split_df in enumerate(splits, start=1):
        path = os.path.join(output_dir, f"client_{i}.csv")
        split_df.to_csv(path, index=False)
        paths.append(path)
        print(f"    client_{i}.csv → {len(split_df):,} rows  →  {path}")

    print(f"\n[✓] {n_clients} client datasets ready in '{output_dir}/'")
    return paths


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download TCGA & split for FL clients")
    parser.add_argument("--max-cases",  type=int, default=10000, help="Max cases to download (default 10000)")
    parser.add_argument("--n-clients",  type=int, default=4,     help="Number of client splits (default 4)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory (default: data/)")
    args = parser.parse_args()

    # 1. Download
    raw = download_tcga(max_cases=args.max_cases)

    if raw.empty:
        print("[!] No data downloaded. Check your internet connection.")
        exit(1)

    # 2. Clean
    clean = clean_and_align(raw)

    # 3. Split
    paths = split_into_clients(clean, n_clients=args.n_clients, output_dir=args.output_dir)

    print("\n" + "=" * 60)
    print("NEXT STEPS — send each teammate their file + this command:")
    print("=" * 60)
    for i, path in enumerate(paths, start=1):
        print(f"\n  Teammate {i}:  {path}")
        print("  python client/client_app.py \\")
        print("    --server  https://YOUR_NGROK_URL \\")
        print(f"    --username hospital_{i} \\")
        print(f"    --password secret{i} \\")
        print(f"    --hospital \"Hospital {i}\" \\")
        print(f"    --email admin{i}@hospital.org \\")
        print(f"    --csv {path} \\")
        print("    --proj 193b8223-311e-4de4-809d-68d431da46ab")
