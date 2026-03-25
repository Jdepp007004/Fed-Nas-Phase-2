"""
shared/model_schema.py
M1: Canonical feature list and model configuration constants.
Shared between server and client to ensure identical feature spaces.
"""

# ─── Model Architecture Constants ────────────────────────────────────────────

INPUT_DIM = 512  # Number of input features (post-preprocessing TCGA)
MAX_DEPTH = 6  # Maximum backbone layers (range 2–8)
HIDDEN_DIM = 256  # Width of each hidden layer
NUM_TOXICITY_CLASSES = 4  # Toxicity severity classes (grade 1–4)

DEFAULT_ACTIVE_DEPTH = 4  # Default NAS depth assigned before profiling

# ─── Default Task Loss Weights ────────────────────────────────────────────────

DEFAULT_TASK_WEIGHTS = {
    "regression": 1.0,
    "toxicity": 0.8,
    "binary": 0.6,
}

# ─── Training Defaults ────────────────────────────────────────────────────────

DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_FEDPROX_MU = 0.01
DEFAULT_CLIP_NORM = 1.0
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_MOMENTUM_BETA = 0.9

# ─── Required TCGA Columns & Schema ──────────────────────────────────────────

# This is the canonical list of TCGA Clinical Subset columns expected by the model.
# Client CSVs must contain at least these columns (after preprocessing they are
# encoded / normalised into INPUT_DIM features).

REQUIRED_COLUMNS = [
    "age_at_diagnosis",
    "gender",
    "race",
    "ethnicity",
    "vital_status",
    "days_to_death",
    "days_to_last_follow_up",
    "cancer_type",
    "tumor_stage",
    "tumor_grade",
    "treatment_type",
    "prior_malignancy",
    "prior_treatment",
    "tissue_or_organ_of_origin",
    "morphology",
    "site_of_resection_or_biopsy",
    "classification_of_primary_diagnosis",
    "tumor_focality",
    "tumor_largest_dimension",
    "lymph_node_involved_site",
    "metastasis_at_diagnosis",
    "synchronous_malignancy",
    "year_of_diagnosis",
    "ann_arbor_b_symptoms",
    "ann_arbor_clinical_stage",
    "burkitt_lymphoma_clinical_variant",
    "cause_of_death",
    "circumferential_resection_margin",
    "colon_polyps_history",
    "days_to_birth",
    "days_to_diagnosis",
    "days_to_hiv_diagnosis",
    "days_to_new_event",
    "figo_stage",
    "hiv_positive",
    "hpv_positive_type",
    "hpv_status_by_p16_testing",
    "iss_stage",
    "laterality",
    "ldh_level_at_diagnosis",
    "ldh_normal_range_upper",
    "masaoka_stage",
    "new_event_anatomic_site",
    "new_event_type",
    "overall_survival",
    "perineural_invasion_present",
    "primary_diagnosis",
    "primary_gleason_grade",
    "secondary_gleason_grade",
    "weight",
    "tobacco_smoking_onset_year",
    "tobacco_smoking_quit_year",
    "type_of_tobacco_used",
    "alcohol_history",
    "alcohol_intensity",
    "bmi",
    "height",
    "weeks_gestation_at_birth",
    "treatment_outcome",
    "treatment_or_therapy",
    "days_to_treatment_start",
    "days_to_treatment_end",
    "initial_disease_status",
    "regimen_or_line_of_therapy",
    "therapeutic_agents",
    "number_of_cycles",
]

# Target column definitions
TARGET_COLUMNS = {
    "regression": "overall_survival",       # Continuous survival days
    "toxicity": "treatment_outcome",        # Ordinal toxicity grade (0–3)
    "binary": "vital_status",              # 0 = alive, 1 = deceased
}

# Numerical columns and their expected [min, max] bounds for normalisation
FEATURE_RANGES = {
    "age_at_diagnosis": [0, 120],
    "days_to_death": [0, 30000],
    "days_to_last_follow_up": [0, 30000],
    "tumor_largest_dimension": [0, 50],
    "days_to_birth": [-50000, 0],
    "days_to_diagnosis": [0, 30000],
    "days_to_new_event": [0, 30000],
    "ldh_level_at_diagnosis": [0, 10000],
    "ldh_normal_range_upper": [0, 1000],
    "weight": [0, 300],
    "bmi": [10, 80],
    "height": [50, 250],
    "number_of_cycles": [0, 100],
    "year_of_diagnosis": [1900, 2030],
    "days_to_treatment_start": [0, 10000],
    "days_to_treatment_end": [0, 30000],
    "overall_survival": [0, 30000],
}

# Categorical columns and their known category sets
CATEGORICAL_VALUES = {
    "gender": ["male", "female", "not reported", "unknown"],
    "vital_status": ["alive", "dead", "not reported"],
    "tumor_stage": [
        "stage i", "stage ia", "stage ib",
        "stage ii", "stage iia", "stage iib", "stage iic",
        "stage iii", "stage iiia", "stage iiib", "stage iiic",
        "stage iv", "stage iva", "stage ivb", "stage ivc",
        "not reported", "unknown",
    ],
    "treatment_type": [
        "chemotherapy", "radiation therapy", "immunotherapy",
        "targeted molecular therapy", "surgery", "pharmaceutical therapy",
        "hormone therapy", "not reported", "unknown",
    ],
    "treatment_or_therapy": ["yes", "no", "not reported", "unknown"],
    "initial_disease_status": [
        "initial primary diagnosis",
        "progression or recurrence",
        "not reported",
    ],
}

# Expected column types (for schema validation)
COLUMN_TYPES = {
    "age_at_diagnosis": "float",
    "gender": "str",
    "race": "str",
    "vital_status": "str",
    "days_to_death": "float",
    "days_to_last_follow_up": "float",
    "cancer_type": "str",
    "tumor_stage": "str",
    "tumor_grade": "str",
    "treatment_type": "str",
    "overall_survival": "float",
    "treatment_outcome": "int",
}

MIN_SAMPLES = 100  # Minimum rows required to participate

# ─── Server → Client Schema Dict (sent via API) ──────────────────────────────

SERVER_SCHEMA = {
    "required_columns": REQUIRED_COLUMNS,
    "target_columns": TARGET_COLUMNS,
    "feature_ranges": FEATURE_RANGES,
    "categorical_values": CATEGORICAL_VALUES,
    "column_types": COLUMN_TYPES,
    "encoding_maps": {},           # populated at runtime
    "min_samples": MIN_SAMPLES,
    "schema_version": "1.0.0",
}

# ─── Model Config Dict (passed to Supernet constructor) ──────────────────────

MODEL_CONFIG = {
    "input_dim": INPUT_DIM,
    "max_depth": MAX_DEPTH,
    "hidden_dim": HIDDEN_DIM,
    "num_toxicity_classes": NUM_TOXICITY_CLASSES,
}
