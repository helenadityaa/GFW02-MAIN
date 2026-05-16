import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
WORK_DIR = DATA_DIR / "work"
OPENSAR_DIR = DATA_DIR / "opensarship_like"
METADATA_DIR = OPENSAR_DIR / "metadata"
XML_DIR = OPENSAR_DIR / "XML"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

DEFAULT_INPUT_CSV = (
    PROJECT_ROOT.parent / "output" / "GFW_SAR_EEZ_Indonesia_202602_202603.csv"
)

REQUIRED_DIRECTORIES = [
    DATA_DIR / "selected_samples",
    WORK_DIR / "downloaded_scenes",
    WORK_DIR / "preprocessed_scenes",
    OPENSAR_DIR / "Patch",
    OPENSAR_DIR / "Patch_Cal",
    OPENSAR_DIR / "Patch_Uint8",
    OPENSAR_DIR / "Patch_RGB",
    XML_DIR,
    METADATA_DIR,
    LOGS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
]

CATEGORY_MAP = {
    "unmatched": "Unmatched",
    "cargo": "Cargo",
    "fishing": "Fishing",
    "passenger": "Passenger",
    "other": "Other",
    "noisy_vessel": "NoisyVessel",
}


def ensure_directories(extra_paths=None):
    paths = list(REQUIRED_DIRECTORIES)
    if extra_paths:
        paths.extend(Path(path) for path in extra_paths)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_csv_checked(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    return pd.read_csv(path)


def require_columns(df, required_columns, file_label):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        print(f"ERROR: kolom wajib tidak ditemukan pada {file_label}: {missing}")
        print("Kolom tersedia:")
        for column in df.columns:
            print(f"  - {column}")
        raise ValueError(f"Kolom wajib tidak lengkap pada {file_label}")


def save_csv(df, path):
    path = Path(path)
    ensure_parent(path)
    df.to_csv(path, index=False)
    return path


def save_json_records(df, path):
    path = Path(path)
    ensure_parent(path)
    records = df.where(pd.notna(df), None).to_dict(orient="records")
    path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    return path


def write_text(path, text):
    path = Path(path)
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")
    return path


def timestamp_now_utc():
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp_utc(value):
    if pd.isna(value):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce", utc=True)


def sanitize_filename_part(value, fallback="Unknown"):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    parts = re.findall(r"[A-Za-z0-9]+", text)
    if not parts:
        return fallback
    return "".join(part[:1].upper() + part[1:] for part in parts)


def safe_path_stem(value, fallback="item"):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def category_name_from_matched_category(value):
    if pd.isna(value):
        return "Unknown"
    raw = str(value).strip()
    if not raw:
        return "Unknown"
    normalized = raw.lower()
    if normalized in CATEGORY_MAP:
        return CATEGORY_MAP[normalized]
    return sanitize_filename_part(normalized, fallback="Unknown")


def apply_limit(df, limit):
    if limit is None:
        return df
    if int(limit) < 0:
        return df
    return df.head(int(limit)).copy()


def save_failures(df, stage_name, status_column=None, failed_status="failed"):
    if status_column and status_column in df.columns:
        failures = df[df[status_column].astype(str).str.lower() == failed_status].copy()
    else:
        failures = pd.DataFrame(columns=df.columns)
    path = METRICS_DIR / f"{stage_name}_failures.csv"
    save_csv(failures, path)
    return path


def optional_read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def coerce_source_row_index(df):
    if "source_row_index" in df.columns:
        df = df.copy()
        df["source_row_index"] = df["source_row_index"].astype(str)
    return df


def add_missing_process_columns(df, columns):
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def remove_file_if_exists(path):
    path = Path(path)
    if path.exists() and path.is_file():
        path.unlink()


def parse_patch_polarization(path):
    stem = Path(path).stem.lower()
    if stem.endswith("_vv"):
        return "vv"
    if stem.endswith("_vh"):
        return "vh"
    return ""


def parse_patch_name(path):
    path = Path(path)
    stem = path.stem
    if stem.lower().endswith("_vv") or stem.lower().endswith("_vh"):
        return stem[:-3]
    return stem


def make_empty_manifest(path, columns):
    df = pd.DataFrame(columns=columns)
    save_csv(df, path)
    return df
