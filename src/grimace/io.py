"""Loading of the overview sheet and per-recording zips (read in place, never extracted)."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from .config import load_config

# e.g. "2025-08-11_RHmigraine_Stimulering A1 (83).MP4_<uuid>.zip"
#      "2025-08-06_RHmigraine_Baseline B1 (91), omkorsel.MP4_<uuid>.zip"
_NAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_RHmigraine_(?P<session_raw>Baseline|Stimulering)\s+"
    r"(?P<id>[A-Za-z]\d+)\s*\((?P<animal_no>\d+)\)(?P<note>[^.]*)\.MP4_(?P<uuid>[0-9a-f-]+)\.zip$"
)
_SESSION_MAP = {"Baseline": "baseline", "Stimulering": "stimulated"}

OVERVIEW_RENAME = {
    "Animal no": "animal_no",
    "ID": "id",
    "Cohorte": "cohort",
    "Sex": "sex",
    "LEV/VEH": "drug",
    "Treatment": "treatment",
    "Genotype": "genotype",
    "Weight": "weight",
    "Baseline Mean (average grimace score)": "xl_baseline_mean_avg",
    "Baseline Mean (total grimace score)": "xl_baseline_mean_total",
    "Baseline Counted frames": "xl_baseline_n_frames",
    "Test Mean (average grimace score)": "xl_test_mean_avg",
    "Test Baseline Mean (total grimace score)": "xl_test_mean_total",
    "Test Counted frames": "xl_test_n_frames",
}


def load_metadata(cfg: dict | None = None) -> pd.DataFrame:
    """One row per mouse from the overview Excel (columns renamed to snake_case)."""
    cfg = cfg or load_config()
    df = pd.read_excel(cfg["overview_path"]).rename(columns=OVERVIEW_RENAME)
    df["animal_no"] = df["animal_no"].astype(int)
    return df


def parse_zip_name(path: Path) -> dict:
    m = _NAME_RE.match(path.name)
    if m is None:
        return {"path": path, "parsed": False}
    d = m.groupdict()
    return {
        "path": path,
        "parsed": True,
        "date": pd.Timestamp(d["date"]),
        "session": _SESSION_MAP[d["session_raw"]],
        "id": d["id"],
        "animal_no": int(d["animal_no"]),
        "note": d["note"].strip(" ,"),
        "uuid": d["uuid"],
    }


def build_manifest(cfg: dict | None = None) -> pd.DataFrame:
    """One row per recording zip on disk, with parsed name fields and mouse metadata merged in."""
    cfg = cfg or load_config()
    rows = []
    for folder in (cfg["baseline_path"], cfg["stimulated_path"]):
        rows += [parse_zip_name(p) for p in sorted(folder.glob("*.zip"))]
    man = pd.DataFrame(rows)
    meta = load_metadata(cfg).drop(columns=["id"])
    man = man.merge(meta, on="animal_no", how="left", validate="m:1")
    man["in_overview"] = man["genotype"].notna()
    man["recording_id"] = (
        man["session"].astype(str) + "_" + man["animal_no"].astype("Int64").astype(str)
        + man["note"].fillna("").map(lambda s: "_" + s if s else "")
    )
    return man


def load_frames(zip_path: Path) -> pd.DataFrame:
    """Per-frame scores from the CSV inside one recording zip."""
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            df = pd.read_csv(f)
    return df


def load_acquisition_json(zip_path: Path) -> dict:
    """Acquisition metadata / raw model output (3 MB; only load when needed)."""
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".json"))
        with z.open(name) as f:
            return json.load(f)


def load_analysis_frames(cfg: dict | None = None) -> pd.DataFrame:
    """Frames of the 71 Excel mice only (recordings of animals not in the sheet are dropped), sorted in time.

    Adds the mouse metadata, time `t` (s), a boolean `miss_<unit>` per action unit and `complete`
    (all four units scorable). Score `-1` ("not scorable") is set to NaN in the unit columns.
    """
    cfg = cfg or load_config()
    aus = cfg["action_units"]
    meta = load_metadata(cfg)[["animal_no", "genotype", "drug", "treatment", "sex", "cohort", "weight"]]
    df = load_all_frames(cfg).merge(meta, on="animal_no", how="inner", validate="m:1")
    df = df.sort_values(["recording_id", "Frame Index"]).reset_index(drop=True)
    df["t"] = df["Timestamp(x)"]
    for a in aus:
        df[f"miss_{a}"] = df[a] < 0
        df.loc[df[f"miss_{a}"], a] = float("nan")
    df["complete"] = df[aus].notna().all(axis=1)
    return df


def load_all_frames(cfg: dict | None = None, use_cache: bool = True) -> pd.DataFrame:
    """All frames of all recordings in the manifest, tidy (one row per frame), cached as parquet."""
    cfg = cfg or load_config()
    cache = cfg["cache_path"] / "frames.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    man = build_manifest(cfg)
    man = man[man["parsed"]]
    parts = []
    for r in man.itertuples():
        df = load_frames(r.path)
        df.insert(0, "recording_id", r.recording_id)
        df.insert(1, "animal_no", r.animal_no)
        df.insert(2, "session", r.session)
        df.insert(3, "note", r.note)
        parts.append(df)
    frames = pd.concat(parts, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(cache)
    return frames
