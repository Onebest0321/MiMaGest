"""
preprocessing.py — preprocessing pipeline integrated into the capture software
==========================================================================

Transforms the raw recordings written by app.py
    data/PXX/<prefix>_finger_<slot>.csv   (one per TapStrap2)
    data/PXX/<prefix>_imu_<slot>.csv      (one per device)
    data/PXX/<prefix>_markers.csv
(<slot> = TAP1, TAP2, TAP3…: any device can occupy any slot)
into a single aligned recording, ready for recognition:
    <out>/Gesture_XX/PXX/<prefix>_Clean.csv

The algorithms are those of Celaningfinali_v3.py (no numerical
change); what changes:
  * no interaction (input/print/os.startfile) → importable from Flask;
  * each repetition returns quality statistics (clock regression
    slope, residual, jitter before/after, native frequency,
    duplicates, filled NaNs) → CSV/JSON report usable directly;
  * progress + cancellation → usable as a background task.

Steps: 1 clock regression · 2 START/STOP cut · 3 resampling
on a common grid · 4 merge.
(Normalization is done by the classifier, per fold, on train data only.)
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
import re
import shutil
import threading
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import interpolate

log = logging.getLogger("preprocessing")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FREQ_HZ = 190          # value retained in the thesis (< 200/208 Hz native)
DEFAULT_METHOD = "linear"      # "linear" | "pchip" | "akima"
METHODS = ("linear", "pchip", "akima")
NON_DATA_COLUMNS = ("Timestamp_LSL", "Timestamp_LSL_raw", "Timestamp_HW", "Type")

# Ex : P01_S01_Gesture_01_rep01_20260520_152408
NAME_PATTERN = re.compile(
    r"^(?P<participant>P\d+)_S(?P<session>\d+)_(?P<gesture>Gesture_\d+)_rep(?P<rep>\d+)_"
)
PARTICIPANT_DIR = re.compile(r"^P\d+$")

REPORT_FIELDS = [
    "key", "participant", "gesture", "repetition", "status", "error",
    "freq_hz", "method", "duration_s", "grid_points", "nan_filled", "n_warnings",
    "sensor", "rows_raw", "rows_kept", "native_hz", "slope_a", "intercept_b",
    "residual_std_ms", "jitter_before_ms", "jitter_after_ms", "dup_removed",
    "arrival_delay_mean_ms", "arrival_delay_p95_ms", "arrival_delay_max_ms",
    "out_path",
]


class Cancelled(Exception):
    """Raised when the examiner cancels the processing. `results` = repetitions already processed (for the report)."""
    def __init__(self, results=None):
        super().__init__()
        self.results = results or []


# ─────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
def _key_from_path(path: str) -> str:
    name = os.path.basename(path).replace(".csv", "")
    m = re.search(r"^(.*?)(?:_finger_|_imu_|_markers)", name)
    return m.group(1) if m else name


def list_participants(data_root: str) -> List[str]:
    """Folders P01, P02, … present (anomalie/test folders are ignored)."""
    if not os.path.isdir(data_root):
        return []
    return sorted(d for d in os.listdir(data_root)
                  if PARTICIPANT_DIR.match(d) and os.path.isdir(os.path.join(data_root, d)))


def list_repetitions(folder: str) -> List[str]:
    """Repetition keys that have markers and at least one sensor stream (fingers or IMU) in `folder` (flat).
    A configuration without TapStrap2 (IMU only) or without IMU is therefore accepted."""
    groups: Dict[str, Dict[str, list]] = {}
    for kind, pat in (("finger", "*_finger_*.csv"), ("imu", "*_imu_*.csv"),
                      ("markers", "*_markers.csv")):
        for p in glob.glob(os.path.join(folder, pat)):
            groups.setdefault(_key_from_path(p), {"finger": [], "imu": [], "markers": []})[kind].append(p)
    return sorted(k for k, g in groups.items() if (g["finger"] or g["imu"]) and g["markers"])


def parse_name(key: str) -> Optional[dict]:
    m = NAME_PATTERN.match(key)
    if not m:
        return None
    return {"participant": m.group("participant"), "gesture": m.group("gesture"),
            "repetition": f"rep{m.group('rep')}"}


def clean_path(out_root: str, key: str) -> Optional[str]:
    info = parse_name(key)
    if not info:
        return None
    return os.path.join(out_root, info["gesture"], info["participant"], f"{key}_Clean.csv")


def stream_sets(data_root: str, participants: List[str]) -> Dict[str, tuple]:
    """repetition key → sensor streams present (e.g. ('finger_TAP1', 'imu_TAP1', 'imu_TAP2')), from the file names."""
    out: Dict[str, set] = {}
    for p in participants:
        folder = os.path.join(data_root, p)
        for pat in ("*_finger_*.csv", "*_imu_*.csv"):
            for path in glob.glob(os.path.join(folder, pat)):
                out.setdefault(_key_from_path(path), set()).add(stream_name(os.path.basename(path).replace(".csv", "")))
    return {k: tuple(sorted(v)) for k, v in out.items()}


def stream_differences(sets: Dict[str, tuple]) -> Optional[dict]:
    """Compares the streams of the repetitions. None if everyone has the same; otherwise
    {reference, n_reference, odd:[{key, missing, extra}]} (reference = the most frequent set).
    Information only: these repetitions are cleaned anyway."""
    from collections import Counter
    if len(set(sets.values())) <= 1:
        return None
    ref, n = Counter(sets.values()).most_common(1)[0]
    odd = [{"key": k, "missing": sorted(set(ref) - set(v)), "extra": sorted(set(v) - set(ref))}
           for k, v in sorted(sets.items()) if v != ref]
    return {"reference": list(ref), "n_reference": n, "odd": odd}


def format_stream_notice(diff: dict, limit: int = 6) -> str:
    """Notification text (one repetition per line, truncated)."""
    head = (f"⚠ {len(diff['odd'])} répétition(s) n'ont pas les mêmes flux que les {diff['n_reference']} autres "
            f"(référence : {', '.join(diff['reference'])}). Elles sont nettoyées quand même, mais leurs colonnes "
            f"diffèrent :")
    lines = []
    for o in diff["odd"][:limit]:
        det = ([f"manque {', '.join(o['missing'])}"] if o["missing"] else []) + ([f"en plus {', '.join(o['extra'])}"] if o["extra"] else [])
        lines.append(f"  · {o['key']} — {' ; '.join(det)}")
    if len(diff["odd"]) > limit:
        lines.append(f"  · … et {len(diff['odd']) - limit} autre(s)")
    return "\n".join([head] + lines)


def _load_repetition(folder: str, key: str):
    """Loads the streams of a repetition. A stream with no row at all (header only) is ignored and
    reported as a warning instead of failing the whole repetition (an empty file can come from
    a copy, a disconnected device or a third-party tool).
    → (sensors, markers, ignored_streams)"""
    sensors: Dict[str, dict] = {}
    skipped: List[str] = []
    for kind, pat in (("finger", "*_finger_*.csv"), ("imu", "*_imu_*.csv")):
        for p in sorted(glob.glob(os.path.join(folder, pat))):
            if _key_from_path(p) == key:
                name = os.path.basename(p).replace(".csv", "")
                df = pd.read_csv(p)
                if df.empty:
                    skipped.append(stream_name(name))
                else:
                    sensors[name] = {"df": df, "type": kind}
    marker_paths = [p for p in glob.glob(os.path.join(folder, "*_markers.csv")) if _key_from_path(p) == key]
    if not sensors or not marker_paths:
        raise FileNotFoundError(f"Fichiers incomplets ou sans données pour {key}")
    return sensors, pd.read_csv(marker_paths[0]), skipped


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CLOCK REGRESSION   LSL_corr = a·HW + b
# ─────────────────────────────────────────────────────────────────────────────
def spread_packet_timestamps(hw):
    """Samples of the same BLE packet share the SAME hardware timestamp. They are spread
    evenly between this timestamp and that of the next packet (last packet: same step as the previous one).
    Backward jumps are left untouched."""
    hw = np.asarray(hw, dtype=float)
    n = len(hw)
    if n < 2:
        return hw.copy()
    starts = np.flatnonzero(np.r_[True, np.diff(hw) != 0])
    counts = np.diff(np.r_[starts, n])
    T = hw[starts]
    step = np.empty(len(starts))
    step[:-1] = T[1:] - T[:-1]
    step[-1] = step[-2] if len(starts) > 1 else 0.0
    step = np.where(step > 0, step, 0.0)
    pos = np.arange(n) - np.repeat(starts, counts)
    return np.repeat(T, counts) + np.repeat(step, counts) * pos / np.repeat(counts, counts)


def step1_clock_regression(sensors: Dict[str, dict], stats: Dict[str, dict],
                           fix_slope: bool = False) -> Dict[str, dict]:
    """fix_slope=False : fitted line LSL = a·HW + b (a and b estimated).
    fix_slope=True  : a = 1 imposed (both clocks run at the same speed), only the offset
                      b = mean(LSL − HW) is estimated (least squares with fixed slope)."""
    out = {}
    for name, data in sensors.items():
        df = data["df"].copy()
        s = {"rows_raw": int(len(df))}
        if "Timestamp_HW" not in df.columns or "Timestamp_LSL" not in df.columns or len(df) < 2:
            s["note"] = "pas de régression (colonnes manquantes ou < 2 lignes)"
            stats[name] = s
            out[name] = {**data, "df": df}
            continue

        hw = df["Timestamp_HW"].values
        lsl = df["Timestamp_LSL"].values
        if fix_slope:
            a, b = 1.0, float(np.mean(lsl - hw))
        else:
            a, b = np.polyfit(hw, lsl, 1)
        lsl_fit = a * hw + b                      # fitted line (sync statistics, unchanged)
        lsl_corr = a * spread_packet_timestamps(hw) + b   # corrected time: timestamps spread within each packet
        # LSL vs HW arrival delay: deviation from the line, brought to 0 for the earliest arrival
        delay_ms = (lsl - lsl_fit - np.min(lsl - lsl_fit)) * 1000

        s.update({
            "slope_a": float(a),
            "intercept_b": float(b),
            "residual_std_ms": float(np.std(lsl - lsl_fit) * 1000),
            "jitter_before_ms": float(np.std(np.diff(lsl)) * 1000),
            "jitter_after_ms": float(np.std(np.diff(lsl_corr)) * 1000),
            "arrival_delay_mean_ms": float(np.mean(delay_ms)),
            "arrival_delay_p95_ms": float(np.percentile(delay_ms, 95)),
            "arrival_delay_max_ms": float(np.max(delay_ms)),
        })
        df["Timestamp_LSL_raw"] = df["Timestamp_LSL"]
        df["Timestamp_LSL"] = lsl_corr
        stats[name] = s
        out[name] = {**data, "df": df}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CUT ON THE MARKERS
# ─────────────────────────────────────────────────────────────────────────────
def extract_markers(df_markers: pd.DataFrame):
    t_start = t_stop = None
    for _, row in df_markers.iterrows():
        marker, ts = str(row["Marker"]), float(row["Timestamp_LSL"])
        if marker.startswith("START") and not marker.startswith("COUNTDOWN"):
            t_start = ts
        elif marker.startswith("STOP"):
            t_stop = ts
    if t_start is None or t_stop is None:
        raise ValueError(f"Markers START/STOP introuvables ({df_markers['Marker'].tolist()})")
    return t_start, t_stop


def step2_trim(sensors: Dict[str, dict], df_markers: pd.DataFrame, stats: Dict[str, dict]):
    """Cuts and grids on the corrected clock a·HW+b (Timestamp_LSL from step 1).
    If the regression could not be done (missing columns or < 2 rows), Timestamp_LSL stays the arrival time."""
    t_start, t_stop = extract_markers(df_markers)
    out = {}
    for name, data in sensors.items():
        df = data["df"].copy()
        t = df["Timestamp_LSL"]
        mask = (t >= t_start) & (t <= t_stop)
        if not mask.any():
            raise ValueError(f"{name} : aucune ligne dans [{t_start:.6f}, {t_stop:.6f}] (Timestamp_LSL corrigé)")
        df = df[mask].copy()
        df["Timestamp_LSL"] = t[mask].values - t_start   # t = 0 au START
        stats.setdefault(name, {})["rows_kept"] = int(len(df))
        out[name] = {**data, "df": df}
    return out, t_start, t_stop


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — RESAMPLING ON A COMMON GRID
# ─────────────────────────────────────────────────────────────────────────────
def _interpolate(t, values, grid, method):
    if method == "linear":
        return interpolate.interp1d(t, values, kind="linear", bounds_error=False, fill_value=np.nan)(grid)
    if method == "pchip":
        return interpolate.PchipInterpolator(t, values, extrapolate=False)(grid)
    if method == "akima":
        return interpolate.Akima1DInterpolator(t, values)(grid, extrapolate=False)
    raise ValueError(f"Méthode d'interpolation inconnue : {method}")


def step3_resample(sensors, t_start, t_stop, freq_hz, method, stats, warnings: List[str]):
    duration = t_stop - t_start
    dt = 1.0 / freq_hz
    n_pts = int(np.floor(duration / dt)) + 1
    grid = np.linspace(0, duration, n_pts)
    nan_total = 0
    out = {}

    for name, data in sensors.items():
        df = data["df"].copy()
        t = df["Timestamp_LSL"].values
        s = stats.setdefault(name, {})

        if len(t) > 1:
            # MEAN rate (n-1)/duration: the median of the Δt is skewed when measurements arrive in packets
            span = t[-1] - t[0]
            native = (len(t) - 1) / span if span > 0 else float("inf")
            s["native_hz"] = float(native)
            if freq_hz > native * 1.5:
                warnings.append(f"{name} : ~{native:.0f} Hz natif, suréchantillonné à {freq_hz} Hz")
            elif native > freq_hz * 1.5:
                warnings.append(f"{name} : ~{native:.0f} Hz natif, sous-échantillonné à {freq_hz} Hz")

        dup = 0
        if len(t) > 1 and np.any(np.diff(t) <= 0):
            keep = np.concatenate(([True], np.diff(t) > 0))
            dup = int((~keep).sum())
            t = t[keep]
            df = df.loc[keep].reset_index(drop=True)
            warnings.append(f"{name} : {dup} timestamps non croissants supprimés")
        s["dup_removed"] = dup

        df_out = pd.DataFrame({"Timestamp_LSL": grid})
        for col in [c for c in df.columns if c not in NON_DATA_COLUMNS]:
            df_out[col] = _interpolate(t, df[col].values.astype(float), grid, method)

        n_nan = int(df_out.isnull().sum().sum())
        if n_nan:
            df_out = df_out.ffill().bfill()
        nan_total += n_nan
        out[name] = {**data, "df": df_out}

    return out, grid, nan_total


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MERGE
# ─────────────────────────────────────────────────────────────────────────────
def stream_name(full_name: str) -> str:
    """'…_imu_TAP1' → 'imu_TAP1' (distinguishes fingers and IMU of the same Tap)."""
    m = re.search(r"_((?:imu|finger)_.+)$", full_name)
    return m.group(1) if m else full_name


def short_sensor_name(full_name: str) -> str:
    m = re.search(r"_(?:imu|finger)_(.+)$", full_name)
    return m.group(1) if m else full_name


def step4_merge(sensors, grid, ident: dict) -> pd.DataFrame:
    """All sensors are already on `grid`: simple column concatenation
    (equivalent to the V3 join on Timestamp_LSL, without the cost or the
    risk of a join on floats)."""
    parts = [pd.DataFrame({"Timestamp_LSL": grid})]
    for name, data in sensors.items():
        short = short_sensor_name(name)
        df = data["df"].drop(columns=["Timestamp_LSL"])
        parts.append(df.rename(columns={c: f"{short}_{c}" for c in df.columns}))
    df_final = pd.concat(parts, axis=1)
    df_final.insert(0, "Participant", ident["participant"])
    df_final.insert(1, "Gesture", ident["gesture"])
    df_final.insert(2, "Repetition", ident["repetition"])
    return df_final


# ─────────────────────────────────────────────────────────────────────────────
# ONE REPETITION END TO END
# ─────────────────────────────────────────────────────────────────────────────
def process_repetition(folder: str, key: str, out_root: str,
                       freq_hz: int = DEFAULT_FREQ_HZ, method: str = DEFAULT_METHOD,
                       overwrite: bool = False, keep_intermediate: bool = False,
                       fix_slope: bool = False) -> dict:
    """Processes one repetition. Never raises: returns a result dict."""
    ident = parse_name(key)
    res = {"key": key, "status": "error", "error": "", "warnings": [], "sensors": {},
           "freq_hz": freq_hz, "method": method, "out_path": None,
           "participant": ident["participant"] if ident else "Unknown",
           "gesture": ident["gesture"] if ident else "Unknown",
           "repetition": ident["repetition"] if ident else "Unknown"}
    try:
        if not ident:
            raise ValueError(f"Nom inattendu (Pxx_Sxx_Gesture_xx_repxx_…) : {key}")

        target = clean_path(out_root, key)
        res["out_path"] = target
        if os.path.isfile(target) and not overwrite:
            res["status"] = "skipped"
            return res

        sensors, df_markers, skipped = _load_repetition(folder, key)
        if skipped:
            res["warnings"].append("flux vide(s) ignoré(s) : " + ", ".join(skipped))
        stats: Dict[str, dict] = {}

        s1 = step1_clock_regression(sensors, stats, fix_slope)
        s2, t_start, t_stop = step2_trim(s1, df_markers, stats)
        s3, grid, nan_filled = step3_resample(s2, t_start, t_stop, freq_hz, method, stats, res["warnings"])
        df_final = step4_merge(s3, grid, ident)

        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".tmp"                    # atomic write: a sudden stop never leaves a truncated _Clean.csv
        df_final.to_csv(tmp, index=False)
        os.replace(tmp, target)

        if keep_intermediate:
            inter = os.path.join(out_root, "_intermediate", key)
            for tag, group in (("step1", s1), ("step2", s2), ("step3", s3)):
                os.makedirs(os.path.join(inter, tag), exist_ok=True)
                for name, d in group.items():
                    d["df"].to_csv(os.path.join(inter, tag, f"{name}_{tag}.csv"), index=False)

        res.update({"status": "ok", "sensors": stats, "duration_s": float(t_stop - t_start),
                    "grid_points": int(len(grid)), "nan_filled": int(nan_filled)})
    except Cancelled:
        raise
    except Exception as exc:  # noqa: BLE001 — the error is reported in the report
        res["error"] = f"{type(exc).__name__}: {exc}"
        log.warning("Échec %s : %s", key, res["error"])
    return res


# ─────────────────────────────────────────────────────────────────────────────
# LOT : PLUSIEURS PARTICIPANTS (progression + annulation)
# ─────────────────────────────────────────────────────────────────────────────
def count_pending(data_root: str, out_root: str, participants: List[str]) -> dict:
    """For the UI: raw and already-cleaned repetitions per participant."""
    info = {}
    for p in participants:
        reps = list_repetitions(os.path.join(data_root, p))
        done = sum(1 for k in reps if (clean_path(out_root, k) and os.path.isfile(clean_path(out_root, k))))
        info[p] = {"raw": len(reps), "clean": done}
    return info


def default_workers() -> int:
    return max(1, min((os.cpu_count() or 2) - 1, 8))


def run_batch(data_root: str, out_root: str, participants: List[str],
              freq_hz: int = DEFAULT_FREQ_HZ, method: str = DEFAULT_METHOD,
              overwrite: bool = False, keep_intermediate: bool = False,
              fix_slope: bool = False,
              progress: Optional[Callable[[dict], None]] = None,
              cancel: Optional[threading.Event] = None,
              workers: Optional[int] = None) -> List[dict]:
    """Processes all repetitions of the given participants.

    workers > 1: repetitions processed in parallel (processes) — each repetition
    is independent, the gain is almost linear. Automatic fallback to sequential
    if the process pool is not available."""
    if method not in METHODS:
        raise ValueError(f"method doit être dans {METHODS}")
    plan = [(p, k) for p in participants for k in list_repetitions(os.path.join(data_root, p))]
    workers = default_workers() if workers is None else max(1, int(workers))
    results: List[dict] = []
    done_keys = set()

    def emit(r):
        results.append(r)
        done_keys.add(r["key"])
        if progress:
            progress({"done": len(results), "total": len(plan), "current": r["key"], "last": r})

    if workers > 1 and len(plan) > 3:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool
        ex = ProcessPoolExecutor(max_workers=workers)
        try:
            futs = [ex.submit(process_repetition, os.path.join(data_root, p), k, out_root,
                              freq_hz, method, overwrite, keep_intermediate, fix_slope) for p, k in plan]
            for fut in as_completed(futs):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                emit(fut.result())
        except Cancelled:
            ex.shutdown(wait=True, cancel_futures=True)       # in-progress repetitions finish cleanly
            for fu in futs:                                    # …and are kept for the report
                if fu.done() and not fu.cancelled():
                    try:
                        r = fu.result()
                    except Exception:
                        continue
                    if r["key"] not in done_keys:
                        results.append(r); done_keys.add(r["key"])
            raise Cancelled(results)
        except BrokenProcessPool:
            log.warning("Pool de processus indisponible → repli séquentiel")
        finally:
            ex.shutdown(wait=True, cancel_futures=True)

    for p, k in plan:                       # sequential (or resume after fallback)
        if k in done_keys:
            continue
        if cancel is not None and cancel.is_set():
            raise Cancelled(results)
        emit(process_repetition(os.path.join(data_root, p), k, out_root, freq_hz, method,
                                overwrite, keep_intermediate, fix_slope))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY REPORT  (one row per repetition × sensor)
# ─────────────────────────────────────────────────────────────────────────────
def write_report(results: List[dict], out_root: str, merge_existing: bool = True) -> dict:
    """Writes preprocessing_report.csv and preprocessing_summary.json at the output
    root. Merges with a previous report (incremental processing)."""
    os.makedirs(out_root, exist_ok=True)
    csv_path = os.path.join(out_root, "preprocessing_report.csv")

    rows = []
    for r in results:
        if r["status"] == "skipped":
            continue
        base = {f: r.get(f, "") for f in REPORT_FIELDS if f not in ("sensor",)}
        base["n_warnings"] = len(r["warnings"])
        if r["sensors"]:
            for sname, s in r["sensors"].items():
                row = dict(base, sensor=stream_name(sname))
                for f in ("rows_raw", "rows_kept", "native_hz", "slope_a", "intercept_b",
                          "residual_std_ms", "jitter_before_ms", "jitter_after_ms", "dup_removed",
                          "arrival_delay_mean_ms", "arrival_delay_p95_ms", "arrival_delay_max_ms"):
                    row[f] = s.get(f, "")
                rows.append(row)
        else:
            rows.append(dict(base, sensor=""))

    df_new = pd.DataFrame(rows, columns=REPORT_FIELDS)
    if merge_existing and os.path.isfile(csv_path) and not df_new.empty:
        old = pd.read_csv(csv_path)
        if "key" not in old.columns:
            old["key"] = ""
        old["key"] = old["key"].fillna("")
        new_keys = set(df_new["key"])
        old = old[~old["key"].isin(new_keys)]
        old = old.reindex(columns=REPORT_FIELDS)
        df_new = pd.concat([old, df_new], ignore_index=True)
    if not df_new.empty or not os.path.isfile(csv_path):
        df_new.to_csv(csv_path, index=False)

    summary = summarize(pd.read_csv(csv_path)) if os.path.isfile(csv_path) else {}
    with open(os.path.join(out_root, "preprocessing_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def summarize(df: pd.DataFrame) -> dict:
    """Aggregates for the dashboard (and for the Evaluation section of the paper)."""
    if df.empty:
        return {"repetitions": 0}
    id_cols = ["key"] if "key" in df.columns and df["key"].fillna("").ne("").all() \
        else ["participant", "gesture", "repetition"]
    reps = df.drop_duplicates(id_cols)
    ok = reps[reps["status"] == "ok"]
    out = {
        "repetitions": int(len(reps)),
        "ok": int(len(ok)),
        "errors": int((reps["status"] == "error").sum()),
        "repetitions_with_warnings": int((ok["n_warnings"] > 0).sum()) if len(ok) else 0,
        "repetitions_with_nan_filled": int((ok["nan_filled"] > 0).sum()) if len(ok) else 0,
        "sensors": {},
    }
    d = df[df["status"] == "ok"]
    for sensor, g in d.groupby("sensor"):
        if not sensor:
            continue

        def q(col):
            v = pd.to_numeric(g[col], errors="coerce").dropna()
            return {"mean": float(v.mean()), "std": float(v.std(ddof=0)),
                    "p50": float(v.quantile(.5)), "p95": float(v.quantile(.95)),
                    "max": float(v.max())} if len(v) else None

        out["sensors"][sensor] = {
            "n": int(len(g)),
            "residual_std_ms": q("residual_std_ms"),
            "jitter_before_ms": q("jitter_before_ms"),
            "jitter_after_ms": q("jitter_after_ms"),
            "native_hz": q("native_hz"),
            "slope_a": q("slope_a"),
            "dup_removed_total": int(pd.to_numeric(g["dup_removed"], errors="coerce").fillna(0).sum()),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Command line (same module, without going through Flask)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Prétraitement des enregistrements TapStrap")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data_clean")
    ap.add_argument("--freq", type=int, default=DEFAULT_FREQ_HZ)
    ap.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
    ap.add_argument("--participants", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-intermediate", action="store_true")
    ap.add_argument("--fix-slope", action="store_true", help="impose a = 1 (n'estime que le décalage b) au lieu d'ajuster la pente")
    ap.add_argument("--workers", type=int, default=None, help="processus en parallèle (défaut : auto)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    parts = a.participants or list_participants(a.data)
    res = run_batch(a.data, a.out, parts, a.freq, a.method, a.overwrite, a.keep_intermediate,
                    fix_slope=a.fix_slope, workers=a.workers, progress=lambda p: print(f"[{p['done']}/{p['total']}] {p['current']} → {p['last']['status']}"))
    print(json.dumps(write_report(res, a.out), indent=2, ensure_ascii=False))
