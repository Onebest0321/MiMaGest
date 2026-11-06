"""
classifier_v2.py — TapStrap classifier, logic of loso_pipeline.py / udep_pipeline.py (a single file + clf_keras_v2.py).

Sections: 1. data & folds · 2. metrics & cross-validation loop · 3. imported models ·
           4. compute process (worker) · 5. Flask routes.
TensorFlow is never loaded in Flask: training / evaluation run in a subprocess
(`python classifier_v2.py --worker <folder>`), only clf_keras_v2.py imports TensorFlow.
"""
import os, sys, re, io, csv, json, time, glob, shutil, threading, subprocess, traceback, importlib
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:                 # the worker runs with "python -I", which does not add the script folder
    sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import dataset_tools as _dt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# GLOBAL classifier space: models, runs, evaluations. Each item just keeps a "dataset" label.
STORE = os.environ.get("CLF_STORE") or os.path.join(HERE, "classifier_store")
if HERE not in sys.path:
    sys.path.insert(0, HERE)



# ████████████████████████████████████████████████████████████
#  1-2. DATA, FOLDS, METRICS, CROSS-VALIDATION
# ████████████████████████████████████████████████████████████

# ═══════════════════════════════════════════════════════════
#  DATA  (load_split_data.py adapted to data_clean/)
# ═══════════════════════════════════════════════════════════
FILENAME_REGEX = re.compile(
    r'^P(\d+)_S(\d+)_Gesture_(\d+)_rep(\d+)_(\d+)_(\d+)_Clean(?:_Norm)?\.csv$', re.IGNORECASE)
META_COLS = {'Participant', 'Gesture', 'Repetition', 'Timestamp_LSL'}
DEFAULT_EXCLUDED_POINTS = ['TAP1_Accel']      # OLD frozen default, kept for compatibility: no longer used (see default_excluded_points)
AUTO = "auto"                                 # exclure_points="auto" → default_excluded_points(clean_root)


class Cancelled(Exception):
    pass


def point_name_from_feature(feature_name):
    """'TAP1_Thumb_X' -> 'TAP1_Thumb' (groups the 3 X/Y/Z axes of a same point)."""
    return re.sub(r'_[XYZ]$', '', feature_name)


_IDX_CACHE = {}          # realpath -> (folder signature, index)
_NFILES = {}             # realpath -> number of *_Clean.csv files actually present (key duplicates included)


def _dir_signature(clean_root):
    """Cheap signature of a dataset: (path, mtime) of each Gesture_*/… folder (a few hundred entries
    instead of thousands of files). Adding / deleting / replacing a file changes its folder's date."""
    sig = []
    stack = [p for p in glob.glob(os.path.join(clean_root, "Gesture_*")) if os.path.isdir(p)]
    while stack:
        d = stack.pop()
        try:
            sig.append((d, os.stat(d).st_mtime_ns))
            with os.scandir(d) as it:
                stack.extend(e.path for e in it if e.is_dir())
        except OSError:
            pass
    return tuple(sorted(sig))


def index_clean_dir(clean_root):
    """{(user, gesture, id) -> path}. id = session*100 + rep. Only Gesture_*/P*/ are read
    (the _excluded, normalized_*, models, classification folders are ignored).
    Cached as long as the folders have not changed (the interface calls this several times per request)."""
    key = os.path.realpath(clean_root)
    sig = _dir_signature(clean_root)
    hit = _IDX_CACHE.get(key)
    if hit and hit[0] == sig:
        return dict(hit[1])
    index = {}
    nfiles = 0
    for gd in glob.glob(os.path.join(clean_root, "Gesture_*")):
        if not os.path.isdir(gd):
            continue
        for root, _dirs, files in os.walk(gd):
            for fname in files:
                m = FILENAME_REGEX.match(fname)
                if not m:
                    continue
                participant, session, gesture, rep = m.group(1), m.group(2), m.group(3), m.group(4)
                nfiles += 1
                index[(f'P{int(participant):02d}', f'Gesture_{int(gesture):02d}', int(session) * 100 + int(rep))] = \
                    os.path.join(root, fname)
    _IDX_CACHE[key] = (sig, index)
    _NFILES[key] = nfiles
    return dict(index)


def scan_clean_dir(clean_root, with_columns=True):
    """Quick inventory (file names only) for the interface. with_columns=False: without reading the CSV headers
    (dataset list: only counts are wanted)."""
    idx = index_clean_dir(clean_root)
    gn = lambda s: int(re.sub(r"\D", "", s) or 0)
    users = sorted({k[0] for k in idx}, key=gn)
    gestures = sorted({k[1] for k in idx}, key=gn)
    out = {"files": len(idx), "total_files": _NFILES.get(os.path.realpath(clean_root), len(idx)), "participants": users, "gestures": gestures,
           "reps": sorted({k[2] % 100 for k in idx}), "sessions": sorted({k[2] // 100 for k in idx})}
    if idx and with_columns:
        try:
            out["column_variants"] = dataset_columns(clean_root)[1]     # > 1: repetitions recorded with different devices
        except Exception:
            pass
    return out


_COLS_CACHE = {}
_HDR_CACHE = {}          # path -> (mtime_ns, size, header): a CSV is only re-read if it changed


def _header(path):
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f), [])


def _header_cached(path):
    try:
        st = os.stat(path); sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    hit = _HDR_CACHE.get(path)
    if hit and hit[0] == sig:
        return hit[1]
    h = tuple(_header(path))
    _HDR_CACHE[path] = (sig, h)
    return h


def dataset_columns(clean_root):
    """(feature columns, number of different configurations) for the whole cleaned dataset.

    The columns are those of the first CSV (numeric, excluding meta); if the repetitions do not all have the same
    columns (e.g. a participant recorded with one more or one fewer device), only the columns common to
    ALL files are kept, so that every repetition can be loaded. Result cached per dataset
    (several datasets in parallel without evicting each other) as long as its folders do not change; headers are read
    in parallel and kept file by file."""
    idx = index_clean_dir(clean_root)
    if not idx:
        raise FileNotFoundError("Aucun fichier nettoyé")
    rp = os.path.realpath(clean_root)
    key = (len(idx), _dir_signature(clean_root))
    hit = _COLS_CACHE.get(rp)
    if hit and hit[0] == key:
        return hit[1]
    paths = [idx[k] for k in sorted(idx)]
    df = pd.read_csv(paths[0], nrows=3)
    cols = [c for c in df.columns if c not in META_COLS and df[c].dtype.kind in "fi"]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        headers = list(ex.map(_header_cached, paths))
    variants = set()
    common = set(cols)
    for h in headers:
        if h is None:
            continue
        variants.add(h)
        common &= set(h)
    cols = [c for c in cols if c in common]
    res = (cols, len(variants))
    _COLS_CACHE[rp] = (key, res)
    return res


def default_excluded_points(clean_root):
    """Points excluded by default: the IMU accelerometer (`<slot>_Accel`) of each TapStrap2, which duplicates the thumb
    accelerometer (`<slot>_Thumb`). A slot is a TapStrap2 if it has finger columns. Deduced from the dataset columns:
    it does not matter whether the TapStrap2 is TAP1, TAP3… or whether there are several. Empty dataset → no exclusion."""
    try:
        cols, _ = dataset_columns(clean_root)
    except Exception:
        return []
    slots = []
    for c in cols:
        m = re.match(r"^([A-Za-z0-9]+)_(?:Thumb|Index|Middle|Ring|Pinky)_", c)
        if m and m.group(1) not in slots:
            slots.append(m.group(1))
    allpts = {point_name_from_feature(c) for c in cols}
    return [f"{s}_Accel" for s in slots if f"{s}_Accel" in allpts]


def get_feature_cols(clean_root, exclude_points=AUTO, exclude_features=None):
    """Common numeric columns of the dataset, minus the excluded points / features.
    exclude_points="auto" (default): default exclusion (default_excluded_points); None or []: none."""
    if isinstance(exclude_points, str) and exclude_points == AUTO:
        exclude_points = default_excluded_points(clean_root)
    exclude_points, exclude_features = set(exclude_points or []), set(exclude_features or [])
    cols, _ = dataset_columns(clean_root)
    return [c for c in cols if point_name_from_feature(c) not in exclude_points and c not in exclude_features]


def list_points(clean_root):
    """[{point, n_cols}] in column order (for the checkbox list)."""
    cols = get_feature_cols(clean_root, exclude_points=[])
    out = {}
    for c in cols:
        out.setdefault(point_name_from_feature(c), 0)
        out[point_name_from_feature(c)] += 1
    dflt = set(default_excluded_points(clean_root))
    return [{"point": p, "n_cols": n, "default_excluded": p in dflt} for p, n in out.items()]


def resample_series(arr, target_len):
    """Resamples a series (T, F) to (target_len, F) by linear interpolation along the time axis."""
    t_src = np.linspace(0, 1, num=arr.shape[0])
    t_dst = np.linspace(0, 1, num=target_len)
    out = np.empty((target_len, arr.shape[1]), dtype=np.float32)
    for f in range(arr.shape[1]):
        out[:, f] = np.interp(t_dst, t_src, arr[:, f])
    return out


def load_all_samples(clean_root, target_len=100, feature_cols=None, users=None, gestures=None,
                     cancel=None, progress=None):
    """Loads ALL repetitions (like the original load_all_samples).
    Returns X (N,T,F), y (class names), users, meta [{'user','id','infosupp','trueClass'}], feature_cols."""
    index = index_clean_dir(clean_root)
    if feature_cols is None:
        feature_cols = get_feature_cols(clean_root)
    X, y, us, meta = [], [], [], []
    items = [(k, p) for k, p in sorted(index.items())
             if (not users or k[0] in users) and (not gestures or k[1] in gestures)]
    for i, ((user, gesture_name, sample_id), path) in enumerate(items):
        if cancel is not None and cancel():
            raise Cancelled()
        df = pd.read_csv(path)
        arr = df[feature_cols].values.astype(np.float32)
        X.append(resample_series(arr, target_len))
        y.append(gesture_name); us.append(user)
        meta.append({'user': user, 'id': sample_id, 'infosupp': f'S{sample_id // 100:02d}', 'trueClass': gesture_name})
        if progress and i % 25 == 0:
            progress(i, len(items))
    X = np.stack(X) if X else np.empty((0, target_len, len(feature_cols)), dtype=np.float32)
    return X, np.array(y), np.array(us), meta, feature_cols


# ═══════════════════════════════════════════════════════════
#  FOLDS  (copie conforme de load_split_data.py)
# ═══════════════════════════════════════════════════════════
def loso_folds(users, n_val_users=2, seed=42):
    """Leave-One-Subject-Out: 1 participant for test, `n_val_users` others (drawn at random, reproducible)
    for validation, the rest for train. Yields (test_user, train_mask, val_mask, test_mask)."""
    rng = np.random.RandomState(seed)
    unique_users = sorted(set(users))

    if n_val_users >= len(unique_users) - 1:
        raise ValueError(
            f"n_val_users={n_val_users} trop grand pour {len(unique_users)} utilisateurs")

    for test_user in unique_users:
        remaining = [u for u in unique_users if u != test_user]
        rng.shuffle(remaining)

        val_users = set(remaining[:n_val_users])
        train_users = set(remaining[n_val_users:])

        train_mask = np.isin(users, list(train_users))
        val_mask = np.isin(users, list(val_users))
        test_mask = (users == test_user)

        yield test_user, train_mask, val_mask, test_mask


def user_dependent_folds(ids, group_by='rep', n_val_groups=1, seed=42):
    """Leave-One-Repetition-Out (or Session-Out), user-dependent: 1 group removed from ALL users
    for test, `n_val_groups` other(s) for validation, the rest for train.
    Yields (test_group, train_mask, val_mask, test_mask)."""
    ids = np.asarray(ids)
    if group_by == 'rep':
        group_vals = ids % 100
    elif group_by == 'session':
        group_vals = ids // 100
    else:
        raise ValueError(f"group_by doit être 'rep' ou 'session', reçu: {group_by!r}")

    rng = np.random.RandomState(seed)
    unique_groups = sorted(set(group_vals.tolist()))

    if n_val_groups >= len(unique_groups) - 1:
        raise ValueError(
            f"n_val_groups={n_val_groups} trop grand pour {len(unique_groups)} groupes")

    for test_group in unique_groups:
        remaining = [g for g in unique_groups if g != test_group]
        rng.shuffle(remaining)

        val_groups = set(remaining[:n_val_groups])
        train_groups = set(remaining[n_val_groups:])

        train_mask = np.isin(group_vals, list(train_groups))
        val_mask = np.isin(group_vals, list(val_groups))
        test_mask = (group_vals == test_group)

        yield test_group, train_mask, val_mask, test_mask


def make_folds(scenario, users, ids, group_by="rep", n_val=None, seed=42):
    """Unified generator. scenario 'loso' → n_val = validation users (2 by default);
    'ud' → n_val = validation groups (1 by default)."""
    if scenario == "loso":
        return loso_folds(users, n_val_users=2 if n_val is None else n_val, seed=seed)
    if scenario == "ud":
        return user_dependent_folds(ids, group_by=group_by, n_val_groups=1 if n_val is None else n_val, seed=seed)
    raise ValueError(f"scénario inconnu : {scenario!r}")


def plan_folds(scenario, users, ids, group_by="rep", n_val=None, seed=42):
    """Preview of the folds (train / val / test sizes) without loading the series. Raises ValueError if impossible."""
    users, ids = np.asarray(users), np.asarray(ids)
    out = []
    for fold_i, (test_group, tr, va, te) in enumerate(make_folds(scenario, users, ids, group_by, n_val, seed)):
        out.append({"fold": fold_i, "test": str(test_group), "train": int(tr.sum()), "val": int(va.sum()), "test_n": int(te.sum())})
    return out


# ═══════════════════════════════════════════════════════════
#  NORMALIZATION + METRICS  (exact copy of the scripts)
# ═══════════════════════════════════════════════════════════
def scale_data(Xtr, Xval, Xte, n_features):
    """StandardScaler fitted on TRAIN only (per feature, over all time steps)."""
    scaler = StandardScaler()
    scaler.fit(Xtr.reshape(-1, n_features))

    def scale(X):
        shape = X.shape
        return scaler.transform(X.reshape(-1, n_features)).reshape(shape).astype(np.float32)

    return scale(Xtr), scale(Xval), scale(Xte), scaler


def calculate_metrics(y_true, y_pred, y_prob, n_classes, classes):
    """Global (macro) AND per-class metrics. A class absent from the fold's y_true (support=0)
    is excluded from the macro average (it stays visible in per_class with support=0)."""
    accuracy = accuracy_score(y_true, y_pred)

    precision_list, recall_list, f1_list, support_list = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0)

    present = support_list > 0
    if present.any():
        macro_precision = np.mean(precision_list[present])
        macro_recall = np.mean(recall_list[present])
        macro_f1 = np.mean(f1_list[present])
    else:
        macro_precision = macro_recall = macro_f1 = float("nan")

    present_labels = [i for i in range(n_classes) if present[i]]
    try:
        if len(present_labels) < 2:
            auc = float("nan")
        else:
            prob = np.asarray(y_prob)
            if len(present_labels) < n_classes:
                # Only difference from the original scripts: sklearn requires probabilities that sum to 1
                # over the present classes; without renormalization the AUC was always NaN as soon as a class was missing.
                prob = prob[:, present_labels]
                prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-12, None)
            auc = roc_auc_score(y_true, prob, multi_class="ovr", average="macro", labels=present_labels)
    except ValueError:
        auc = float("nan")

    return {
        "global": {"acc": accuracy, "prec": macro_precision, "rec": macro_recall, "f1": macro_f1, "auc": auc},
        "per_class": {
            classes[i]: (
                {"precision": precision_list[i], "recall": recall_list[i], "f1": f1_list[i], "support": int(support_list[i])}
                if present[i] else
                {"precision": None, "recall": None, "f1": None, "support": 0}
            ) for i in range(n_classes)
        }
    }


# ═══════════════════════════════════════════════════════════
#  RESULT FILES  (same names / columns as the scripts)
# ═══════════════════════════════════════════════════════════
def result_files(scenario):
    p = "loso" if scenario == "loso" else "udep"
    return {"per_fold": f"results_{p}_per_fold.csv", "per_class": f"results_{p}_per_class.csv",
            "per_class_summary": f"results_{p}_per_class_summary.csv", "summary": f"results_{p}_summary.csv",
            "predictions": f"results_{p}_predictions.json"}


METRIC_COLS = ["testAccuracy", "testMacroPrecision", "testMacroRecall", "testMacroF1", "testMacroAUC"]


def _nz(v):
    """JSON : NaN / numpy → types natifs."""
    if isinstance(v, (np.floating, float)):
        return None if (v != v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def per_class_summary_df(per_class_df):
    present_df = per_class_df[per_class_df["support"] > 0]
    return (present_df.groupby(["recognizer", "gesture"])
            .agg(n_folds_present=("support", "count"), support_total=("support", "sum"),
                 precision_mean=("precision", "mean"), precision_std=("precision", "std"),
                 recall_mean=("recall", "mean"), recall_std=("recall", "std"),
                 f1_mean=("f1", "mean"), f1_std=("f1", "std"))
            .reset_index())


def write_results(out_dir, scenario, per_fold_rows, per_class_rows, all_predictions):
    """Writes the 5 files. Returns (df_per_fold, summary_df, per_class_summary_df)."""
    names = result_files(scenario)
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(per_fold_rows)
    df.to_csv(os.path.join(out_dir, names["per_fold"]), index=False)
    pcd = pd.DataFrame(per_class_rows)
    pcd.to_csv(os.path.join(out_dir, names["per_class"]), index=False)
    pcs = per_class_summary_df(pcd) if len(pcd) else pd.DataFrame()
    pcs.to_csv(os.path.join(out_dir, names["per_class_summary"]), index=False)
    summary = df.groupby("recognizer")[METRIC_COLS].agg(["mean", "std"]) if len(df) else pd.DataFrame()
    summary.to_csv(os.path.join(out_dir, names["summary"]))
    with open(os.path.join(out_dir, names["predictions"]), "w") as f:
        json.dump(all_predictions, f, indent=2)
    return df, summary, pcs


def confusion_from_predictions(preds, classes):
    ix = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for p in preds:
        t, q = ix.get(p["trueClass"]), ix.get(p["predictedClass"])
        if t is not None and q is not None:
            cm[t, q] += 1
    return cm.tolist()


# ═══════════════════════════════════════════════════════════
#  CROSS-VALIDATION LOOP
# ═══════════════════════════════════════════════════════════
def run_cv(cfg, X, y, users, meta, feature_cols, engine, out_dir, progress=None, cancel=None, save_model=None):
    """Loop of the loso_pipeline.py / udep_pipeline.py scripts.

    cfg: scenario ('loso'|'ud'), group_by ('rep'|'session'), n_val, seed, archs [..],
          tune_mode ('per_fold'|'once'|'manual'), hyperband {max_epochs,factor,iterations,patience},
          manual {epochs,batch,lr,dropout,patience}, epochs, patience, save_folds ('all'|[..]|None)
    progress(dict) / cancel(): optional callbacks. save_model(model, arch, test_group, scaler, fold_spec): saves.
    Returns the report (JSON-serializable dict) and writes the 5 result files in out_dir."""
    progress = progress or (lambda d: None)
    cancel = cancel or (lambda: False)
    scenario = cfg["scenario"]; group_by = cfg.get("group_by", "rep")
    archs = list(cfg["archs"]); tune_mode = cfg.get("tune_mode", "per_fold")
    hb = cfg.get("hyperband", {}); manual = cfg.get("manual", {})
    cp_rows = {}                                     # "copy" mode: {(fold, arch): {fold, arch, hp}} from the source run
    if tune_mode == "copy":
        cp_rows = {(int(r["fold"]), r["arch"]): r for r in cfg["copy_hp"]["rows"]}
    epochs = int(cfg.get("epochs", hb.get("max_epochs", 20))); patience = int(cfg.get("patience", hb.get("patience", 3)))
    ids = np.array([m["id"] for m in meta])
    n_features = X.shape[2]
    input_shape = (X.shape[1], n_features)

    classes = sorted(set(y))
    le = LabelEncoder(); le.fit(classes)
    y_enc = le.transform(y)
    n_classes = len(classes)

    group_vals = (ids % 100 if group_by == "rep" else ids // 100) if scenario == "ud" else None
    folds = list(make_folds(scenario, users, ids, group_by, cfg.get("n_val"), cfg.get("seed", 42)))
    n_folds = len(folds)
    names = result_files(scenario)

    per_fold_rows, per_class_rows, all_predictions, hp_rows = [], [], {}, []
    best_hp_cache = {}; first_hp = {}; save_errors = []
    t0 = time.time(); units_done = 0; units_total = n_folds * len(archs)
    save_all = cfg.get("save_folds") == "all"
    save_set = set(str(s) for s in (cfg.get("save_folds") or [])) if not save_all else set()

    def emit(stage, msg, fold=None, arch=None, **kw):
        el = time.time() - t0
        eta = el / units_done * (units_total - units_done) if units_done else None
        progress({"stage": stage, "message": msg, "fold": fold, "n_folds": n_folds, "arch": arch,
                  "done": units_done, "total": units_total, "elapsed": round(el, 1),
                  "eta": None if eta is None else round(eta, 1), **kw})

    for fold_i, (test_group, train_mask, val_mask, test_mask) in enumerate(folds):
        if cancel(): raise Cancelled()
        Xtr, ytr = X[train_mask], y_enc[train_mask]
        Xval, yval = X[val_mask], y_enc[val_mask]
        Xte, yte = X[test_mask], y_enc[test_mask]
        meta_te = [meta[i] for i in np.where(test_mask)[0]]

        if scenario == "loso":
            val_key = ",".join(str(u) for u in sorted(set(np.array(users)[val_mask])))
            train_list = sorted(set(np.array(users)[train_mask]))
            fold_spec = {"scenario": "loso", "train": [str(u) for u in train_list],
                         "val": [str(u) for u in sorted(set(np.array(users)[val_mask]))], "test": str(test_group)}
        else:
            val_key = ",".join(str(g) for g in sorted(set(group_vals[val_mask].tolist())))
            fold_spec = {"scenario": "ud", "group_by": group_by,
                         "train": sorted(set(int(g) for g in group_vals[train_mask])),
                         "val": sorted(set(int(g) for g in group_vals[val_mask])), "test": int(test_group)}

        Xtr_s, Xval_s, Xte_s, scaler = scale_data(Xtr, Xval, Xte, n_features)

        for arch in archs:
            if cancel(): raise Cancelled()
            tag = f"fold{fold_i}"
            if tune_mode == "manual":
                hp, mode = engine.manual_hp(arch, manual), "manuel"
            elif tune_mode == "copy":
                cp = cfg["copy_hp"]
                src = (cp_rows.get((fold_i, arch)) if cp.get("scope") == "per_fold" else None) or cp_rows[(0, arch)]
                hp = engine.fixed_hp_for(arch, src["hp"]) if hasattr(engine, "fixed_hp_for") else engine.fixed_hp(src["hp"])
                mode = f"copié de {cp['run']} (pli {src['fold'] + 1})"
            elif tune_mode == "per_fold" or arch not in best_hp_cache:
                srch = "Grid search" if arch in ML_ARCHS else "Hyperband"
                emit("tuning", f"{srch} {arch.upper()} · fold {fold_i + 1}/{n_folds}", fold_i, arch, trial=0)
                hp = engine.tune(arch, input_shape, n_classes, Xtr_s, ytr, Xval_s, yval,
                                 (cfg.get("grid") or {}).get(arch) if arch in ML_ARCHS else hb, f"{scenario}_{tag}",
                                 lambda n, _a=arch, _f=fold_i, _s=srch: emit("tuning", f"{_s} {_a.upper()} · fold {_f + 1}/{n_folds} · trial {n}", _f, _a, trial=n))
                mode = srch.lower() + (" (ce pli)" if tune_mode == "per_fold" else " (pli 1, réutilisé)")
                if tune_mode == "once":
                    best_hp_cache[arch] = hp
            else:
                hp, mode = best_hp_cache[arch], ("grid search" if arch in ML_ARCHS else "hyperband") + " (pli 1, réutilisé)"

            first_hp.setdefault(arch, hp)                   # hyperparameters of fold 1, reused for the final model
            emit("training", f"Train {arch.upper()} · fold {fold_i + 1}/{n_folds}", fold_i, arch)
            start = time.time()
            model, test_prob = engine.train_eval(arch, hp, input_shape, n_classes, Xtr_s, ytr, Xval_s, yval, Xte_s,
                                                 epochs, patience,
                                                 lambda ep, logs, _a=arch, _f=fold_i: emit("training", f"Train {_a.upper()} · fold {_f + 1}/{n_folds} · epoch {ep}", _f, _a, epoch=ep))
            test_pred = np.argmax(test_prob, axis=1)
            res = calculate_metrics(yte, test_pred, test_prob, n_classes, classes)
            gm, pcm = res["global"], res["per_class"]
            elapsed = time.time() - start

            rec_name = arch.upper() + ("_ML" if arch in ML_ARCHS else "_DL")
            fold_keys = ({"test_user": test_group, "val_users": val_key} if scenario == "loso"
                         else {"test_group": test_group, "val_groups": val_key, "group_by": group_by})
            for gesture, m in pcm.items():
                per_class_rows.append({"fold": fold_i, **fold_keys, "recognizer": rec_name, "gesture": gesture,
                                       "support": m["support"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
            per_fold_rows.append({"fold": fold_i, **fold_keys, "recognizer": rec_name,
                                  "trainSize": len(Xtr), "valSize": len(Xval), "testSize": len(Xte),
                                  "testAccuracy": gm["acc"], "testMacroPrecision": gm["prec"], "testMacroRecall": gm["rec"],
                                  "testMacroF1": gm["f1"], "testMacroAUC": gm["auc"]})
            hp_rows.append({"fold": fold_i, "test": str(test_group), "arch": arch, "mode": mode, "hp": engine.hp_values(hp),
                            "seconds": round(elapsed, 1)})

            if save_model and (save_all or str(test_group) in save_set):
                try:
                    save_model(model, arch, test_group, scaler, fold_spec, hp_rows[-1]["hp"])
                except Exception as e:                      # a failed save must not lose the computation
                    save_errors.append(f"{arch} · fold {test_group} : {type(e).__name__}: {e}")
                    emit("training", f"⚠ sauvegarde du modèle impossible : {e}", fold_i, arch)
            engine.free(model)

            fk = "foldTestUser" if scenario == "loso" else "foldTestGroup"
            ap = all_predictions.setdefault(arch, [])
            for m, true_id, pred_id, prob in zip(meta_te, yte, test_pred, test_prob):
                ap.append({"user": m["user"], "id": m["id"], "infosupp": m.get("infosupp"),
                           "trueClass": classes[true_id], "predictedClass": classes[pred_id],
                           "score": float(prob[pred_id]), fk: test_group})
            units_done += 1
            emit("fold_done", f"{arch.upper()} · fold {fold_i + 1}/{n_folds} : acc={gm['acc']:.3f} f1={gm['f1']:.3f} {elapsed:.1f}s", fold_i, arch)
        write_results(out_dir, scenario, per_fold_rows, per_class_rows, all_predictions)     # partial results at each fold

    final_models = []
    if cfg.get("final_model") and save_model:
        # FINAL model: all participants (or groups) are used for training, except a small randomly drawn validation set
        # (early stopping + hyperparameter choice). No test: its score cannot be measured, the score to cite is that of the folds.
        rs = np.random.RandomState(cfg.get("seed", 42))
        if scenario == "loso":
            all_g = sorted(str(u) for u in set(users)); rs.shuffle(all_g)
            n_v = max(1, min(int(cfg.get("n_val") or 2), len(all_g) - 1)); val_g = set(all_g[:n_v])
            vmask = np.array([str(u) in val_g for u in users])
            f_spec = {"scenario": "loso", "train": sorted(set(all_g) - val_g), "val": sorted(val_g), "test": None}
        else:
            all_g = sorted(set(int(g) for g in group_vals)); rs.shuffle(all_g)
            n_v = max(1, min(int(cfg.get("n_val") or 1), len(all_g) - 1)); val_g = set(all_g[:n_v])
            vmask = np.isin(group_vals, list(val_g))
            f_spec = {"scenario": "ud", "group_by": group_by, "train": sorted(set(all_g) - val_g), "val": sorted(val_g), "test": None}
        tmask = ~vmask
        Xtr, ytr, Xval, yval = X[tmask], y_enc[tmask], X[vmask], y_enc[vmask]
        Xtr_s, Xval_s, _unused, scaler = scale_data(Xtr, Xval, Xval, n_features)
        for arch in archs:
            if cancel(): raise Cancelled()
            hp = first_hp[arch]                             # no new search: the hyperparameters of fold 1
            emit("training", f"Train {arch.upper()} · final", None, arch)
            model, _p = engine.train_eval(arch, hp, input_shape, n_classes, Xtr_s, ytr, Xval_s, yval, Xval_s, epochs, patience,
                                          lambda ep, logs, _a=arch: emit("training", f"Train {_a.upper()} · final · epoch {ep}", None, _a, epoch=ep))
            try:
                save_model(model, arch, "final", scaler, f_spec, engine.hp_values(hp))
                final_models.append(arch)
            except Exception as e:
                save_errors.append(f"{arch} · final : {type(e).__name__}: {e}")
                emit("training", f"⚠ sauvegarde du modèle final impossible : {e}", None, arch)
            engine.free(model)

    df, summary, pcs = write_results(out_dir, scenario, per_fold_rows, per_class_rows, all_predictions)
    report = build_report(cfg, scenario, classes, feature_cols, len(X), n_folds, df, pcs, all_predictions, hp_rows,
                          time.time() - t0, list(names.values()))
    report["final_models"] = final_models; report["save_errors"] = save_errors
    return report


def build_report(cfg, scenario, classes, feature_cols, n_samples, n_folds, df, pcs, all_predictions, hp_rows, seconds, files):
    summary = {}
    if len(df):
        for rec, g in df.groupby("recognizer"):
            summary[rec] = {m: {"mean": _nz(g[m].mean()), "std": _nz(g[m].std())} for m in METRIC_COLS}
    return {
        "scenario": scenario, "classes": classes, "n_features": len(feature_cols), "features": feature_cols,
        "n_samples": int(n_samples), "n_folds": int(n_folds), "seconds": round(seconds, 1), "files": files,
        "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "summary": summary,
        "per_fold": [{k: _nz(v) for k, v in r.items()} for r in df.to_dict("records")],
        "per_class_summary": [{k: _nz(v) for k, v in r.items()} for r in pcs.to_dict("records")] if len(pcs) else [],
        "confusion": {a: confusion_from_predictions(p, classes) for a, p in all_predictions.items()},
        "hyperparameters": hp_rows,
        "leak_note": ("Hyperband par fold."
                      if cfg.get("tune_mode") == "per_fold" else
                      "Hyperparamètres choisis sur le fold 1 puis réutilisés."
                      if cfg.get("tune_mode") == "once" else
                      f"Hyperparamètres copiés du run {cfg.get('copy_hp', {}).get('run')} (choisis sur ses propres plis)."
                      if cfg.get("tune_mode") == "copy" else
                      "Hyperparamètres manuels."),
    }


# ████████████████████████████████████████████████████████████
#  3. IMPORTED MODELS
# ████████████████████████████████████████████████████████████


FORMATS = {".keras": "keras", ".h5": "keras", ".hdf5": "keras", ".joblib": "sklearn"}   # Keras, or scikit-learn exported by model_template_ml.py
FORMAT_LABEL = {"keras": "Keras", "sklearn": "scikit-learn"}
ML_ARCHS = ("logreg", "svm", "rf")                # classic machine learning (clf_sklearn.py), no TensorFlow needed
MAX_UPLOAD = 1024 * 1024 * 1024
RE_FINGER = re.compile(r"Thumb|Index|Middle|Ring|Pinky", re.I)
RE_IMU = re.compile(r"Accel|Gyro", re.I)


def available_libs():
    import importlib.util as u
    return {k: u.find_spec(k) is not None for k in ("tensorflow", "keras_tuner", "tensorboard", "sklearn")}


# ═══════════════════════════════════════════════════════════
#  INVENTORY  (data_clean/models/<name>.<ext> + <name>.json)
# ═══════════════════════════════════════════════════════════
def models_dir(clean_root=None):
    """GLOBAL models folder (clean_root is kept for compatibility, it is no longer used)."""
    d = os.path.join(STORE, "models"); os.makedirs(d, exist_ok=True); return d


def safe_name(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")      # "modèle" → "modele"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", (lambda b: os.path.splitext(b)[0] if os.path.splitext(b)[1].lower() in FORMATS else b)(os.path.basename(s))).strip("._")[:60]


def list_models(clean_root):
    out = []
    for f in sorted(glob.glob(os.path.join(models_dir(clean_root), "*.json")), key=os.path.getmtime, reverse=True):
        try:
            with open(f, encoding="utf-8") as fh: m = json.load(fh)
        except Exception:
            continue
        if not os.path.exists(os.path.join(os.path.dirname(f), m.get("file", "?"))): continue
        out.append({k: m.get(k) for k in ("name", "format", "source", "created", "arch", "T", "n_features", "classes",
                                          "norm", "fold_spec", "note", "file", "dataset", "dataset_path")})
        out[-1]["dataset"] = _short(out[-1]["dataset"])
    return out


def get_model(clean_root, name):
    p = os.path.join(models_dir(clean_root), safe_name(name) + ".json")
    if not os.path.isfile(p): raise ValueError("Modèle introuvable")
    with open(p, encoding="utf-8") as fh: m = json.load(fh)
    m["path"] = os.path.join(models_dir(clean_root), m["file"])
    m["meta_path"] = p
    return m


def delete_model(clean_root, name):
    m = get_model(clean_root, name)
    sc = os.path.join(models_dir(clean_root), m["scaler_file"]) if m.get("scaler_file") else None
    for p in (m["path"], m["meta_path"], sc):
        if not p: continue
        if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
        elif os.path.exists(p): os.remove(p)


def _unique(d, base, ext):
    name, k = base, 2
    while os.path.exists(os.path.join(d, name + ext)) or os.path.exists(os.path.join(d, name + ".json")):
        name = f"{base}_{k}"; k += 1
    return name


def register_trained(clean_root, run_id, arch, test_group, model_ext, meta):
    """Reserves the name of a model coming out of a training run. Returns (name, model file path)."""
    d = models_dir(clean_root)
    name = _unique(d, safe_name(f"{run_id}_{arch}_final" if test_group == "final" else f"{run_id}_{arch}_fold_{test_group}"), model_ext)
    return name, os.path.join(d, name + model_ext), os.path.join(d, name + ".json")


def stage_import(clean_root, filename, data, name, norm="none", exclude_points="", classes="", T="", scaler_file=None,
                 scaler_name="", seen=""):
    """Stores the file (without loading it) + a provisional record. Inspection (input shape…) is then done
    in the worker process (inspect_model). Returns the name."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in FORMATS:
        raise ValueError("Formats acceptés (" + ", ".join(sorted(FORMATS)) + "), reçu : " + (ext or "sans extension"))
    if len(data) > MAX_UPLOAD: raise ValueError("Fichier trop gros, max 1 Go")
    if norm not in ("none", "scaler"):
        raise ValueError("Normalisation non autorisée")
    if norm == "scaler" and not scaler_file:
        raise ValueError("Fichier de moyenne et écart-type manquant")
    d = models_dir(clean_root)
    name = _unique(d, safe_name(name or filename) or "modele", ext)
    path = os.path.join(d, name + ext)
    with open(path, "wb") as fh: fh.write(data)
    meta = {"name": name, "file": name + ext, "format": FORMATS[ext], "source": "imported", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "norm": norm,
            "exclude_points": [p.strip() for p in (exclude_points or "").split(",") if p.strip()],
            "classes": [c.strip() for c in (classes or "").split(",") if c.strip()] or None,
            "T": int(T) if str(T).strip().isdigit() else None}
    seen_users = [u.strip() for u in (seen or "").split(",") if u.strip()]
    if seen_users:                                    # participants seen during training → same overlap check as the models trained here
        meta["fold_spec"] = {"scenario": "loso", "train": seen_users, "val": [], "test": None}
    if scaler_file:
        try:                                              # the model script also writes the classes, T and the columns there
            sj = json.loads(scaler_file.decode("utf-8"))
        except Exception:
            sj = None
        if isinstance(sj, dict):
            if not meta["classes"] and isinstance(sj.get("classes"), list): meta["classes"] = [str(c) for c in sj["classes"]]
            if meta["T"] is None and isinstance(sj.get("T"), int): meta["T"] = sj["T"]
            if isinstance(sj.get("columns"), list): meta["columns"] = [str(c) for c in sj["columns"]]
        meta["scaler_file"] = name + "_scaler" + os.path.splitext(scaler_name)[1].lower()
        with open(os.path.join(d, meta["scaler_file"]), "wb") as fh: fh.write(scaler_file)
        meta["norm"] = "scaler"
    with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as fh: json.dump(meta, fh, ensure_ascii=False, indent=1)
    return name


# ═══════════════════════════════════════════════════════════
#  CHARGEMENT  (processus de travail)
# ═══════════════════════════════════════════════════════════
def _softmax_if_needed(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1: p = p[None, :]
    if p.size and (p.min() < -1e-6 or np.abs(p.sum(1) - 1).max() > 1e-3):
        e = np.exp(p - p.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
    return p


class Predictor:
    """proba(X): X (N, T, F) float32 → (N, C). Attributes: fmt, T, F (None if unknown), n_classes, class_names."""
    fmt = ""; T = None; F = None; n_classes = None; class_names = None

    def proba(self, X): raise NotImplementedError


class KerasPredictor(Predictor):
    fmt = "keras"
    def __init__(self, path):
        from tensorflow import keras
        self.m = keras.models.load_model(path, compile=False)
        shp = self.m.input_shape
        if isinstance(shp, list): shp = shp[0]
        if len(shp) == 3: self.T, self.F = shp[1], shp[2]
        elif len(shp) == 2: self.T, self.F = None, shp[1]
        self.n_classes = int(self.m.output_shape[-1])
    def proba(self, X):
        return _softmax_if_needed(self.m.predict(X, verbose=0))


class SklearnPredictor(Predictor):
    """scikit-learn model saved by the page or by model_template_ml.py (see clf_sklearn.py)."""
    fmt = "sklearn"
    def __init__(self, path):
        import clf_sklearn
        self.p = clf_sklearn.BundlePredictor(path)
        self.T, self.F, self.n_classes, self.class_names = self.p.T, self.p.F, self.p.n_classes, self.p.class_names
        self.columns = self.p.columns
    def proba(self, X):
        return self.p.proba(X)


LOADERS = {"keras": KerasPredictor, "sklearn": SklearnPredictor}
NEEDS = {"keras": "tensorflow", "sklearn": "sklearn"}
INSTALL = {"tensorflow": "pip install tensorflow", "sklearn": "pip install scikit-learn"}


def load_predictor(meta_or_path, fmt=None):
    path = meta_or_path["path"] if isinstance(meta_or_path, dict) else meta_or_path
    fmt = fmt or (meta_or_path["format"] if isinstance(meta_or_path, dict) else FORMATS.get(os.path.splitext(path)[1].lower()))
    if fmt not in LOADERS:
        raise ValueError(f"Format « {fmt} » non pris en charge : modèles Keras (.keras, .h5) ou scikit-learn exportés par model_template_ml.py (.joblib)")
    lib = NEEDS[fmt]
    if not available_libs().get(lib, False):
        raise ValueError(f"{FORMAT_LABEL[fmt]} : {lib} manquant, {INSTALL[lib]}")
    try:
        return LOADERS[fmt](path)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Impossible de lire ce fichier comme {FORMAT_LABEL[fmt]} : {type(e).__name__}: {str(e)[:300]}")


# ═══════════════════════════════════════════════════════════
#  FEATURES / NORMALISATION
# ═══════════════════════════════════════════════════════════
def feature_candidates(clean_root, exclude_points=None):
    """Plausible column selections, in trial order: [(label, columns)]."""
    allc = get_feature_cols(clean_root, exclude_points=[])
    pn = point_name_from_feature
    dflt = set(default_excluded_points(clean_root))
    lbl = ", ".join(sorted(dflt)) or "rien"
    cands = []
    if exclude_points is not None: cands.append(("exclusion demandée", [c for c in allc if pn(c) not in set(exclude_points)]))
    cands += [(f"défaut (sans {lbl})", [c for c in allc if pn(c) not in dflt]),
              ("toutes les colonnes", allc),
              ("doigts seulement", [c for c in allc if RE_FINGER.search(c)]),
              ("IMU seulement", [c for c in allc if RE_IMU.search(c)]),
              (f"IMU sans {lbl}", [c for c in allc if RE_IMU.search(c) and pn(c) not in dflt])]
    return cands


def resolve_features(clean_root, F, exclude_points=None, explicit=None):
    """Columns to feed the model. `explicit` (model trained here) takes precedence; otherwise look for the column combination
    that gives exactly F features (by default: without the Accel of each TapStrap2, like the scripts)."""
    if explicit: return list(explicit)
    cands = feature_candidates(clean_root, exclude_points)
    if F is None: return cands[0][1]
    for _lab, cols in cands:
        if len(cols) == F: return cols
    sizes = ", ".join(f"{lab} = {len(c)}" for lab, c in cands)
    raise ValueError(f"Le modèle attend {F} colonnes, sélections possibles : {sizes}")


def load_scaler_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as f: d = json.load(f)
        return np.array(d["mean"], dtype=np.float32), np.array(d.get("scale", d.get("std")), dtype=np.float32)
    if ext == ".npz":
        d = np.load(path); return d["mean"].astype(np.float32), d["scale" if "scale" in d else "std"].astype(np.float32)
    import joblib
    s = joblib.load(path)
    return np.asarray(s.mean_, dtype=np.float32), np.asarray(s.scale_, dtype=np.float32)


def apply_norm(meta, X, models_path):
    """Model-specific normalization. 'scaler': recorded mean/std; 'zscore_data': computed on X."""
    norm = meta.get("norm", "none")
    if norm == "none": return X
    if norm == "scaler":
        if meta.get("mean") is not None: mean, scale = np.array(meta["mean"], np.float32), np.array(meta["scale"], np.float32)
        else: mean, scale = load_scaler_file(os.path.join(models_path, meta["scaler_file"]))
        if len(mean) != X.shape[2]:
            raise ValueError(f"Le scaler a {len(mean)} features, les données en ont {X.shape[2]}")
    else:
        flat = X.reshape(-1, X.shape[2]); mean, scale = flat.mean(0), flat.std(0)
    scale = np.where(np.asarray(scale) == 0, 1, scale)
    return ((X - mean) / scale).astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  INSPECTION / EVALUATION / PREDICTION  (worker process)
# ═══════════════════════════════════════════════════════════
def inspect_model(clean_root, name):
    """Loads the imported model, reads its input / output shape, completes and saves the record. Raises ValueError if unusable."""
    m = get_model(clean_root, name)
    pred = load_predictor(m)
    if not m.get("columns") and getattr(pred, "columns", None):   # scikit-learn file of model_template_ml.py: columns inside
        m["columns"] = [str(c) for c in pred.columns]
    T = pred.T or m.get("T")
    F = pred.F
    ambiguous = []
    if not T:                                           # variable-length model (or format without input shape): take the convention, T=100
        T = 100
        ambiguous = ["ce modèle n'indique pas de nombre de pas de temps : 100 utilisé (à corriger dans « Réglages avancés » si besoin)"]
    if m.get("columns"):
        allc = set(get_feature_cols(clean_root, exclude_points=[]))
        missing = [c for c in m["columns"] if c not in allc]
        if missing:
            raise ValueError(f"Colonnes absentes de ce dataset : {missing[:4]}")
        if F is not None and len(m["columns"]) != F:
            raise ValueError(f"Le modèle attend {F} colonnes, le fichier de normalisation en a {len(m['columns'])}")
    cols = resolve_features(clean_root, F, exclude_points=m.get("exclude_points") or None, explicit=m.get("columns"))
    if m.get("norm") == "scaler":
        try:
            mean, scale = load_scaler_file(os.path.join(models_dir(clean_root), m["scaler_file"]))
        except Exception as e:
            raise ValueError("Fichier de normalisation illisible : " + type(e).__name__)
        if len(mean) != len(cols) or len(scale) != len(cols):
            raise ValueError(f"Normalisation : {len(mean)} valeurs, le modèle utilise {len(cols)} colonnes")
        if np.any(np.asarray(scale) == 0): raise ValueError("Le fichier de normalisation contient un écart-type nul")
    n_out = pred.n_classes
    classes = m.get("classes") or pred.class_names
    if not classes:
        scan = scan_clean_dir(clean_root)
        classes = scan["gestures"] if (not n_out or n_out == len(scan["gestures"])) else [f"Gesture_{i + 1:02d}" for i in range(n_out)]
    if n_out and len(classes) != n_out:
        raise ValueError(f"Le modèle a {n_out} sorties, le dataset {len(classes)} gestes")
    m.update({"T": int(T), "n_features": len(cols), "features": cols, "classes": classes, "n_outputs": n_out})
    for k in ("path", "meta_path"): m.pop(k, None)
    with open(os.path.join(models_dir(clean_root), m["name"] + ".json"), "w", encoding="utf-8") as fh: json.dump(m, fh, ensure_ascii=False, indent=1)
    return {**{k: m.get(k) for k in ("name", "format", "T", "n_features", "classes")}, "ambiguous": ambiguous}


def _load_X(clean_root, meta, users=None, gestures=None, cancel=None):
    cols = meta.get("features") or resolve_features(clean_root, meta.get("n_features"), meta.get("exclude_points") or None)
    return load_all_samples(clean_root, int(meta["T"]), feature_cols=cols, users=users, gestures=gestures, cancel=cancel)


def evaluate_model(clean_root, name, users=None, gestures=None, cancel=None, progress=None, data_root=None):
    """Applies the model to all chosen repetitions. Returns a report (dict).
    clean_root = model folder; data_root = dataset being evaluated (by default the same)."""
    meta = get_model(clean_root, name)
    X, y, us, smeta, cols = _load_X(data_root or clean_root, meta, users, gestures, cancel)
    if not len(X): raise ValueError("Aucune répétition dans la sélection")
    pred = load_predictor(meta)
    Xn = apply_norm(meta, X, models_dir(clean_root))
    prob = pred.proba(Xn)
    classes = list(meta["classes"])
    if prob.shape[1] != len(classes): raise ValueError(f"Le modèle renvoie {prob.shape[1]} sorties pour {len(classes)} classes")
    known = np.array([c in classes for c in y])
    if not known.any(): raise ValueError("Aucun geste évalué dans les classes du modèle")
    ix = {c: i for i, c in enumerate(classes)}
    yt = np.array([ix[c] for c in y[known]]); P = prob[known]; pr = P.argmax(1); uk = us[known]; ids = np.array([m["id"] for m in smeta])[known]
    res = calculate_metrics(yt, pr, P, len(classes), classes)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, q in zip(yt, pr): cm[t, q] += 1
    per_user = []
    for u in sorted(set(uk.tolist())):
        mk = uk == u
        g = calculate_metrics(yt[mk], pr[mk], P[mk], len(classes), classes)["global"]
        nn = lambda v: None if v != v else float(v)
        per_user.append({"participant": u, "n": int(mk.sum()), "accuracy": float((pr[mk] == yt[mk]).mean()),
                         "precision": nn(g["prec"]), "recall": nn(g["rec"]), "f1": nn(g["f1"])})
    # overlap with the training (models trained here)
    overlap = None; spec = meta.get("fold_spec")
    if spec:
        seen = (np.isin(uk, spec["train"] + spec["val"]) if spec["scenario"] == "loso"
                else np.isin((ids % 100 if spec["group_by"] == "rep" else ids // 100), spec["train"] + spec["val"]))
        overlap = {"seen_fraction": float(seen.mean()), "n_seen": int(seen.sum()), "n": int(len(seen)),
                   "test": spec["test"], "scenario": spec["scenario"]}
    return {"kind": "evaluate", "model": meta["name"], "format": meta["format"], "n": int(known.sum()), "n_ignored": int((~known).sum()),
            "classes": classes, "global": {k: (None if v != v else float(v)) for k, v in res["global"].items()},
            "per_class": {c: {k: (None if v is None or v != v else v) for k, v in m.items()} for c, m in res["per_class"].items()},
            "confusion": cm.tolist(), "per_participant": per_user, "overlap": overlap, "norm": meta.get("norm"), "model_dataset": _short(meta.get("dataset")),
            "features": len(cols), "T": int(meta["T"])}


def predict_one(clean_root, name, rel):
    """Recognizes the gesture of ONE cleaned repetition (relative path Gesture_XX/PXX/….csv)."""
    meta = get_model(clean_root, name)
    if not re.fullmatch(r"Gesture_\d+/P\d+/[^/\\]+\.csv", rel or ""): raise ValueError("Chemin de fichier invalide")
    path = os.path.realpath(os.path.join(clean_root, rel))
    if not path.startswith(os.path.realpath(clean_root) + os.sep) or not os.path.isfile(path): raise ValueError("Fichier introuvable")
    import pandas as pd
    cols = meta.get("features") or resolve_features(clean_root, meta.get("n_features"), meta.get("exclude_points") or None)
    df = pd.read_csv(path)
    X = resample_series(df[cols].values.astype(np.float32), int(meta["T"]))[None]
    Xn = apply_norm(meta, X, models_dir(clean_root))
    p = load_predictor(meta).proba(Xn)[0]
    classes = meta["classes"]
    order = np.argsort(-p)[:5]
    m = re.search(r"(Gesture_\d+)", rel)
    return {"predicted": classes[int(order[0])], "true": m.group(1) if m else None,
            "top": [{"class": classes[int(i)], "p": float(p[i])} for i in order]}


def predict_many(clean_root, name, rels):
    """Recognizes several repetitions at once (the model is loaded only once)."""
    meta = get_model(clean_root, name)
    root = os.path.realpath(clean_root)
    rels = list(dict.fromkeys(rels))[:2000]
    if not rels: raise ValueError("Aucune répétition choisie")
    cols = meta.get("features") or resolve_features(clean_root, meta.get("n_features"), meta.get("exclude_points") or None)
    X, info = [], []
    for rel in rels:
        if not re.fullmatch(r"Gesture_\d+/P\d+/[^/\\]+\.csv", rel or ""): raise ValueError("Chemin de fichier invalide")
        path = os.path.realpath(os.path.join(clean_root, rel))
        if not path.startswith(root + os.sep) or not os.path.isfile(path): raise ValueError("Fichier introuvable : " + rel)
        X.append(resample_series(pd.read_csv(path)[cols].values.astype(np.float32), int(meta["T"])))
        m = FILENAME_REGEX.match(os.path.basename(rel))
        g = re.search(r"(Gesture_\d+)", rel)
        info.append({"rel": rel, "user": f"P{int(m.group(1)):02d}" if m else None, "id": int(m.group(2)) * 100 + int(m.group(4)) if m else None,
                     "true": g.group(1) if g else None})
    Xn = apply_norm(meta, np.stack(X), models_dir(clean_root))
    P = load_predictor(meta).proba(Xn)
    classes = meta["classes"]
    items = []
    for it, p in zip(info, P):
        order = np.argsort(-p)[:5]
        pred = classes[int(order[0])]
        items.append({**it, "predicted": pred, "score": float(p[order[0]]), "ok": pred == it["true"],
                      "top": [{"class": classes[int(i)], "p": float(p[i])} for i in order]})
    return {"n": len(items), "n_ok": int(sum(i["ok"] for i in items)), "items": items}


# ████████████████████████████████████████████████████████████
#  4. PROCESSUS DE CALCUL
# ████████████████████████████████████████████████████████████




def _atomic(path, obj):
    """Atomic JSON write. On Windows, os.replace fails (WinError 5/32) if Flask reads the file at the same instant:
    we retry, then write directly as a last resort. Never raises for a simple progress file."""
    tmp = path + ".tmp"
    data = json.dumps(obj, ensure_ascii=False)
    for _ in range(40):
        try:
            with open(tmp, "w", encoding="utf-8") as f: f.write(data)
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(0.05)
    try:
        with open(path, "w", encoding="utf-8") as f: f.write(data)
    except OSError:
        if not path.endswith("progress.json"): raise


def make_dl_engine(tuner_dir, cancel):
    mod = os.environ.get("CLF_ENGINE_MODULE")
    if mod:
        if os.environ.get("CLF_ENGINE_PATH"): sys.path.insert(0, os.environ["CLF_ENGINE_PATH"])
        return importlib.import_module(mod).make_engine(tuner_dir, cancel)
    import clf_keras_v2
    return clf_keras_v2.KerasEngine(tuner_dir, cancel)


class MixedEngine:
    """Sends each architecture to its engine: logreg / svm / rf → scikit-learn, dense / cnn / lstm → Keras.
    The Keras engine (and TensorFlow) is only loaded if a network is actually trained."""
    def __init__(self, tuner_dir, cancel, seed=0):
        import clf_sklearn
        self.ml = clf_sklearn.SklearnEngine(cancel, seed=seed, cancelled_exc=Cancelled)
        self.tuner_dir, self.cancel, self._dl = tuner_dir, cancel, None
        self.owner = {}                               # id(hp or model) → engine, for the calls without arch

    def _dl_engine(self):
        if self._dl is None: self._dl = make_dl_engine(self.tuner_dir, self.cancel)
        return self._dl

    def _for(self, arch):
        return self.ml if arch in ML_ARCHS else self._dl_engine()

    def _mark(self, obj, eng):
        self.owner[id(obj)] = eng; return obj

    def tune(self, arch, *a, **k): return self._mark(self._for(arch).tune(arch, *a, **k), self._for(arch))
    def manual_hp(self, arch, manual): return self._mark(self._for(arch).manual_hp(arch, manual), self._for(arch))
    def hp_values(self, hp): return self.owner.get(id(hp), self.ml).hp_values(hp)
    def train_eval(self, arch, hp, *a, **k):
        model, prob = self._for(arch).train_eval(arch, hp, *a, **k)
        return self._mark(model, self._for(arch)), prob
    def save(self, model, path, scaler=None): return self.owner[id(model)].save(model, path, scaler=scaler)
    def free(self, model):
        eng = self.owner.pop(id(model), None)
        if eng is not None: eng.free(model)

    def fixed_hp_for(self, arch, values):
        return self._mark(self._for(arch).fixed_hp(values), self._for(arch))


def make_engine(tuner_dir, cancel, seed=0):
    return MixedEngine(tuner_dir, cancel, seed)


def seed_everything(seed, archs=None):
    np.random.seed(seed)
    if archs is not None and all(a in ML_ARCHS for a in archs):
        return                                         # scikit-learn only: no TensorFlow to seed
    if os.environ.get("CLF_ENGINE_MODULE"):
        return
    import clf_keras_v2
    clf_keras_v2.seed_everything(seed)


def run_train(job, jd, progress, cancel):
    cfg = job["cfg"]; root = job["clean_root"]
    exclude = cfg.get("exclude_points")
    cols = get_feature_cols(root, exclude_points=AUTO if exclude is None else exclude)
    if not cols: raise ValueError("Aucune feature sélectionnée")
    T = int(cfg.get("T", 100))
    progress({"stage": "loading", "message": "Chargement des répétitions…", "done": 0, "total": 1})
    X, y, users, meta, cols = load_all_samples(
        root, T, feature_cols=cols, users=cfg.get("users") or None, gestures=cfg.get("gestures") or None, cancel=cancel,
        progress=lambda i, n: progress({"stage": "loading", "message": f"Chargement {i}/{n}", "done": 0, "total": 1}))
    if not len(X): raise ValueError("Aucune répétition trouvée")
    seed_everything(int(cfg.get("run_seed", 0)), cfg.get("archs"))
    out_dir = job["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    engine = make_engine(os.path.join(jd, "tuner"), cancel, int(cfg.get("run_seed", 0)))
    classes = sorted(set(y.tolist()))
    run_id = os.path.basename(out_dir)
    ext = job.get("model_ext", ".keras")
    saved = []

    def save_model(model, arch, test_group, scaler, fold_spec, hp):
        is_ml = arch in ML_ARCHS
        name, path, meta_path = register_trained(root, run_id, arch, test_group, ".joblib" if is_ml else ext, {})
        baked = engine.save(model, path, scaler=scaler)           # True: the normalization is INSIDE the file
        m = {"name": name, "file": os.path.basename(path), "format": "sklearn" if is_ml else "keras", "source": "trained",
             "dataset": job.get("dataset_label") or os.path.basename(root), "dataset_path": root,
             "created": time.strftime("%Y-%m-%d %H:%M:%S"), "arch": arch, "T": T, "n_features": len(cols), "features": cols,
             "classes": classes, "norm": "none" if baked else "scaler", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
             "fold_spec": fold_spec, "hp": hp, "note": f"{run_id} · modèle final (tous les participants)" if test_group == "final" else f"{run_id} · test {test_group}"}
        _atomic(meta_path, m); saved.append(name)

    report = run_cv(cfg, X, y, users, meta, cols, engine, out_dir, progress=progress, cancel=cancel,
                         save_model=save_model if (cfg.get("save_folds") or cfg.get("final_model")) else None)
    report["saved_models"] = saved
    report["dataset"] = job.get("dataset_label") or os.path.basename(root); report["dataset_path"] = root
    _atomic(os.path.join(out_dir, "report.json"), report)
    _atomic(os.path.join(out_dir, "cfg.json"), report["cfg"])
    return {"kind": "train", "run": run_id, "n_folds": report["n_folds"]}


def save_eval(root, res, gestures=None, data=None):
    """Keeps a summary of each evaluation (data_clean/classification/evals/) to compare models."""
    d = os.path.join(_runs_dir(root), "evals"); os.makedirs(d, exist_ok=True)
    eid = "eval_" + time.strftime("%Y%m%d_%H%M%S") + "_" + safe_name(res["model"])
    _atomic(os.path.join(d, eid + ".json"), {
        "id": eid, "model": res["model"], "when": time.strftime("%d/%m/%Y %H:%M"), "n": res["n"],
        "users": [p["participant"] for p in res["per_participant"]], "gestures": gestures or None,
        "acc": res["global"]["acc"], "f1": res["global"]["f1"], "overlap": res.get("overlap"), "data": data, "model_dataset": res.get("model_dataset")})


def worker_main(jd):
    job = json.load(open(os.path.join(jd, "job.json"), encoding="utf-8"))
    global STORE
    if job.get("store"): STORE = job["store"]
    cflag = os.path.join(jd, "cancel")
    cancel = lambda: os.path.exists(cflag)
    ppath = os.path.join(jd, "progress.json")
    last = [0.0]

    def progress(d):
        now = time.time()
        if now - last[0] < 0.25 and d.get("stage") not in ("fold_done", "tuning_end"): return
        last[0] = now
        _atomic(ppath, {"state": "running", **d, "t": now})

    mode = job["mode"]; root = job["clean_root"]
    try:
        if mode == "train": res = run_train(job, jd, progress, cancel)
        elif mode == "evaluate":
            res = evaluate_model(root, job["model"], users=job.get("users") or None, gestures=job.get("gestures") or None, cancel=cancel, data_root=job.get("data_root") or None)
            save_eval(root, res, job.get("gestures"), job.get("data_label") or None)
        elif mode == "predict": res = {"kind": "predict", **predict_one(root, job["model"], job["file"])}
        elif mode == "predict_many": res = {"kind": "predict_many", **predict_many(root, job["model"], job["files"])}
        elif mode == "inspect": res = {"kind": "inspect", **inspect_model(root, job["model"])}
        else: raise ValueError("mode inconnu")
        _atomic(os.path.join(jd, "result.json"), {"ok": True, "result": res})
        _atomic(ppath, {"state": "done", "stage": "done", "message": "Terminé", "t": time.time()})
    except Cancelled:
        _atomic(os.path.join(jd, "result.json"), {"ok": False, "cancelled": True, "error": "Annulé"})
        _atomic(ppath, {"state": "cancelled", "stage": "cancelled", "message": "Annulé", "t": time.time()})
    except Exception as e:
        _atomic(os.path.join(jd, "result.json"), {"ok": False, "error": f"{type(e).__name__}: {e}" if not isinstance(e, ValueError) else str(e),
                                                   "trace": traceback.format_exc()[-1500:]})
        _atomic(ppath, {"state": "error", "stage": "error", "message": str(e)[:300], "t": time.time()})





# ████████████████████████████████████████████████████████████
#  5. ROUTES FLASK
# ████████████████████████████████████████████████████████████


def reset_selected():
    """Replaced by register(); no effect as long as the routes are not registered."""


job_lock = threading.Lock()
job = {"running": False, "kind": None, "title": "", "started": 0, "finished": 0, "state": "idle", "run": None, "error": None,
       "result": None, "progress": {}, "jd": None}
_proc = {"p": None}

ARCHS = [{"id": "dense", "label": "Dense", "desc": "Aplatit la série puis 1 à 3 couches denses"},
         {"id": "cnn", "label": "CNN 1D", "desc": "1 à 3 blocs Conv1D + pooling, moyenne globale"},
         {"id": "lstm", "label": "LSTM", "desc": "1 à 3 couches BiLSTM puis dense"},
         {"id": "logreg", "label": "LogReg", "desc": "Régression logistique sur des statistiques par colonne (ML, sans TensorFlow)"},
         {"id": "svm", "label": "SVM", "desc": "SVM à noyau RBF sur des statistiques par colonne (ML, sans TensorFlow)"},
         {"id": "rf", "label": "Random Forest", "desc": "Forêt aléatoire sur des statistiques par colonne (ML, sans TensorFlow)"}]
HB_DEFAULTS = {"max_epochs": 20, "factor": 3, "iterations": 2, "patience": 3}
MANUAL_DEFAULTS = {"epochs": 20, "batch": 32, "lr": 0.001, "dropout": 0.2, "patience": 3}
MAX_SLOTS = 5                                    # layers / blocks adjustable at most in manual / copy mode


def _slots(**per_slot):
    """{'units_0': .., 'units_1': ..} for MAX_SLOTS layers; per_slot = {prefix: (values per layer …)}."""
    out = {}
    for pre, vals in per_slot.items():
        for i in range(MAX_SLOTS):
            out[f"{pre}_{i}"] = vals[min(i, len(vals) - 1)]
    return out


# Full hyperparameters per architecture (same names as the clf_keras_v2 builders): default value of the form.
ARCH_MANUAL_DEFAULTS = {
    "dense": {"lr": 0.001, "batch_size": 32, "n_layers": 2, **_slots(units=(128, 64, 32), dropout=(0.2,))},
    "cnn": {"lr": 0.001, "batch_size": 32, "n_conv_blocks": 2, "dense_units": 64,
            **_slots(filters=(32, 64, 64), kernel=(5,), dropout=(0.2,))},
    "lstm": {"lr": 0.001, "batch_size": 32, "n_lstm_layers": 1, "dense_units": 32,
             **_slots(lstm_units=(64, 32, 32), dropout=(0.2,))},
    # classic ML (clf_sklearn.py): svm.gamma = 0 → "scale", rf.max_depth = 0 → no limit
    "logreg": {"C": 1.0},
    "svm": {"C": 10.0, "gamma": 0.0},
    "rf": {"n_estimators": 300, "max_depth": 0, "min_samples_leaf": 1},
}
_HP_RANGES = {"lr": (1e-6, 1.0, float), "batch_size": (1, 1024, int), "dropout": (0.0, 0.9, float), "kernel": (1, 31, int),
              "n_layers": (1, MAX_SLOTS, int), "n_conv_blocks": (1, MAX_SLOTS, int), "n_lstm_layers": (1, MAX_SLOTS, int),
              "units": (1, 4096, int), "lstm_units": (1, 4096, int), "filters": (1, 2048, int), "dense_units": (1, 4096, int),
              "C": (1e-6, 1e6, float), "gamma": (0.0, 100.0, float), "n_estimators": (1, 5000, int), "max_depth": (0, 200, int),
              "min_samples_leaf": (1, 100, int)}


def clean_arch_hp(arch, raw):
    """Validates / completes the full hyperparameters of an architecture (missing values → default). Raises ValueError."""
    base = ARCH_MANUAL_DEFAULTS[arch]; raw = raw if isinstance(raw, dict) else {}
    out = {}
    for k, dv in base.items():
        v = raw.get(k, dv)
        lo, hi, typ = _HP_RANGES[re.sub(r"_\d+$", "", k)]
        try: v = typ(v)
        except (TypeError, ValueError): raise ValueError(f"Hyperparamètre invalide : {arch}.{k}")
        out[k] = max(lo, min(hi, v))
    return out


MAX_GRID_VALUES, MAX_GRID_POINTS = 12, 300


def _ml_grids():
    import clf_sklearn
    return clf_sklearn.GRIDS


def clean_grid(arch, raw):
    """Grid search of an ML model set in the page: {param: [values]} (missing param → default list). Raises ValueError."""
    base = _ml_grids()[arch]; raw = raw if isinstance(raw, dict) else {}
    out, n = {}, 1
    for k, dv in base.items():
        vals = raw.get(k, dv)
        if isinstance(vals, str): vals = [v for v in re.split(r"[,;\s]+", vals) if v]
        if not isinstance(vals, list) or not vals: raise ValueError(f"Grid search {arch} : aucune valeur pour {k}")
        lo, hi, typ = _HP_RANGES[k]
        clean = []
        for v in vals:
            try: v = typ(float(v))
            except (TypeError, ValueError): raise ValueError(f"Grid search {arch} : valeur invalide pour {k} : {v}")
            if not lo <= v <= hi: raise ValueError(f"Grid search {arch} : {k} = {v} hors de [{lo}, {hi}]")
            if v not in clean: clean.append(v)
        if len(clean) > MAX_GRID_VALUES: raise ValueError(f"Grid search {arch} : {MAX_GRID_VALUES} valeurs max pour {k}")
        out[k] = clean; n *= len(clean)
    if n > MAX_GRID_POINTS: raise ValueError(f"Grid search {arch} : {n} combinaisons, {MAX_GRID_POINTS} max")
    return out


def load_copy_hp(root, run_id, scope, archs):
    """Hyperparameters (actually used) of a finished run: {run, scope, rows:[{fold, arch, hp}], epochs, patience}.
    Raises ValueError if the run is not found or does not have all the requested architectures."""
    if not re.match(r"^run_[A-Za-z0-9_.-]+$", str(run_id or "")): raise ValueError("Run source invalide")
    rp = os.path.join(_runs_dir(root), run_id, "report.json")
    if not os.path.isfile(rp): raise ValueError("Run source introuvable")
    with open(rp, encoding="utf-8") as fh: rep = json.load(fh)
    rows = [{"fold": int(r["fold"]), "arch": r["arch"], "hp": r["hp"]} for r in rep.get("hyperparameters", []) if r.get("hp")]
    missing = [a for a in archs if not any(r["arch"] == a and r["fold"] == 0 for r in rows)]
    if missing: raise ValueError("Le run source n'a pas d'hyperparamètres pour : " + ", ".join(missing))
    c = rep.get("cfg", {})
    return {"run": run_id, "scope": "per_fold" if scope == "per_fold" else "fold1", "rows": [r for r in rows if r["arch"] in archs],
            "epochs": int(c.get("epochs") or 20), "patience": int(c.get("patience") or 3)}


def _runs_dir(root=None):
    """GLOBAL folder of runs, evaluations and jobs (root kept for compatibility)."""
    d = os.path.join(STORE, "classification"); os.makedirs(d, exist_ok=True); return d


_RUN_CACHE = {}          # report.json -> ((mtime_ns, size), history row): big reports are only read once


def list_runs(root):
    out = []
    for d in sorted(glob.glob(os.path.join(_runs_dir(root), "run_*")), reverse=True):
        rp = os.path.join(d, "report.json")
        try:
            st = os.stat(rp)
        except OSError:
            continue
        sig = (st.st_mtime_ns, st.st_size)
        hit = _RUN_CACHE.get(rp)
        if hit and hit[0] == sig:
            out.append(hit[1]); continue
        try:
            with open(rp, encoding="utf-8") as fh: r = json.load(fh)
        except Exception:
            continue
        s = r.get("summary", {})
        row = {"id": os.path.basename(d), "scenario": r.get("scenario"), "n_folds": r.get("n_folds"), "seconds": r.get("seconds"),
               "archs": r.get("cfg", {}).get("archs"), "tune_mode": r.get("cfg", {}).get("tune_mode"),
               "acc": {k: v["testAccuracy"]["mean"] for k, v in s.items()}, "when": time.strftime("%d/%m/%Y %H:%M", time.localtime(st.st_mtime)), "ts": st.st_mtime,
               "dataset": _short(r.get("dataset")), "dataset_path": r.get("dataset_path")}
        _RUN_CACHE[rp] = (sig, row)
        out.append(row)
    return out


def _clean_cfg(d, scan, root=None, copy_root=None):
    """Validates and normalizes the configuration sent by the interface. Raises ValueError."""
    scenario = d.get("scenario", "loso")
    if scenario not in ("loso", "ud"): raise ValueError("Scénario inconnu")
    archs = [a for a in (d.get("archs") or []) if a in ("dense", "cnn", "lstm") + ML_ARCHS]
    if not archs: raise ValueError("Choisir au moins une architecture")
    if len({a in ML_ARCHS for a in archs}) > 1: raise ValueError("Deep learning et machine learning ne se mélangent pas dans un même run")
    tm = d.get("tune_mode", "per_fold")
    if tm not in ("per_fold", "once", "manual", "copy"): raise ValueError("Mode de réglage inconnu")
    def gi(k, default, lo, hi):
        try: v = int(d.get(k, default))
        except (TypeError, ValueError): raise ValueError(f"Valeur invalide : {k}")
        return max(lo, min(hi, v))
    hb = {k: gi_v for k, gi_v in ((k, int(((d.get("hyperband") or {}).get(k, v)))) for k, v in HB_DEFAULTS.items())}
    hb["max_epochs"] = max(1, min(200, hb["max_epochs"])); hb["factor"] = max(2, min(10, hb["factor"]))
    hb["iterations"] = max(1, min(5, hb["iterations"])); hb["patience"] = max(1, min(30, hb["patience"]))
    mn = {**MANUAL_DEFAULTS, **{k: v for k, v in (d.get("manual") or {}).items() if k in MANUAL_DEFAULTS}}
    mn = {"epochs": max(1, min(500, int(mn["epochs"]))), "batch": int(mn["batch"]), "lr": float(mn["lr"]),
          "dropout": max(0.0, min(0.9, float(mn["dropout"]))), "patience": max(1, min(50, int(mn["patience"])))}
    if mn["batch"] not in (8, 16, 32, 64, 128): raise ValueError("Taille de batch invalide")
    cfg = {"scenario": scenario, "group_by": d.get("group_by", "rep") if d.get("group_by") in ("rep", "session") else "rep",
           "n_val": gi("n_val", 2 if scenario == "loso" else 1, 1, 10), "seed": gi("seed", 42, 0, 10 ** 9),
           "run_seed": gi("run_seed", gi("seed", 42, 0, 10 ** 9), 0, 10 ** 9), "archs": archs, "tune_mode": tm, "hyperband": hb, "manual": mn,
           "T": gi("T", 100, 10, 1000), "exclude_points": ([str(p) for p in d["exclude_points"]] if d.get("exclude_points") is not None else None),   # None → automatic default
           "users": [u for u in (d.get("users") or []) if u in scan["participants"]],
           "gestures": [g for g in (d.get("gestures") or []) if g in scan["gestures"]]}
    cfg["epochs"] = mn["epochs"] if tm == "manual" else hb["max_epochs"]
    cfg["patience"] = mn["patience"] if tm == "manual" else hb["patience"]
    if tm == "manual":                               # full hyperparameters per architecture
        mhp = (d.get("manual") or {}).get("hp") or {}
        cfg["manual"] = {**mn, "hp": {a: clean_arch_hp(a, mhp.get(a)) for a in archs}}
    elif tm == "copy":                               # same hyperparameters as a previous run
        if root is None: raise ValueError("Dossier de travail inconnu")
        cp = load_copy_hp(copy_root or root, d.get("copy_run"), d.get("copy_scope"), archs)
        cfg["copy_hp"] = cp; cfg["epochs"] = cp["epochs"]; cfg["patience"] = cp["patience"]
    if archs[0] in ML_ARCHS and tm in ("per_fold", "once"):      # grid search set in the page
        cfg["grid"] = {a: clean_grid(a, (d.get("grid") or {}).get(a)) for a in archs}
    sf = d.get("save_folds")
    cfg["save_folds"] = "all" if sf == "all" else None
    cfg["final_model"] = bool(d.get("final_model"))
    return cfg


def _index_subset(root, cfg):
    idx = index_clean_dir(root)
    keys = [k for k in sorted(idx) if (not cfg["users"] or k[0] in cfg["users"]) and (not cfg["gestures"] or k[1] in cfg["gestures"])]
    return [k[0] for k in keys], np.array([k[2] for k in keys]), keys


def plan(root, cfg):
    users, ids, keys = _index_subset(root, cfg)
    if not keys: raise ValueError("Aucune répétition pour cette sélection")
    folds = plan_folds(cfg["scenario"], users, ids, cfg["group_by"], cfg["n_val"], cfg["seed"])
    return folds, len(keys), len(set(k[1] for k in keys))


def _spawn(jd, jobdata):
    os.makedirs(jd, exist_ok=True)
    with open(os.path.join(jd, "job.json"), "w", encoding="utf-8") as f: json.dump({**jobdata, "store": STORE}, f)
    return subprocess.Popen([sys.executable, "-I", os.path.abspath(__file__), "--worker", jd], stdout=open(os.path.join(jd, "stdout.log"), "w"), stderr=subprocess.STDOUT,
                            cwd=HERE, env={**os.environ, "PYTHONPATH": HERE})


def _read(path):
    try:
        with open(path, encoding="utf-8") as f: return json.load(f)
    except Exception:
        return None


def _watch(p, jd, kind, cleanup):
    p.wait()
    time.sleep(0.2)
    res = _read(os.path.join(jd, "result.json"))
    with job_lock:
        job["finished"] = time.time(); job["running"] = False
        if res is None:
            tail = ""
            try: tail = open(os.path.join(jd, "stdout.log"), encoding="utf-8", errors="replace").read()[-600:]
            except Exception: pass
            job.update(state="error", error="Le processus de calcul s'est arrêté brutalement (mémoire insuffisante ?). " + tail)
        elif res.get("cancelled"): job.update(state="cancelled", error=None)
        elif not res.get("ok"): job.update(state="error", error=res.get("error"))
        else: job.update(state="done", error=None, result=res["result"], run=res["result"].get("run"))
    if cleanup: shutil.rmtree(jd, ignore_errors=True)


def start_job(root, kind, title, jobdata, cleanup=True):
    with job_lock:
        if job["running"]: return False, "Un calcul est déjà en cours"
        jd = os.path.join(_runs_dir(root), "_work", time.strftime("%Y%m%d_%H%M%S"))
        job.update(running=True, kind=kind, title=title, started=time.time(), finished=0, state="running", run=None, error=None,
                   result=None, progress={}, jd=jd)
    try:
        p = _spawn(jd, {**jobdata, "clean_root": root})
    except Exception as e:
        with job_lock: job.update(running=False, state="error", error=str(e))
        return False, str(e)
    _proc["p"] = p
    threading.Thread(target=_watch, args=(p, jd, kind, cleanup), daemon=True).start()
    return True, "ok"


def run_sync(root, jobdata, timeout=180):
    """Small synchronous job (inspecting an import, predicting a file)."""
    jd = os.path.join(_runs_dir(root), "_work", "s_" + str(time.time_ns()))
    try:
        p = _spawn(jd, {**jobdata, "clean_root": root})
        try: p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill(); raise ValueError("Délai dépassé")
        res = _read(os.path.join(jd, "result.json"))
        if res is None: raise ValueError("Le processus de calcul s'est arrêté brutalement")
        if not res.get("ok"): raise ValueError(res.get("error") or "Erreur")
        return res["result"]
    finally:
        shutil.rmtree(jd, ignore_errors=True)


def _short(label):
    """Old label "set · dataset · version" → "set · version" (display of models / runs already saved)."""
    p = str(label).split(" · ") if label else []
    return f"{p[0]} · {p[2]}" if len(p) == 3 else label


def register(app, clean_root_src, base_dir, busy=None):
    from flask import jsonify, request, send_file
    _default = clean_root_src if callable(clean_root_src) else (lambda: clean_root_src)
    _cfg_path = os.path.join(base_dir, "clf_datasets.json")
    try:
        with open(_cfg_path, encoding="utf-8") as fh: _st = json.load(fh)
    except Exception:
        _st = {}
    _st.setdefault("extra", []); _st.setdefault("selected", "")

    def _save_st():
        try:
            with open(_cfg_path, "w", encoding="utf-8") as fh: json.dump(_st, fh, ensure_ascii=False, indent=1)
        except Exception: pass

    def _has_data(d):
        try: return os.path.isdir(d) and bool(glob.glob(os.path.join(d, "Gesture_*")))
        except Exception: return False

    clean_base = os.path.join(base_dir, "data_clean")

    def _label(d):
        """The "set · dataset" label for a data_clean/<set>/<dataset> folder, otherwise the folder name."""
        try:
            rel = os.path.relpath(d, os.path.realpath(clean_base)).split(os.sep)
            if len(rel) == 2 and ".." not in rel:
                return f"{rel[0]} · {rel[1]}"                                   # cleaned dataset or version: "set · name"
        except ValueError: pass
        return os.path.basename(d) or d

    def _datasets():
        """Clean folders: the app's active dataset, all data_clean/<set>/<dataset>/ that contain Gesture_*,
        and those added by hand."""
        dflt = os.path.realpath(_default()); out = {dflt: True}
        try:
            for s in sorted(os.listdir(clean_base)):
                sd = os.path.join(clean_base, s)
                if not os.path.isdir(sd): continue
                for n in sorted(os.listdir(sd)):
                    d = os.path.realpath(os.path.join(sd, n))
                    if _has_data(d): out.setdefault(d, True)
        except Exception: pass
        for d in _st["extra"]:
            d = os.path.realpath(d)
            if _has_data(d): out.setdefault(d, True)
        return [d for d in out]

    def _reset_selected():
        """Called when the app's active set / dataset changes: the classifier follows the active one again."""
        _st["selected"] = ""; _save_st()
    globals()["reset_selected"] = _reset_selected

    def CR():
        sel = _st.get("selected")
        if sel and os.path.realpath(sel) in _datasets(): return os.path.realpath(sel)
        return _default()

    def _busy(): return busy() if busy else None

    @app.route("/api/clf/datasets")
    def clf_datasets():
        cur = os.path.realpath(CR()); items = []
        for d in _datasets():
            try: sc = scan_clean_dir(d, with_columns=False)
            except Exception: sc = {"files": 0, "participants": [], "gestures": []}
            items.append({"path": d, "name": _label(d), "files": sc["files"], "total_files": sc.get("total_files", sc["files"]), "participants": len(sc["participants"]),
                          "gestures": len(sc["gestures"]), "current": d == cur, "removable": d in [os.path.realpath(x) for x in _st["extra"]]})
        return jsonify({"items": items, "current": cur})

    @app.route("/api/clf/datasets/select", methods=["POST"])
    def clf_datasets_select():
        d = request.get_json(silent=True) or {}
        b = _busy()
        if b: return jsonify({"ok": False, "msg": b})
        path = os.path.realpath(d.get("path", ""))
        if path not in _datasets(): return jsonify({"ok": False, "msg": "Dataset inconnu"})
        _st["selected"] = path; _save_st()
        threading.Thread(target=_warm, daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/api/clf/datasets/add", methods=["POST"])
    def clf_datasets_add():
        d = request.get_json(silent=True) or {}
        path = os.path.realpath(os.path.expanduser((d.get("path") or "").strip()))
        if not _has_data(path): return jsonify({"ok": False, "msg": "Dossier invalide"})
        if path not in [os.path.realpath(x) for x in _st["extra"]]: _st["extra"].append(path)
        _st["selected"] = path; _save_st()
        return jsonify({"ok": True})

    @app.route("/api/clf/datasets/remove", methods=["POST"])
    def clf_datasets_remove():
        d = request.get_json(silent=True) or {}
        path = os.path.realpath(d.get("path", ""))
        _st["extra"] = [x for x in _st["extra"] if os.path.realpath(x) != path]
        if os.path.realpath(_st.get("selected") or "") == path: _st["selected"] = ""
        _save_st()
        return jsonify({"ok": True})

    def _warm():
        """Warms up the caches (index, columns, history) at startup: the first click on "Classifier" is then instant."""
        try:
            root = CR()
            if scan_clean_dir(root)["files"]: list_points(root)
            list_runs(root); list_models(root)
        except Exception:
            pass
    threading.Thread(target=_warm, daemon=True).start()

    @app.route("/classifier")
    def clf_page():
        for p in (os.path.join(base_dir, "templates", "classifier_v2.html"), os.path.join(base_dir, "classifier_v2.html")):
            if os.path.isfile(p): return send_file(p, mimetype="text/html")
        return "classifier_v2.html introuvable (le mettre dans templates/)", 404

    @app.route("/api/clf/meta")
    def clf_meta():
        root = CR()
        try: s = scan_clean_dir(root); pts = list_points(root) if s["files"] else []
        except Exception: s, pts = {"files": 0, "participants": [], "gestures": [], "reps": [], "sessions": []}, []
        return jsonify({**s, "points": pts, "libs": available_libs(), "formats": sorted(FORMATS), "archs": ARCHS,
                        "hb_defaults": HB_DEFAULTS, "manual_defaults": MANUAL_DEFAULTS, "arch_defaults": ARCH_MANUAL_DEFAULTS, "models": list_models(root),
                        "default_excluded": default_excluded_points(root),
                        "train_ready": bool(os.environ.get("CLF_ENGINE_MODULE") or available_libs()["tensorflow"]), "ml_archs": list(ML_ARCHS), "ml_grids": _ml_grids(), "cpus": os.cpu_count() or 1})

    @app.route("/api/clf/validate", methods=["POST"])
    def clf_validate():
        d = request.get_json(silent=True) or {}
        try:
            root = CR(); scan = scan_clean_dir(root)
            if not scan["files"]: return jsonify({"ok": False, "msg": "Aucun fichier nettoyé"})
            cfg = _clean_cfg(d, scan, root, _copy_root(d))
            folds, n, n_g = plan(root, cfg)
            n_trials = None
            if cfg["tune_mode"] == "per_fold":     # order of magnitude: Hyperband trials per architecture and per fold
                hb = cfg["hyperband"]; import math
                s_max = int(math.log(hb["max_epochs"]) / math.log(hb["factor"]) + 1e-9)
                trials = sum(int(math.ceil((s_max + 1) / (s + 1) * hb["factor"] ** s)) for s in range(s_max, -1, -1)) * hb["iterations"]
                import clf_sklearn
                n_dl = len([a for a in cfg["archs"] if a not in ML_ARCHS])
                n_trials = (trials * n_dl + sum(len(clf_sklearn.grid_points(a, (cfg.get("grid") or {}).get(a))) for a in cfg["archs"] if a in ML_ARCHS)) * len(folds)
            return jsonify({"ok": True, "folds": folds, "n_samples": n, "n_classes": n_g, "n_trainings": len(folds) * len(cfg["archs"]),
                            "n_trials": n_trials, "cfg": {k: cfg[k] for k in ("scenario", "tune_mode")}})
        except ValueError as e:
            return jsonify({"ok": False, "msg": str(e)})

    @app.route("/api/clf/start", methods=["POST"])
    def clf_start():
        m = _busy()
        if m: return jsonify({"ok": False, "msg": m})
        d = request.get_json(silent=True) or {}
        root = CR()
        try:
            scan = scan_clean_dir(root)
            cfg = _clean_cfg(d, scan, root, _copy_root(d))
            plan(root, cfg)
        except ValueError as e:
            return jsonify({"ok": False, "msg": str(e)})
        libs = available_libs()
        has_dl = any(a not in ML_ARCHS for a in cfg["archs"])
        if not libs["sklearn"] and not has_dl:
            return jsonify({"ok": False, "msg": "scikit-learn manquant : pip install scikit-learn"})
        if has_dl and not libs["tensorflow"] and not os.environ.get("CLF_ENGINE_MODULE"):
            return jsonify({"ok": False, "msg": "TensorFlow manquant : pip install tensorflow keras-tuner scikit-learn"})
        if has_dl and cfg["tune_mode"] not in ("manual", "copy") and not os.environ.get("CLF_ENGINE_MODULE"):
            if not libs["keras_tuner"]:
                return jsonify({"ok": False, "msg": "keras-tuner manquant : pip install keras-tuner"})
            if not libs["tensorboard"]:
                return jsonify({"ok": False, "msg": "tensorboard manquant : pip install tensorboard"})
        nm = re.sub(r"[^0-9A-Za-z]+", "_", safe_name(str(d.get("name") or ""))).strip("_")[:40]
        run_id = "run_" + (nm + "_" if nm else "") + time.strftime("%Y%m%d_%H%M%S") + "_" + cfg["scenario"]
        out_dir = os.path.join(_runs_dir(root), run_id)
        ok, msg = start_job(root, "train", f"{cfg['scenario'].upper()} · {', '.join(cfg['archs'])}",
                            {"mode": "train", "cfg": cfg, "out_dir": out_dir, "dataset_label": _label(root)})
        return jsonify({"ok": ok, "msg": msg, "run": run_id})

    @app.route("/api/clf/cancel", methods=["POST"])
    def clf_cancel():
        with job_lock: jd, running = job.get("jd"), job["running"]
        if running and jd and os.path.isdir(jd):
            open(os.path.join(jd, "cancel"), "w").close()
            p = _proc["p"]
            def _force():
                time.sleep(15)                              # grace period: a Hyperband may finish its epoch
                if p and p.poll() is None: p.terminate()
            threading.Thread(target=_force, daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/api/clf/status")
    def clf_status():
        with job_lock: s = {k: v for k, v in job.items() if k not in ("jd",)}; jd = job.get("jd")
        if s["running"] and jd: s["progress"] = _read(os.path.join(jd, "progress.json")) or {}
        s["elapsed"] = round((s["finished"] or time.time()) - s["started"], 1) if s["started"] else 0
        return jsonify(s)

    def _copy_root(d):
        """Dataset from which to copy the hyperparameters (field copy_dataset); only a known dataset, otherwise the current dataset."""
        c = os.path.realpath(str(d.get("copy_dataset") or "")) if d.get("copy_dataset") else ""
        return c if c and c in _datasets() else None

    # ── migration: the old per-dataset layout (<dataset>/models, <dataset>/classification) → global space ──
    def _migrate_legacy():
        mp = os.path.join(STORE, "migrated.json"); os.makedirs(STORE, exist_ok=True)
        done = (_read(mp) or {}).get("done", [])
        changed = False
        for ds in _datasets():
            if ds in done: continue
            label = _label(ds); tag = {"dataset": label, "dataset_path": ds}; renamed = {}
            try:
                # models
                src = os.path.join(ds, "models"); dst = models_dir()
                for jf in glob.glob(os.path.join(src, "*.json")):
                    m = _read(jf)
                    if not m or not m.get("file") or not os.path.exists(os.path.join(src, m["file"])): continue
                    name = m.get("name") or os.path.splitext(os.path.basename(jf))[0]
                    old = _read(os.path.join(dst, safe_name(name) + ".json"))
                    if old and old.get("created") == m.get("created") and old.get("file") == m.get("file"): continue
                    new, k = safe_name(name), 2
                    while os.path.exists(os.path.join(dst, new + ".json")): new = f"{safe_name(name)}_{k}"; k += 1
                    def cp(fn, nf):
                        a, b = os.path.join(src, fn), os.path.join(dst, nf)
                        if os.path.isdir(a): shutil.copytree(a, b)
                        else: shutil.copy2(a, b)
                    nf = m["file"] if new == safe_name(name) else new + os.path.splitext(m["file"])[1]
                    cp(m["file"], nf); m["file"] = nf
                    if m.get("scaler_file"):
                        ns = m["scaler_file"] if new == safe_name(name) else new + "_" + m["scaler_file"]
                        cp(m["scaler_file"], ns); m["scaler_file"] = ns
                    m["name"] = new; renamed[name] = new
                    if m.get("source") == "trained": m.setdefault("dataset", label); m.setdefault("dataset_path", ds)
                    _atomic(os.path.join(dst, new + ".json"), m)
                # runs
                rsrc = os.path.join(ds, "classification"); rdst = _runs_dir()
                for rd in glob.glob(os.path.join(rsrc, "run_*")):
                    if not os.path.isfile(os.path.join(rd, "report.json")): continue
                    nm, k = os.path.basename(rd), 2
                    if os.path.isdir(os.path.join(rdst, nm)):
                        if os.path.getmtime(os.path.join(rdst, nm, "report.json")) == os.path.getmtime(os.path.join(rd, "report.json")): continue
                        while os.path.isdir(os.path.join(rdst, f"{os.path.basename(rd)}_{k}")): k += 1
                        nm = f"{os.path.basename(rd)}_{k}"
                    shutil.copytree(rd, os.path.join(rdst, nm))
                    rp = os.path.join(rdst, nm, "report.json"); rep_ = _read(rp)
                    if rep_ is not None and not rep_.get("dataset"):
                        rep_.update(tag); st = os.stat(rp); _atomic(rp, rep_); os.utime(rp, (st.st_atime, st.st_mtime))
                # evaluations
                edst = os.path.join(rdst, "evals"); os.makedirs(edst, exist_ok=True)
                for ef in glob.glob(os.path.join(rsrc, "evals", "eval_*.json")):
                    if os.path.exists(os.path.join(edst, os.path.basename(ef))): continue
                    e = _read(ef)
                    if e is None: continue
                    e["data"] = e.get("data") or label; e["model"] = renamed.get(e.get("model"), e.get("model"))
                    _atomic(os.path.join(edst, os.path.basename(ef)), e)
            except Exception as ex:
                print("  [classifieur] migration de", ds, "incomplète :", ex); continue
            done.append(ds); changed = True
        if changed: _atomic(mp, {"done": done})
    try: _migrate_legacy()
    except Exception as ex: print("  [classifieur] migration impossible :", ex)

    @app.route("/api/clf/compat")
    def clf_compat():
        """Is the model compatible with this dataset? (warning only, never blocking)"""
        dr = os.path.realpath(request.args.get("dataset", "") or CR())
        if dr not in _datasets(): return jsonify({"error": "dataset inconnu"}), 404
        try: m = get_model(CR(), request.args.get("model", ""))
        except ValueError: return jsonify({"error": "modèle introuvable"}), 404
        sc = scan_clean_dir(dr, with_columns=False)
        out = {"missing_gestures": [], "extra_gestures": [], "missing_features": [], "ok": True}
        if m.get("classes"):
            out["missing_gestures"] = [c for c in m["classes"] if c not in sc["gestures"]]
            out["extra_gestures"] = [g for g in sc["gestures"] if g not in m["classes"]]
        if m.get("features") and sc["files"]:
            try: cols = set(dataset_columns(dr)[0])
            except Exception: cols = None
            if cols is not None: out["missing_features"] = [c for c in m["features"] if c not in cols]
        out["ok"] = not (out["missing_gestures"] or out["missing_features"])
        return jsonify(out)

    @app.route("/api/clf/dataset_scan")
    def clf_dataset_scan():
        """Participants / gestures of a known dataset (to choose what to evaluate a model on)."""
        dr = os.path.realpath(request.args.get("path", ""))
        if dr not in _datasets(): return jsonify({"error": "inconnu"}), 404
        s = scan_clean_dir(dr, with_columns=False)
        return jsonify({"participants": s["participants"], "gestures": s["gestures"], "files": s["files"]})

    @app.route("/api/clf/runs/all")
    def clf_runs_all():
        """Same as /api/clf/runs: the history is global (kept for compatibility)."""
        return jsonify({"runs": sorted(list_runs(CR()), key=lambda r: r.get("ts", 0), reverse=True), "current": os.path.realpath(CR())})

    @app.route("/api/clf/runs")
    def clf_runs(): return jsonify({"runs": sorted(list_runs(CR()), key=lambda r: r.get("ts", 0), reverse=True)})

    def _run_dir(rid):
        if not re.fullmatch(r"run_[0-9A-Za-z_]+", rid or ""): return None
        d = os.path.join(_runs_dir(), rid)
        return d if os.path.isdir(d) else None

    @app.route("/api/clf/run/<rid>")
    def clf_run(rid):
        d = _run_dir(rid)
        r = d and _read(os.path.join(d, "report.json"))
        return jsonify(r) if r else (jsonify({"error": "introuvable"}), 404)

    @app.route("/api/clf/run/<rid>/file/<name>")
    def clf_run_file(rid, name):
        d = _run_dir(rid); name = os.path.basename(name)
        if not d or not os.path.isfile(os.path.join(d, name)) or not name.endswith((".csv", ".json")): return jsonify({"error": "introuvable"}), 404
        return send_file(os.path.join(d, name), as_attachment=True, download_name=f"{rid}_{name}")

    @app.route("/api/clf/run/<rid>", methods=["DELETE"])
    def clf_run_delete(rid):
        d = _run_dir(rid)
        if not d: return jsonify({"ok": False, "msg": "introuvable"}), 404
        shutil.rmtree(d, ignore_errors=True); return jsonify({"ok": True})

    # ── models ──
    @app.route("/api/clf/models")
    def clf_models(): return jsonify({"models": list_models(CR()), "dir": os.path.abspath(models_dir(CR()))})

    @app.route("/api/clf/models/import", methods=["POST"])
    def clf_import():
        m = _busy()
        if m: return jsonify({"ok": False, "msg": m})
        f = request.files.get("file")
        if not f or not f.filename: return jsonify({"ok": False, "msg": "Aucun fichier"})
        sc = request.files.get("scaler")
        root = CR(); name = None
        try:
            name = stage_import(root, f.filename, f.read(), request.form.get("name", ""), request.form.get("norm", "none"),
                                    request.form.get("exclude_points", ""), request.form.get("classes", ""),
                                    request.form.get("T", ""), sc.read() if sc and sc.filename else None, sc.filename if sc else "",
                                    request.form.get("seen", ""))
            info = run_sync(root, {"mode": "inspect", "model": name})
            return jsonify({"ok": True, "model": info})
        except ValueError as e:
            if name:
                try: delete_model(root, name)
                except Exception: pass
            return jsonify({"ok": False, "msg": str(e)})

    def _evals_dir(): return os.path.join(_runs_dir(CR()), "evals")

    @app.route("/api/clf/evals")
    def clf_evals():
        out = [r for r in (_read(f) for f in sorted(glob.glob(os.path.join(_evals_dir(), "eval_*.json")), reverse=True)) if r]
        return jsonify({"evals": out})

    @app.route("/api/clf/evals/<eid>", methods=["DELETE"])
    def clf_eval_delete(eid):
        if not re.fullmatch(r"eval_[0-9A-Za-z_.-]+", eid or ""): return jsonify({"ok": False}), 404
        p = os.path.join(_evals_dir(), eid + ".json")
        if os.path.isfile(p): os.remove(p)
        return jsonify({"ok": True})

    @app.route("/api/clf/models/<name>", methods=["DELETE"])
    def clf_model_delete(name):
        try: delete_model(CR(), name)
        except ValueError as e: return jsonify({"ok": False, "msg": str(e)}), 404
        return jsonify({"ok": True})

    @app.route("/api/clf/models/<name>/download")
    def clf_model_download(name):
        try: m = get_model(CR(), name)
        except ValueError: return jsonify({"error": "introuvable"}), 404
        if os.path.isdir(m["path"]): return jsonify({"error": "dossier"}), 400
        return send_file(m["path"], as_attachment=True)

    @app.route("/api/clf/template")
    def clf_template():
        fn = "model_template_ml.py" if request.args.get("kind") == "ml" else "model_template.py"
        p = os.path.join(HERE, fn)
        if not os.path.isfile(p): return jsonify({"error": f"{fn} introuvable à côté de classifier_v2.py"}), 404
        return send_file(p, as_attachment=True, download_name=fn)

    @app.route("/api/clf/tree")
    def clf_tree():
        idx = index_clean_dir(CR()); out = {}
        for (u, g, i), p in sorted(idx.items()):
            out.setdefault(g, {}).setdefault(u, []).append({"id": i, "rel": os.path.relpath(p, CR()).replace(os.sep, "/")})
        return jsonify(out)

    @app.route("/api/clf/predict", methods=["POST"])
    def clf_predict():
        d = request.get_json(silent=True) or {}
        try:
            if d.get("files"):
                return jsonify({"ok": True, **run_sync(CR(), {"mode": "predict_many", "model": d.get("model", ""), "files": d["files"]}, timeout=900)})
            return jsonify({"ok": True, **run_sync(CR(), {"mode": "predict", "model": d.get("model", ""), "file": d.get("file", "")})})
        except ValueError as e: return jsonify({"ok": False, "msg": str(e)})

    @app.route("/api/clf/evaluate", methods=["POST"])
    def clf_evaluate():
        m = _busy()
        if m: return jsonify({"ok": False, "msg": m})
        d = request.get_json(silent=True) or {}
        dr = os.path.realpath(str(d.get("data_dataset") or "")) if d.get("data_dataset") else ""
        if dr and dr not in _datasets(): return jsonify({"ok": False, "msg": "Dataset d'évaluation inconnu"})
        dr = dr or os.path.realpath(CR()); other = True
        ok, msg = start_job(CR(), "evaluate", "Évaluation · " + str(d.get("model", "")),
                            {"mode": "evaluate", "model": d.get("model", ""), "users": d.get("users") or [], "gestures": d.get("gestures") or [],
                             "data_root": dr, "data_label": _label(dr)})
        return jsonify({"ok": ok, "msg": msg})


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2])
