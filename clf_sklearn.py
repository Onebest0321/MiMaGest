"""
clf_sklearn.py — classic machine learning for the "Classifier" page (scikit-learn).

Three models: Logistic regression ("logreg"), SVM ("svm", RBF kernel) and Random Forest ("rf").
They are trained by the same cross-validation loop as the Keras networks (classifier_v2.run_cv),
on the same folds, with the same per-fold StandardScaler, and give the same result files.

A classic model does not read the time series directly. Each repetition (T × F, already
z-scored per column) is turned into one vector by `ml_features`:
    for each column: mean, standard deviation, minimum, maximum,
                     then the mean of each of N_SEGMENTS equal time segments (the shape of the gesture).
→ (4 + N_SEGMENTS) × F numbers per repetition.

The saved model file (.joblib) is a dict:
    {"format": "mimagest-sklearn", "features": "stats", "n_segments": 5,
     "model": <scikit-learn estimator trained on ml_features>,
     "mean": [...], "scale": [...]   ← per-column z-score of the training data (None if already applied),
     "n_classes": C, "T": T, "F": F, "classes": [...] or None}
so it takes the raw resampled repetitions, like the Keras models saved by the page.
model_template_ml.py writes exactly the same file.

Dependencies: numpy, scikit-learn, joblib (installed with scikit-learn). TensorFlow is NOT needed.
"""
import numpy as np

ML_ARCHS = ("logreg", "svm", "rf")
N_SEGMENTS = 5
BUNDLE_FORMAT = "mimagest-sklearn"


# ═════════════════════════════════════════════════════════════
#  FEATURES  (model_template_ml.py contains an identical copy)
# ═════════════════════════════════════════════════════════════
def ml_features(X, n_segments=N_SEGMENTS):
    """X (N, T, F) → (N, (4 + n_segments) × F): mean, std, min, max, then the mean of n_segments time segments."""
    X = np.asarray(X, dtype=np.float32)
    parts = [X.mean(1), X.std(1), X.min(1), X.max(1)]
    for seg in np.array_split(np.arange(X.shape[1]), n_segments):
        parts.append(X[:, seg, :].mean(1))
    return np.concatenate(parts, axis=1)


# ═════════════════════════════════════════════════════════════
#  MODELS AND HYPERPARAMETERS
# ═════════════════════════════════════════════════════════════
# Default values (manual mode) — numbers only, so that the page can show them in the form.
#   svm.gamma = 0  → "scale" (scikit-learn default);   rf.max_depth = 0 → no limit
DEFAULTS = {
    "logreg": {"C": 1.0},
    "svm":    {"C": 10.0, "gamma": 0.0},
    "rf":     {"n_estimators": 300, "max_depth": 0, "min_samples_leaf": 1},
}
# Search grid (tuning modes "per fold" / "fold 1"): every combination is trained on train and scored on val.
GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "svm":    {"C": [0.1, 1.0, 10.0, 100.0], "gamma": [0.0, 0.01, 0.001]},
    "rf":     {"n_estimators": [200, 500], "max_depth": [0, 10, 20], "min_samples_leaf": [1, 3]},
}


def grid_points(arch, grid=None):
    """Every combination of the grid (the page's grid if given, otherwise GRIDS[arch])."""
    import itertools
    g = {**GRIDS[arch], **(grid or {})}
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]


def build_estimator(arch, hp, seed=0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if arch == "logreg":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(C=float(hp["C"]), max_iter=3000))
    if arch == "svm":
        from sklearn.svm import SVC
        gamma = float(hp["gamma"]) or "scale"
        # no probability=True: its internal calibration can contradict predict() on small folds
        # (and it is deprecated in scikit-learn 1.9); the scores come from decision_function (see full_proba)
        return make_pipeline(StandardScaler(), SVC(C=float(hp["C"]), gamma=gamma, decision_function_shape="ovr", random_state=seed))
    if arch == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=int(hp["n_estimators"]), max_depth=int(hp["max_depth"]) or None,
                                      min_samples_leaf=int(hp["min_samples_leaf"]), n_jobs=-1, random_state=seed)
    raise ValueError(f"Modèle ML inconnu : {arch}")


def full_proba(est, F, n_classes):
    """Scores with one column per class 0..n_classes-1 (even if a class was absent from train).
    predict_proba if the estimator has it, otherwise softmax of decision_function (e.g. SVC)."""
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(F)
    else:
        d = np.asarray(est.decision_function(F), dtype=np.float64)
        if d.ndim == 1: d = np.stack([-d, d], axis=1)
        e = np.exp(d - d.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
    out = np.zeros((p.shape[0], n_classes), dtype=np.float64)
    for j, c in enumerate(est.classes_):
        out[:, int(c)] = p[:, j]
    return out


# ═════════════════════════════════════════════════════════════
#  ENGINE  (same interface as clf_keras_v2.KerasEngine, used by classifier_v2.run_cv)
# ═════════════════════════════════════════════════════════════
class Cancelled(Exception):
    pass


class SklearnEngine:
    def __init__(self, cancel=None, seed=0, cancelled_exc=Cancelled):
        self.cancel = cancel or (lambda: False)
        self.seed = seed
        self.exc = cancelled_exc

    def tune(self, arch, input_shape, n_classes, Xtr, ytr, Xval, yval, hb, tag, on_trial=None):
        """Grid search: each combination is trained on train and scored on val (accuracy). Returns the best."""
        Ftr, Fval = ml_features(Xtr), ml_features(Xval)
        best, best_acc = None, -1.0
        grid = hb if isinstance(hb, dict) and all(isinstance(v, list) for v in hb.values()) else None   # grid set in the page
        for n, hp in enumerate(grid_points(arch, grid), 1):
            if self.cancel(): raise self.exc()
            est = build_estimator(arch, hp, self.seed).fit(Ftr, ytr)
            acc = float((est.predict(Fval) == yval).mean()) if len(yval) else 0.0
            if acc > best_acc:
                best, best_acc = dict(hp), acc
            if on_trial: on_trial(n)
        return best

    def manual_hp(self, arch, manual):
        return {**DEFAULTS[arch], **((manual.get("hp") or {}).get(arch) or {})}

    def fixed_hp(self, values):
        return dict(values)

    def hp_values(self, hp):
        return {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
                for k, v in dict(hp).items()}

    def train_eval(self, arch, hp, input_shape, n_classes, Xtr, ytr, Xval, yval, Xte, epochs, patience, on_epoch=None):
        """Trains on train only (val was used for the hyperparameter choice), returns (model, test probabilities)."""
        if self.cancel(): raise self.exc()
        est = build_estimator(arch, hp, self.seed).fit(ml_features(Xtr), ytr)
        model = {"est": est, "arch": arch, "n_classes": n_classes, "T": input_shape[0], "F": input_shape[1]}
        return model, full_proba(est, ml_features(Xte), n_classes)

    def save(self, model, path, scaler=None):
        """Standalone .joblib: the per-fold normalization is stored in the file → returns True."""
        import joblib
        joblib.dump(make_bundle(model["est"], model["n_classes"], model["T"], model["F"],
                                None if scaler is None else scaler.mean_, None if scaler is None else scaler.scale_), path)
        return scaler is not None

    def free(self, model):
        pass


def make_bundle(est, n_classes, T, F, mean=None, scale=None, classes=None):
    return {"format": BUNDLE_FORMAT, "features": "stats", "n_segments": N_SEGMENTS, "model": est,
            "mean": None if mean is None else [float(v) for v in mean],
            "scale": None if scale is None else [float(v) for v in scale],
            "n_classes": int(n_classes), "T": int(T), "F": int(F), "classes": classes}


# ═════════════════════════════════════════════════════════════
#  LOADING  (used by classifier_v2.SklearnPredictor)
# ═════════════════════════════════════════════════════════════
class BundlePredictor:
    """proba(X): X (N, T, F) raw resampled → (N, C)."""

    def __init__(self, path):
        import joblib
        b = joblib.load(path)
        if not (isinstance(b, dict) and b.get("format") == BUNDLE_FORMAT):
            raise ValueError("ce .joblib n'est pas un modèle MiMaGest : utilise model_template_ml.py pour l'exporter")
        self.b = b
        self.est = b["model"]
        self.T, self.F, self.n_classes = b.get("T"), b.get("F"), b.get("n_classes")
        self.class_names = [str(c) for c in b["classes"]] if b.get("classes") else None
        self.columns = b.get("columns")
        self.mean = None if b.get("mean") is None else np.asarray(b["mean"], np.float32)
        self.scale = None if b.get("scale") is None else np.where(np.asarray(b["scale"], np.float32) == 0, 1, np.asarray(b["scale"], np.float32))

    def proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.mean is not None:
            X = (X - self.mean) / self.scale
        F = ml_features(X, int(self.b.get("n_segments", N_SEGMENTS)))
        return full_proba(self.est, F, int(self.n_classes or len(self.est.classes_)))
