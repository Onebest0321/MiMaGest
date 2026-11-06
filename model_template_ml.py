"""
model_template_ml.py — model script for CLASSIC MACHINE LEARNING (scikit-learn): trains YOUR model on the
TapStrap dataset and prepares it for import into the "Classifier" page.

The page already trains Logistic regression, SVM and Random Forest itself (choose them in "Architectures").
This script is for any other scikit-learn model, or for your own features.

This script does, in this order:
  A. loads the cleaned repetitions exactly like the page (same files, same columns, T time steps);
  B. splits train / validation / test by PARTICIPANT (the test set is never seen during training);
  C. normalizes: per-column z-score, mean and standard deviation computed on TRAIN only  ← the page's rule;
  D. turns each repetition into one vector (features) — same features as the page;
  E. calls YOUR model (the only part to modify: build_model), trains it, evaluates on the test participant;
  F. saves  <NAME>.joblib  (model + normalization inside) → to import alone in the page.

Usage:   python model_template_ml.py "path/to/data_clean"
Dependencies: numpy pandas scikit-learn  (TensorFlow is NOT needed)

Rules so that the page can read the file:
  * build_model must return a scikit-learn classifier (or Pipeline) with predict_proba or decision_function
    (SVC is fine without probability=True). Use only scikit-learn / numpy classes: a class defined in this
    script cannot be reloaded by the page.
  * If you change the features (FEATURES / ml_features), the page will not compute the same ones:
    keep FEATURES = "stats" for a model you want to import.
"""
import os, re, sys, glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ═════════════════════════════════════════════════════════════
#  SETTINGS
# ═════════════════════════════════════════════════════════════
NAME = "mon_modele_ml"               # name of the produced file
T = 100                              # time steps: each repetition is resampled to T points (linear interpolation)
EXCLUDE_POINTS = ["TAP1_Accel"]      # sensor points NOT to use (like the page: TAP1_Accel excluded by default)
TEST_USER = "P01"                    # participant held out for testing; None = everything is used for training (no evaluation)
N_VAL_USERS = 2                      # validation participants (drawn at random among the others)
SEED = 42
N_SEGMENTS = 5                       # do not change if you want to import the model (the page uses 5)


# ═════════════════════════════════════════════════════════════
#  D. YOUR MODEL  ← the only part to modify
# ═════════════════════════════════════════════════════════════
def build_model(n_classes):
    """Returns a scikit-learn classifier. Uncomment the one you want, or write yours."""
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier

    return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", random_state=SEED))
    # return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    # return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
    # return ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=SEED)
    # return GradientBoostingClassifier(random_state=SEED)
    # return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))


# ═════════════════════════════════════════════════════════════
#  A. LOADING  (identical to the page — do not modify if you want comparable results)
# ═════════════════════════════════════════════════════════════
FILENAME = re.compile(r"^P(\d+)_S(\d+)_Gesture_(\d+)_rep(\d+)_(\d+)_(\d+)_Clean\.csv$", re.I)
META_COLS = {"Participant", "Gesture", "Repetition", "Timestamp_LSL"}


def point_of(col):
    return re.sub(r"_[XYZ]$", "", col)             # 'TAP1_Thumb_X' -> 'TAP1_Thumb'


def resample(arr, n):
    src, dst = np.linspace(0, 1, arr.shape[0]), np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, arr[:, j]) for j in range(arr.shape[1])], axis=1).astype(np.float32)


def load_dataset(root):
    files = {}
    for gd in glob.glob(os.path.join(root, "Gesture_*")):
        for r, _d, fs in os.walk(gd):
            for f in fs:
                m = FILENAME.match(f)
                if m:
                    files[(f"P{int(m[1]):02d}", f"Gesture_{int(m[3]):02d}", int(m[2]) * 100 + int(m[4]))] = os.path.join(r, f)
    if not files:
        sys.exit("Aucun fichier *_Clean.csv trouvé dans " + root)
    first = pd.read_csv(files[sorted(files)[0]], nrows=3)
    cols = [c for c in first.columns if c not in META_COLS and first[c].dtype.kind in "fi" and point_of(c) not in EXCLUDE_POINTS]
    X, y, users = [], [], []
    for (u, g, _i), path in sorted(files.items()):
        X.append(resample(pd.read_csv(path)[cols].values.astype(np.float32), T))
        y.append(g); users.append(u)
    return np.stack(X), np.array(y), np.array(users), cols


# ═════════════════════════════════════════════════════════════
#  D. FEATURES  (identical to clf_sklearn.ml_features of the page — do not modify for an import)
# ═════════════════════════════════════════════════════════════
def ml_features(X, n_segments=N_SEGMENTS):
    """X (N, T, F) → (N, (4 + n_segments) × F): mean, std, min, max, then the mean of n_segments time segments."""
    X = np.asarray(X, dtype=np.float32)
    parts = [X.mean(1), X.std(1), X.min(1), X.max(1)]
    for seg in np.array_split(np.arange(X.shape[1]), n_segments):
        parts.append(X[:, seg, :].mean(1))
    return np.concatenate(parts, axis=1)


# ═════════════════════════════════════════════════════════════
#  PROGRAMME
# ═════════════════════════════════════════════════════════════
def main(root):
    X, y, users, cols = load_dataset(root)
    classes = sorted(str(g) for g in set(y))
    y_id = np.array([classes.index(g) for g in y])
    print(f"{len(X)} répétitions · {len(set(users))} participants · {len(classes)} gestes · {len(cols)} colonnes · T={T}")

    # B. split by participant
    others = sorted(str(u) for u in set(users) - {TEST_USER})
    rng = np.random.RandomState(SEED); rng.shuffle(others)
    val_users, train_users = others[:N_VAL_USERS], others[N_VAL_USERS:]
    tr, va = np.isin(users, train_users), np.isin(users, val_users)
    te = users == TEST_USER if TEST_USER else np.zeros(len(users), bool)
    print(f"train {sorted(train_users)} · val {sorted(val_users)} · test {TEST_USER}")

    # C. normalization: per-column z-score, TRAIN statistics only
    sc = StandardScaler().fit(X[tr].reshape(-1, X.shape[2]))
    norm = lambda A: sc.transform(A.reshape(-1, A.shape[2])).reshape(A.shape).astype(np.float32)

    # D. features
    Ftr, Fva = ml_features(norm(X[tr])), ml_features(norm(X[va]))
    print(f"{Ftr.shape[1]} features par répétition")

    # E. training + evaluation (val: to compare settings; test: participant never seen)
    model = build_model(len(classes)).fit(Ftr, y_id[tr])
    if not (hasattr(model, "predict_proba") or hasattr(model, "decision_function")):
        sys.exit("Le modèle doit avoir predict_proba ou decision_function")
    if va.any():
        print(f"Accuracy val {sorted(val_users)} : {(model.predict(Fva) == y_id[va]).mean():.3f}")
    if te.any():
        Fte = ml_features(norm(X[te]))
        print(f"\nAccuracy sur {TEST_USER} (jamais vu à l'entraînement) : {(model.predict(Fte) == y_id[te]).mean():.3f}")

    # F. save: ONE file, model + normalization + description (same format as the models trained by the page)
    import joblib
    bundle = {"format": "mimagest-sklearn", "features": "stats", "n_segments": N_SEGMENTS, "model": model,
              "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
              "n_classes": len(classes), "T": T, "F": len(cols), "classes": classes, "columns": cols}
    joblib.dump(bundle, NAME + ".joblib")
    print(f"\nFichier prêt à importer dans la page : {NAME}.joblib")
    print("→ Classifieur › Modèles › Importer un modèle : choisis le .joblib, normalisation « aucune » (elle est dans le fichier).")
    print(f"  Participants vus à l'entraînement : {', '.join(sorted(train_users + val_users))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('Usage : python model_template_ml.py "chemin/vers/data_clean"')
    main(sys.argv[1])
