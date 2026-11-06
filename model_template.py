"""
model_template.py — model script: trains YOUR model on the TapStrap dataset and prepares it for import into the "Classifier" page.

Two ways to use the page:
  1. Use our classifiers (Dense / CNN / LSTM): nothing to do, everything is in the page.
  2. Do it your own way (this script) and import your model in "Models › Import a model".

This script does, in this order:
  A. loads the cleaned repetitions exactly like the page (same files, same columns, T time steps);
  B. splits train / validation / test by PARTICIPANT (the test set is never seen during training);
  C. normalizes: per-column z-score, mean and standard deviation computed on TRAIN only  ← the page's rule;
  D. calls YOUR model (the only part to modify: build_model);
  E. evaluates on the test participant;
  F. saves  <NAME>.keras  +  <NAME>_normalization.json  → to import together in the page.

Usage:   python model_template.py "path/to/data_clean"
Dependencies: numpy pandas scikit-learn tensorflow
The page imports Keras models only (.keras or .h5).
"""
import os, re, sys, glob, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ═════════════════════════════════════════════════════════════
#  SETTINGS
# ═════════════════════════════════════════════════════════════
NAME = "mon_modele"                  # name of the produced files
T = 100                              # time steps: each repetition is resampled to T points (linear interpolation)
EXCLUDE_POINTS = ["TAP1_Accel"]      # sensor points NOT to use (like the page: TAP1_Accel excluded by default)
TEST_USER = "P01"                    # participant held out for testing; None = everything is used for training (no evaluation)
N_VAL_USERS = 2                      # validation participants (drawn at random among the others)
SEED = 42
EPOCHS, BATCH = 30, 32


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
#  D. YOUR MODEL  ← the only part to modify
# ═════════════════════════════════════════════════════════════
def build_model(input_shape, n_classes):
    """input_shape = (T, number_of_columns). Returns a compiled Keras model."""
    from tensorflow import keras
    from tensorflow.keras import layers
    m = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, 5, activation="relu", padding="same"),     # ← replace with your architecture
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),               # the last layer must output n_classes probabilities
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


# ═════════════════════════════════════════════════════════════
#  PROGRAMME
# ═════════════════════════════════════════════════════════════
def main(root):
    X, y, users, cols = load_dataset(root)
    classes = sorted(set(y))
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
    Xtr, Xva, Xte = norm(X[tr]), norm(X[va]), norm(X[te]) if te.any() else None

    # D/E. training + evaluation
    model = build_model((T, len(cols)), len(classes))
    from tensorflow import keras
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(Xtr, y_id[tr], validation_data=(Xva, y_id[va]), epochs=EPOCHS, batch_size=BATCH, callbacks=[stop], verbose=2)
    predict = lambda A: model.predict(A, verbose=0).argmax(1)
    if Xte is not None:
        print(f"\nAccuracy sur {TEST_USER} (jamais vu à l'entraînement) : {(predict(Xte) == y_id[te]).mean():.3f}")

    # F. save: model + normalization file
    model.save(NAME + ".keras"); model_file = NAME + ".keras"
    with open(NAME + "_normalization.json", "w", encoding="utf-8") as f:
        json.dump({"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "columns": cols, "classes": classes, "T": T,
                   "note": "z-score par colonne, calculé sur le train du script model_template.py"}, f)
    print(f"\nFichiers prêts à importer dans la page : {model_file}  +  {NAME}_normalization.json")
    print("→ Classifieur › Modèles › Importer un modèle : choisis le modèle, normalisation « Z-score », fichier de normalisation = le .json.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('Usage : python model_template.py "chemin/vers/data_clean"')
    main(sys.argv[1])
