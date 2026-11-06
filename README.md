# TapSoftMod

A web app to **record hand gestures** with Tap wearables (TapStrap 2 on the fingers, TapXR on the forearm),
**check and clean** the recordings, and **train and test gesture classifiers** on them.
Everything runs on one Windows PC, in the browser.

- [What you can do](#what-you-can-do)
- [Install and start](#install-and-start)
- [How to use it, step by step](#how-to-use-it-step-by-step)
- [The pages](#the-pages)
- [Where the files go](#where-the-files-go)
- [Tips and problems](#tips-and-problems)
- [For developers: how the code is organised](#for-developers)

## What you can do

| Step | Page | What you do |
|---|---|---|
| Prepare | **Gesture sets** | Choose the gestures, the number of repetitions, their durations; the order is balanced across participants (Latin square) |
| Record | **Examiner** + **Participant** | Connect the sensors, run the session gesture by gesture, redo or skip a repetition, watch the signals live |
| Check | **Data** › Overview / Audit | See what is missing or duplicated, run 20+ quality checks per file, move bad repetitions aside (reversible) |
| Clean | **Data** › Preprocessing | Synchronise sensor and PC clocks, cut each repetition between START and STOP, put all sensors on one time grid |
| Explore | **Data** › Repetition / Statistics | Plot a raw or cleaned repetition, compare raw and clean, statistics per gesture, sensor, participant |
| Learn | **Classifier** › Training | Train deep learning (Dense, CNN, LSTM) or machine learning (LogReg, SVM, Random Forest) with leave-one-subject-out or user-dependent validation |
| Use | **Classifier** › Models / History | Evaluate a saved model on chosen participants, predict single repetitions, import your own model, compare runs |

All sensor streams and protocol markers are also published on **LSL** (Lab Streaming Layer).

## Install and start

### What you need

- **Windows 10 or 11** with **Bluetooth** (the Tap SDK only works on Windows).
- **Python 3.12** , from https://www.python.org/downloads/windows/, with
  **"Add Python to PATH"** ticked during installation.
- An **internet connection** for the first installation (not needed if the folder contains a `wheels\` folder).
- About **10 GB** of free disk space (TensorFlow is large).
- **TapXR and/or TapStrap devices**.

LSL (`pylsl`, with its library) and the Tap SDK are installed by `installer.bat`: nothing to install by hand.

### Install (once)

1. Open the local repository directory created by git clone (for example
C:\Projects\REPOSITORY\). Make sure you can write to it (not in "Program Files").
2. Double-click **`installer.bat`**. It:
   - finds Python 3.12,
   - creates a private environment in `venv\` (nothing is installed elsewhere on the PC),
   - installs the exact library versions of `requirements.txt`, including the Tap SDK, LSL, Flask,
     scikit-learn and TensorFlow.
   This takes several minutes. Wait for **"Installation finished"**.
### Start

1. Double-click **`lancer.bat`**. A black window opens: **leave it open** while you work; close it to stop the app.
2. After a few seconds the browser opens on the participant page. The pages are:

   | Page | French | English |
   |---|---|---|
   | Examiner (control of the session) | http://localhost:5000/examiner | http://localhost:5000/en/examiner |
   | Participant (screen shown to the participant) | http://localhost:5000 | http://localhost:5000/en |
   | Classifier | http://localhost:5000/classifier | http://localhost:5000/en/classifier |

### Update to a new version

Keep `data\`, `data_clean\`, `gesture_sets\`, `classifier_store\` and `static\videos\`, copy the new files
over the old ones, then run `installer.bat` again.

### Install on a PC without internet

On a PC with internet, run in the app folder
`py -3.12 -m pip download -r requirements.txt -d wheels`, then copy the whole folder (with `wheels\`).
`installer.bat` then installs from `wheels\` without internet.

### Uninstall

Delete the folder. Nothing else was installed (except Python itself).

### Installation problems

| Problem | What to do |
|---|---|
| "No supported Python was found" | Install Python 3.12 with "Add Python to PATH", then run `installer.bat` again |
| A library fails to install | Check the internet connection and run `installer.bat` again; if it fails again, keep the last 20 lines of the black window |
| "The app is not installed yet" | Run `installer.bat` first |
| The page does not open | Port 5000 is used by another program: close it or restart the PC, run `lancer.bat`, open http://localhost:5000 yourself |
| The TapStrap / TapXR is not found | Bluetooth on, device charged and in range, not connected to a phone or another PC |
| A device connects in Windows but not in the app | The slot name must be exactly its Bluetooth name (see step 2 below) |

## How to use it, step by step

### 1. Prepare the gesture set (once per study)

*Gesture sets* page (tab in the examiner).

1. Use the built-in set (34 gestures) or click **+ New set**: name, gestures, repetitions per gesture,
   recording duration per gesture.
2. A set has its own **Latin square** (gesture order balanced across participants) and its **own data
   folder**. Inside a set you can have several **datasets** (for example "principal" and "pilot").
3. Make the set **active**: recording, cleaning, audit and the classifier all use the active set.
4. Put one demonstration video per gesture in `static/videos/` (shown to the participant).

### 2. Connect the sensors

*Examiner* page, card **Tap Sensors**.

1. Each slot (TAP1, TAP2…) has a **type** (TapStrap2 or TapXR) and the **exact Bluetooth name** of the
   device, for example `TapXR_A367310`. A device whose name is in no slot is ignored.
   Names already seen are offered in the list. **+ Add a slot** adds a device.
2. Click **Connect** and switch the devices on. Each slot turns green when connected; the badge
   **LSL ACTIVE** means the LSL streams are published.
3. Check the live rates (*Fingers Hz*, *IMU Hz*). A warning appears if a slot sends no data of its declared
   type after 10 s (wrong type or device not in raw mode).
4. **Reset** disconnects everything. The slots can only be edited after a Reset.

### 3. Record a participant

1. Enter the **participant** number (or `test` for a trial run that goes to `datatest/`) and the
   **session** number, click **Set**. The gesture order of this participant is computed from the
   Latin square and shown in *Gesture order*.
2. Click **Start**. The participant page shows the gesture, its video and description, then a
   3-second countdown; recording starts at "GO" and stops by itself after the gesture duration
   (or with **Stop**).
3. After each repetition, check the counts (*N fingers*, *N IMU*) and the **Plot** tab.
4. If the repetition is bad: **Redo this repetition**. Its files are moved to `anomalie/`
   (nothing is deleted) and the same repetition is recorded again.
   **Skip** goes to the next gesture; **Replay video** shows the video again;
   **Force Repetition** jumps to a given repetition number.
5. The *History* tab lists every repetition with its counts and status; *Log* shows every event.
6. When all gestures are done, the participant page shows "Experiment completed".

If the app or the PC crashes, set the same participant and session again and use **Force Repetition**
to continue where you stopped.

### 4. Check the data

*Data* page.

- **Overview**: one cell per participant × gesture with the number of repetitions found
  (complete, missing, duplicate, moved to anomaly, cleaned). Click a cell to see its files.
  *Only participants with issues* hides the complete ones.
- **Audit** › **Run audit**: checks every file (presence, header, values, gaps, coverage, consistency
  with the counters of the STOP marker). The result lists each problem with its severity; dots appear on
  the overview. **CSV report** exports it.
- Bad repetitions can be **moved to anomaly** and **restored** later (history in
  `data/.../anomalie/moves_log.csv`). Re-run preprocessing after a move or a restore.

### 5. Clean the data (preprocessing)

*Data* page › **Preprocessing**.

1. Choose the participants, the **grid frequency** (Hz) and the interpolation.
2. Optionally give a **clean dataset name** to keep several cleaned versions side by side
   (for example "100Hz" and "200Hz").
3. **Run preprocessing**. For each repetition: sensor/PC clock synchronisation, cut between START and
   STOP, common time grid, all sensors merged in one `_Clean.csv` file.
4. *Synchronization quality* shows, per sensor, the native rate, the clock slope and the residual error.

### 6. Explore

- **Repetition**: pick participant, gesture, repetition, then **View**: raw or cleaned signals,
  zoom between START and STOP.
- **Statistics** / **Raw vs Clean**: tables and heatmaps by gesture, sensor or participant, CSV export.

### 7. Train a classifier

*Classifier* page › **Training**.

1. **Source**: the cleaned dataset to use.
2. **Deep learning** or **Machine learning** (switch above the architectures):
   - Deep learning: **Dense**, **CNN 1D**, **LSTM** (needs TensorFlow). The time series goes in directly.
   - Machine learning: **LogReg**, **SVM**, **Random Forest** (scikit-learn only). Each repetition becomes
     a vector of statistics per column (mean, std, min, max, and the mean of 5 time segments).
   You can tick several models of the same family: they are trained on the same folds and compared.
3. **Validation**:
   - **LOSO** (leave one subject out): each fold tests one participant never seen in training.
     This is the score to report for new users.
   - **User-dependent**: test on one repetition (or session) of each participant; the same people are
     seen in training, so scores are higher.
4. **Data**: sensor points to use, T (each repetition is resampled to T points), participants, gestures.
5. **Hyperparameters**:
   - Deep learning: **Hyperband per fold** (search redone in each fold, no leakage), **Hyperband fold 1 only**
     (faster, slightly optimistic), **Manual** (layers, units, dropout, learning rate…), **Copy from a run**.
   - Machine learning: **Grid search per fold / fold 1 only**: type the values to try for each parameter,
     separated by commas (for example C = `0.1, 1, 10`); every combination is trained on train and scored on
     validation. **Manual** and **Copy from a run** work as for deep learning.
6. **Output**: *Keep fold models* saves the model of every fold; *Train final model* trains one model on all
   participants with the hyperparameters of fold 1 (its score cannot be measured: report the fold scores).
7. **Train**. Progress, fold by fold, is shown live; **Stop** cancels. Results: accuracy, precision,
   recall, F1, AUC per fold, per gesture and per participant, confusion matrix, files in
   `classifier_store/classification/<run>/`.

### 8. Use and compare models

*Classifier* page › **Models** and **History**.

- **Evaluate** a saved model on chosen participants and gestures (it warns when some of them were seen
  in training). **Predict** shows the top-5 gestures for up to 1000 single repetitions.
- **Import a model** trained elsewhere:
  - Keras network (`.keras`, `.h5`): start from **model_template.py** (button *Model script (Keras)*);
  - scikit-learn model (`.joblib`): start from **model_template_ml.py** (button *Model script (ML scikit-learn)*).
    Change only `build_model`, run `python model_template_ml.py "path/to/data_clean"`, import the
    `.joblib` with normalization "none".
  Both templates load and split the data exactly like the page, so scores are comparable.
  A `.joblib` saved by another program (with its own features) cannot be imported as is.
- **History**: every run with its accuracy per model; **⧉** reuses its hyperparameters, **✕** deletes it.
  Evaluations are listed too.

## The pages

| Address | Page | For |
|---|---|---|
| `/examiner` (`/en/examiner`) | Examiner, with tabs Plot, History, Data, Gesture sets, Log | the experimenter |
| `/` (`/en`) | Participant: gesture, video, countdown | the participant's screen |
| `/sets`, `/dataset` | Same examiner page opened on Gesture sets / Data | |
| `/classifier` (`/en/classifier`) | Classifier: Training, Models, History | analysis |
| `/clock` | Large LSL clock | filming it to check video synchronisation |

The language is switched with **EN / FR** at the top of each page.

## Where the files go

```
data/<set>/<dataset>/P07/                 raw recordings: one CSV per stream + markers.csv per repetition
                    P07/videos/            phone videos (if a phone camera is used)
                    anomalie/P07/          repetitions moved by "Redo" or by the Data page (never deleted)
datatest/                                  recordings of participant "test"
data_clean/<set>/<dataset>[/<version>]/Gesture_XX/P07/   cleaned repetitions (*_Clean.csv) + reports
gesture_sets/                              gesture sets and their Latin squares
classifier_store/models/                   saved and imported models (+ one .json description each)
classifier_store/classification/           training runs and evaluations
known_taps.json                            Bluetooth names already seen
```

File names: `P07_S01_Gesture_03_rep02_<date>_<time>_imu_TAP2.csv` = participant 7, session 1,
gesture 3, repetition 2, IMU stream of slot TAP2.
Markers (`COUNTDOWN_START`, `START`, `STOP`, `RETRY`, `SKIP`) carry the LSL time of each event.

## Tips and problems

- **A device does not attach** although Windows connects it: the slot name must be exactly its
  Bluetooth name. Reset, fix the name, Connect again.
- **Start is refused**: preprocessing, audit or training is running; they are blocked during a trial
  (and the reverse) so Bluetooth reception is never slowed down.
- **Back up** `data/` after each session; cleaned data, models and runs can be recomputed from it.
- Running a training needs TensorFlow, keras-tuner and tensorboard for deep learning, only scikit-learn
  for machine learning. The page says what is missing.


---


## Classifier: classic machine learning (scikit-learn)

The Classifier page trains, besides Dense / CNN / LSTM, three classic models: **LogReg**, **SVM** (RBF) and
**Random Forest** (`clf_sklearn.py`). They need scikit-learn only, not TensorFlow.

* Same folds, same per-fold StandardScaler, same result files and reports as the networks.
* Input: each repetition (T × F) becomes one vector: mean, std, min, max of each column, and the mean of
  5 time segments (`clf_sklearn.ml_features`).
* Tuning "per fold" / "fold 1" = grid search on the validation participants (grids in `clf_sklearn.GRIDS`);
  "Manual" = values typed in the page (C, gamma, trees, depth…).
* Saved models: `.joblib`, normalization inside, usable in Evaluate / Predict like the Keras ones.

**Your own ML model:** `model_template_ml.py` (button "Model script (ML scikit-learn)" in Models).
Change `build_model`, run `python model_template_ml.py "path/to/data_clean"`, then import the `.joblib`
with normalization "none". `model_template.py` stays the template for Keras networks.
