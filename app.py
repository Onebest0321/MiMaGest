"""
TapStrap Experiment — web layer (v4)

Same behaviour, same pages and same /api/* answers as app_v3.py, but the data-gathering logic is
split into the five modules of the paper (package `gathering/`):

    gathering/gesture_set.py      1 Gesture Set       what is studied, how many times, in what order
    gathering/session_control.py  2 Session Control   how one trial runs, and for whom
    gathering/devices.py          3 Device            which sensors are present, what they deliver
    gathering/sync.py             4 Synchronisation   one time base (LSL) for streams, markers, camera
    gathering/recording.py        5 Recording         how one repetition is stored

This file only (1) builds the modules and connects them, and (2) translates HTTP requests into
calls to them. The analysis part (dataset_tools, preprocessing, classifier_v2, gestureset_customdata)
is unchanged and is wired exactly as in v3.

Wiring:
    Device ──sample──► Synchronisation.on_sample   (LSL push, continuous)
           └─sample──► Recording.on_sample         (RAM buffer while capturing)
    Synchronisation ──marker──► Recording.on_marker (markers.csv)
    Session Control ──calls──► Gesture Set, Recording, Synchronisation (injected)
"""

from flask import Flask, render_template, jsonify, request, send_file, make_response
import csv, os, threading, time, json, glob
from datetime import datetime
from collections import deque

import dataset_tools as dt   # inventory / check / move of the raw dataset
import preprocessing as pp   # preprocessing chain (clock regression → aligned dataset)
import classifier_v2 as clf  # classifiers (Dense / LSTM / CNN), LORO / LOUO, pre-trained models
import gestureset_customdata as gs

from gathering.gesture_set import GestureSet
from gathering.devices import DeviceManager, KnownTaps, DEVICE_MODELS
from gathering.sync import Timeline
from gathering.recording import Recorder
from gathering.session_control import SessionController

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════
#  EXAMINER LOG  (shown in the examiner page; every module writes to it)
# ═══════════════════════════════════════════════════════════
_log = deque(maxlen=50)
_log_lock = threading.Lock()

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    with _log_lock:
        _log.appendleft(f"[{ts}] {msg}")

# ═══════════════════════════════════════════════════════════
#  MODULES
# ═══════════════════════════════════════════════════════════
# tapsdk is imported at startup, in the main thread, exactly like v3 (the Windows SDK loads its
# .NET library at import time). The modules never import it: it is injected into Device.
from tapsdk import TapSDK, TapInputMode

TIMELINE = Timeline(log=log)                                                   # 4 Synchronisation
GESTURES = GestureSet.create(BASE_DIR)                                         # 1 Gesture Set
SETS = GESTURES.store
RECORDER = Recorder(BASE_DIR, log=log)                                         # 5 Recording
DEVICES = DeviceManager(TapSDK, TapInputMode, TIMELINE.now, log=log,        # 3 Device
                        known_taps=KnownTaps(os.path.join(BASE_DIR, "known_taps.json"), log))

# Device → Sync and Recording; Sync markers → Recording
DEVICES.add_sink(TIMELINE.on_sample)
DEVICES.add_sink(RECORDER.on_sample)
DEVICES.capture_flag = RECORDER.is_capturing
DEVICES.registry_listeners.append(lambda d: RECORDER.set_streams(d.stream_keys, d.slots))
DEVICES.stop_listeners.append(TIMELINE.close_outlets)
RECORDER.set_streams(DEVICES.stream_keys, DEVICES.slots)
TIMELINE.add_marker_sink(RECORDER.on_marker)
if DEVICES.untested:
    print(f"[Registre] ⚠ {DEVICES.untested_note}", flush=True)

# ═══════════════════════════════════════════════════════════
#  ACTIVE GESTURE SET + DISPLAYED DATASET
# ═══════════════════════════════════════════════════════════
ACTIVE_SET = gs.DEFAULT_ID
ACTIVE_DATASET = gs.DEFAULT_DATASET
CLEAN_BASE = os.path.join(BASE_DIR, "data_clean", gs.DEFAULT_ID, gs.DEFAULT_DATASET)   # root of the cleaned dataset
VIEW_SET = VIEW_DS = None     # DISPLAYED dataset (overview, audit, preprocessing, classifier); None = the recording one
VDATA, VGESTURES, VREPS = None, [], 5   # raw data, gestures and repetitions of the displayed dataset (updated by _set_view)
CLEAN_VERSION = ""            # "" = root; otherwise name (slug) of a named version: data_clean/<set>/<slug>
CLEAN_ROOT = CLEAN_BASE       # ACTIVE cleaned version: the rest of the app (stats, repetition, classifier) follows it

def _apply_set(sid, ds=None):
    """Makes `sid` active with its dataset `ds` (Gesture Set module) and points the display to it."""
    global CLEAN_ROOT, CLEAN_BASE, CLEAN_VERSION, ACTIVE_SET, ACTIVE_DATASET
    st = GESTURES.apply(sid, ds)
    CLEAN_BASE = st["clean_root"]
    CLEAN_VERSION, CLEAN_ROOT = "", CLEAN_BASE               # another dataset: back to its default version
    ACTIVE_SET, ACTIVE_DATASET = sid, st["dataset"]
    print(f"[Gesture set] actif : {st['name']} ({sid}) · dataset {st['dataset']} — {len(GESTURES.names)} gestes × {GESTURES.repetitions} rép. — {GESTURES.data_root}")
    _set_view()

def _set_view(sid=None, ds=None):
    """Chooses the dataset DISPLAYED by the Dataset, Preprocessing and Classifier pages. Recording stays on the
    active dataset (GESTURES.data_root)."""
    global VIEW_SET, VIEW_DS, VDATA, VGESTURES, VREPS, CLEAN_BASE, CLEAN_VERSION, CLEAN_ROOT
    if sid is None or (sid == ACTIVE_SET and (ds is None or ds == ACTIVE_DATASET)):
        VIEW_SET = VIEW_DS = None
        VDATA, VGESTURES, VREPS = GESTURES.data_root, list(GESTURES.names), int(GESTURES.repetitions)
        base = SETS.roots(ACTIVE_SET, ACTIVE_DATASET)[1]
    else:
        st = SETS.get(sid, ds)
        VIEW_SET, VIEW_DS = sid, st["dataset"]
        VDATA, base = st["data_root"], st["clean_root"]
        VGESTURES, VREPS = [g["name"] for g in st["gestures"]], int(st["reps"])
    CLEAN_BASE, CLEAN_VERSION, CLEAN_ROOT = base, "", base

try:                                   # old format (data/P01…, data_clean/Gesture_01…) → data/default/principal/…
    for _line in SETS.migrate_legacy():
        print("[Migration]", _line)
except Exception as _e:
    print(f"[Migration] ⚠ échec : {_e} — rien n'a été supprimé, relance après avoir corrigé le problème")

try:
    _apply_set(SETS.active_id(), SETS.active_dataset())
except Exception as _e:
    print(f"[Gesture set] ⚠ jeu actif illisible ({_e}) — retour au jeu d'origine")
    _apply_set(gs.DEFAULT_ID)

# Jobs of the analysis part (defined further down); a trial cannot start while one runs
def _jobs_blocking_trial():
    if preproc["running"] or audit_job["running"]:
        return "Prétraitement ou audit en cours — attendre la fin ou l'annuler"
    if clf.job["running"]:
        return "Un entraînement de classifieur est en cours — attendre la fin ou l'annuler (page Classifieur)"
    return None

SESSION = SessionController(GESTURES, RECORDER, TIMELINE, log=log,              # 2 Session Control
                            blocked=lambda: _jobs_blocking_trial())

# ═══════════════════════════════════════════════════════════
#  FLASK — pages
# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template("index_customdata.html")

@app.route('/examiner')
def examiner():
    # Language choice: ?lang=fr|en (saved in a cookie), otherwise cookie, otherwise French
    lang = request.args.get('lang') or request.cookies.get('lang', 'fr')
    if lang not in ('fr', 'en'):
        lang = 'fr'
    resp = make_response(render_template('examiner_en.html' if lang == 'en' else 'examiner_customdata.html'))
    if request.args.get('lang') in ('fr', 'en'):
        resp.set_cookie('lang', lang, max_age=60*60*24*365)
    return resp

@app.route('/sets')
@app.route('/dataset')
def examiner_views():
    # Same files as /examiner: the page reads its URL and only shows the wanted tab
    return render_template("examiner_customdata.html")

# ───── Versions anglaises (style B · Soft Modern) : /en, /en/examiner, /en/sets, /en/dataset, /en/classifier ─────
@app.route('/en')
def index_en():
    return render_template("index_customdata_en.html")

@app.route('/en/examiner')
@app.route('/en/sets')
@app.route('/en/dataset')
def examiner_en_views():
    # Same page for the three paths, like /examiner, /sets and /dataset
    return render_template("examiner_en.html")

@app.route('/en/classifier')
def classifier_en():
    for p in (os.path.join(BASE_DIR, "templates", "classifier_v2_en.html"),
              os.path.join(BASE_DIR, "classifier_v2_en.html")):
        if os.path.isfile(p):
            return send_file(p, mimetype="text/html")
    return "classifier_v2_en.html introuvable (le mettre dans templates/)", 404

@app.route('/api/state')
def get_state():
    s = SESSION.view()
    fc, ic, slot_counts = RECORDER.counts()
    devices = DEVICES.view()
    with _log_lock:
        recent = list(_log)[:10]
    s.update({
        "untested_configuration": DEVICES.untested,
        "devices":             devices,
        "tap1_connected":      devices.get("TAP1", {}).get("connected", False),   # old fields (v2)
        "tap2_connected":      devices.get("TAP2", {}).get("connected", False),
        "tap_scanning":        DEVICES.scanning,
        "lsl_active":          TIMELINE.active,
        "finger_count":        fc,
        "imu_count":           ic,
        "imu_counts":          {sl: c["imu"] for sl, c in slot_counts.items()},
        "imu_count_tap1":      slot_counts.get("TAP1", {}).get("imu", 0),         # old fields (v2)
        "imu_count_tap2":      slot_counts.get("TAP2", {}).get("imu", 0),
        "log":                 recent,
        "gesture_set":         ACTIVE_SET,
        "dataset":             ACTIVE_DATASET,
        "gesture_labels":      GESTURES.labels(),
    })
    return jsonify(s)

# ═══════════════════════════════════════════════════════════
#  DEVICE REGISTRY  (Device module)
# ═══════════════════════════════════════════════════════════
def _registry_edit_error():
    """Error text if the registry cannot be modified now, otherwise None."""
    err = DEVICES.edit_error()
    if err:
        return err
    if SESSION.busy():
        return "Impossible pendant l'enregistrement"
    return None

def _apply_registry(entries):
    DEVICES.set_registry(entries)          # ValueError if invalid; Recording follows through registry_listeners
    if DEVICES.untested:
        log(f"⚠ {DEVICES.untested_note}")

@app.route('/api/devices', methods=['GET'])
def devices_get():
    return jsonify({"devices": DEVICES.entries(), "models": list(DEVICE_MODELS),
                    "untested_configuration": DEVICES.untested})

@app.route('/api/known_taps', methods=['GET'])
def known_taps_get():
    return jsonify({"taps": DEVICES.known.list()})

@app.route('/api/known_taps/forget', methods=['POST'])
def known_taps_forget():
    DEVICES.known.forget((request.get_json(silent=True) or {}).get("name"))
    return jsonify({"ok": True})

@app.route('/api/devices/add', methods=['POST'])
def devices_add():
    err = _registry_edit_error()
    if err:
        return jsonify({"ok": False, "msg": err})
    body = request.get_json(silent=True) or {}
    try:
        _apply_registry(DEVICES.entries() + [
            {"slot": body.get("slot") or None, "model": body.get("model") or "TapStrap2",
             "position": body.get("position", ""), "name": body.get("name", "")}])
    except ValueError as e:
        return jsonify({"ok": False, "msg": str(e)})
    last = DEVICES.registry[-1]
    log(f"Slot ajouté : {last['slot']} ({last['model']}) — choisis son type et son nom Bluetooth")
    return jsonify({"ok": True})

@app.route('/api/devices/remove', methods=['POST'])
def devices_remove():
    err = _registry_edit_error()
    if err:
        return jsonify({"ok": False, "msg": err})
    slot = (request.get_json(silent=True) or {}).get("slot")
    entries = [e for e in DEVICES.entries() if e["slot"] != slot]
    if len(entries) == len(DEVICES.registry):
        return jsonify({"ok": False, "msg": f"Slot inconnu : {slot}"})
    try:
        _apply_registry(entries)
    except ValueError as e:
        return jsonify({"ok": False, "msg": str(e)})
    log(f"Slot supprimé : {slot}")
    return jsonify({"ok": True})

@app.route('/api/devices/update', methods=['POST'])
def devices_update():
    err = _registry_edit_error()
    if err:
        return jsonify({"ok": False, "msg": err})
    body = request.get_json(silent=True) or {}
    entries, found = DEVICES.entries(), False
    for e in entries:
        if e["slot"] == body.get("slot"):
            found = True
            for k in ("name", "position", "model"):
                if k in body:
                    e[k] = body[k]
    if not found:
        return jsonify({"ok": False, "msg": f"Slot inconnu : {body.get('slot')}"})
    new_slot = str(body.get("new_slot") or "").strip()
    if new_slot and new_slot != body.get("slot"):
        # Changing number: if the target number is already taken, the two slots swap their number.
        old = body.get("slot")
        for e in entries:
            if e["slot"] == new_slot:
                e["slot"] = old
            elif e["slot"] == old:
                e["slot"] = new_slot
        def _num(e):
            import re
            m = re.fullmatch(r"TAP(\d+)", e["slot"])
            return (0, int(m.group(1)), "") if m else (1, 0, e["slot"])
        entries.sort(key=_num)
    try:
        _apply_registry(entries)
    except ValueError as e:
        return jsonify({"ok": False, "msg": str(e)})
    log(f"Slot {body.get('slot')} mis à jour")
    return jsonify({"ok": True})

@app.route('/api/tap_connect', methods=['POST'])
def tap_connect():
    err = DEVICES.begin_scan()
    if err:
        return jsonify({"ok": False, "msg": err})
    TIMELINE.open_outlets(DEVICES.registry)   # LSL first
    DEVICES.launch()
    if DEVICES.untested:
        log(f"⚠ {DEVICES.untested_note}")
    log("🔵 Scan BT démarré — allume les appareils Tap (" + ", ".join(f"{d['slot']}={d['model']}" for d in DEVICES.registry) + ")")
    return jsonify({"ok": True})

@app.route('/api/tap_disconnect', methods=['POST'])
def tap_disconnect():
    DEVICES.reset()
    log("⚫ Connexions réinitialisées")
    return jsonify({"ok": True})

# ═══════════════════════════════════════════════════════════
#  SESSION  (Session Control module)
# ═══════════════════════════════════════════════════════════
def _result(err):
    return jsonify({"ok": False, "msg": err} if err else {"ok": True})

@app.route('/api/set_session', methods=['POST'])
def set_session():
    SESSION.set_session(int(request.get_json().get('session_number', 1)))
    return jsonify({"ok": True})

@app.route('/api/set_repetition', methods=['POST'])
def set_repetition():
    return _result(SESSION.set_repetition(int(request.get_json().get('repetition', 1))))

@app.route('/api/set_participant', methods=['POST'])
def set_participant():
    SESSION.set_participant(request.get_json().get('participant_id', '1'))
    return jsonify({"ok": True})

@app.route('/api/start', methods=['POST'])
def start_recording():
    return _result(SESSION.start())

@app.route('/api/stop', methods=['POST'])
def stop_recording():
    return jsonify(SESSION.stop())

@app.route('/api/retry', methods=['POST'])
def retry_rep():
    return _result(SESSION.retry())

@app.route('/api/skip', methods=['POST'])
def skip_gesture():
    return _result(SESSION.skip())

@app.route('/api/replay_video', methods=['POST'])
def replay_video():
    SESSION.request_replay()
    return jsonify({"ok": True})

@app.route('/api/replay_ack', methods=['POST'])
def replay_ack():
    SESSION.ack_replay()
    return jsonify({"ok": True})

@app.route('/api/set_video', methods=['POST'])
def set_video():
    SESSION.set_video(request.get_json().get('filename', ''))
    return jsonify({"ok": True})

# ═══════════════════════════════════════════════════════════
#  RECORDED FILES  (read-only views of the Recording output)
# ═══════════════════════════════════════════════════════════
def _is_inside(path, roots):
    """True if `path` (after resolving links and `..`) is inside one of the `roots` folders."""
    real = os.path.realpath(path)
    return any(real == r or real.startswith(r + os.sep) for r in (os.path.realpath(x) for x in roots))

def _csv_rows(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= 2000: break
            rows.append(row)
    return jsonify({"file": os.path.basename(path), "rows": rows})

@app.route('/api/plot_csv_by_path')
def plot_csv_by_path():
    path = request.args.get('path', '')
    allowed = [os.path.join(BASE_DIR, d) for d in ("data", "data_clean", "datatest")]
    if not path or not path.lower().endswith('.csv') or not _is_inside(path, allowed) or not os.path.isfile(path):
        return jsonify({"error": "Fichier introuvable ou non autorisé", "rows": []})
    return _csv_rows(path)

@app.route('/api/plot_csv')
def plot_csv():
    key = f"{request.args.get('type','finger')}_{request.args.get('device','TAP1')}"
    path = RECORDER.paths.get(key)
    if not path or not os.path.exists(path):
        return jsonify({"error": f"Pas de fichier pour {key}", "rows": []})
    return _csv_rows(path)

@app.route('/api/last_markers')
def last_markers():
    """Returns the N latest markers from the active log file."""
    n = int(request.args.get('n', 10))
    with RECORDER.marker_lock:
        path = RECORDER.marker_path
    if not path or not os.path.exists(path):
        files = sorted(glob.glob(os.path.join(GESTURES.data_root, '**', '*_markers.csv'), recursive=True), key=os.path.getmtime)
        if not files:
            return jsonify({"markers": []})
        path = files[-1]
    rows = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    rows.append({"ts": row[0], "val": row[1]})
        return jsonify({"markers": rows[-n:]})
    except Exception as e:
        return jsonify({"markers": [], "error": str(e)})

# ═══════════════════════════════════════════════════════════
#  CAMERA  (Synchronisation for the timestamps, Recording for the file)
# ═══════════════════════════════════════════════════════════
def _parse_ts(value):
    return float(str(value).strip().replace(',', '.'))

@app.route('/api/cam_first_frame_ts', methods=['POST'])
def cam_first_frame_ts():
    # lsl_cam_start: System.nanoTime() → same clock as the LSL NTP sync; lsl_cam_start_hw: camera2 sensor timestamp
    ts_sys = request.form.get('lsl_cam_start',    '0.0')
    ts_hw  = request.form.get('lsl_cam_start_hw', '0.0')
    try:
        ts_sys_f, ts_hw_f = _parse_ts(ts_sys), _parse_ts(ts_hw)
        TIMELINE.set_camera_first_frame(ts_sys_f, ts_hw_f)
        log(f"📷 First frame sys={ts_sys_f:.6f}  hw={ts_hw_f:.6f}  lag={(ts_sys_f - ts_hw_f) * 1000:.2f}ms")
        return jsonify({"ok": True})
    except Exception as e:
        log(f"⚠ Erreur parsing timestamp: {e}")
        return jsonify({"ok": False, "msg": f"Timestamp invalide: {ts_sys} / {ts_hw}"})

@app.route('/api/lsl_sync', methods=['GET'])
def lsl_sync():
    return jsonify({"lsl_now": TIMELINE.now()})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    video   = request.files.get('video')
    gesture = request.form.get('gesture', 'unknown')
    rep     = request.form.get('rep', '0')
    if not video:
        return jsonify({"ok": False, "msg": "Pas de fichier"})
    ff_ts, ff_ts_hw = TIMELINE.take_camera_first_frame()
    t2, t3 = SESSION.record_times()
    view = SESSION.view()
    pid, session_num = int(view["participant_id"]), int(view["session_number"])
    # t1: prefer the value received in real time (cam_first_frame_ts), otherwise the one in the upload
    t1_sys = ff_ts    if ff_ts    > 0 else _parse_ts(request.form.get('lsl_cam_start',    '0.0'))
    t1_hw  = ff_ts_hw if ff_ts_hw > 0 else _parse_ts(request.form.get('lsl_cam_start_hw', '0.0'))
    t4     = _parse_ts(request.form.get('lsl_cam_stop', '0.0'))
    log(f"📊 t1_sys={t1_sys:.6f}  t1_hw={t1_hw:.6f}  lag={(t1_sys - t1_hw) * 1000:.2f}ms")
    filename = RECORDER.save_video(video, GESTURES.data_root, pid, session_num, gesture, rep, t1_sys, t1_hw, t2, t3, t4)
    log(f"Vidéo reçue: {filename}")
    return jsonify({"ok": True, "filename": filename})

@app.route('/clock')
def lsl_clock():
    return r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>LSL Clock</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
 
  body {
    background: #0a0a0a;
    color: #e8e8e8;
    font-family: 'Courier New', monospace;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    gap: 32px;
  }
 
  h1 {
    font-size: 13px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #444;
    font-weight: normal;
  }
 
  /* ── Main clock ── */
  #clock {
    font-size: clamp(40px, 8vw, 80px);
    font-weight: bold;
    letter-spacing: 2px;
    color: #4fffb0;
    text-shadow: 0 0 30px #4fffb044;
    transition: color 0.1s;
  }
 
  #clock.flash { color: #fff; text-shadow: 0 0 40px #4fffb0aa; }
 
  /* ── Secondary info ── */
  .info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px 32px;
    font-size: 13px;
    color: #555;
  }
  .info-grid span { color: #888; }
 
  /* ── Markers ── */
  #markers {
    width: min(600px, 90vw);
    max-height: 220px;
    overflow-y: auto;
    border: 1px solid #1e1e1e;
    border-radius: 6px;
    padding: 10px 14px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
 
  .marker-row {
    display: flex;
    gap: 14px;
    font-size: 12px;
    animation: fadein 0.3s ease;
  }
  @keyframes fadein { from { opacity:0; transform:translateY(-4px); } to { opacity:1; } }
 
  .marker-ts  { color: #4fffb0; min-width: 120px; flex-shrink: 0; }
  .marker-val { color: #aaa; word-break: break-all; }
 
  #markers-label {
    font-size: 11px;
    letter-spacing: 3px;
    color: #333;
    text-transform: uppercase;
    align-self: flex-start;
  }
 
  /* ── Button ── */
  button {
    background: none;
    border: 1px solid #333;
    color: #555;
    font-family: inherit;
    font-size: 11px;
    letter-spacing: 2px;
    padding: 8px 20px;
    cursor: pointer;
    text-transform: uppercase;
    transition: all 0.2s;
  }
  button:hover { border-color: #4fffb0; color: #4fffb0; }
</style>
</head>
<body>
 
<h1>LSL local_clock — live</h1>
 
<div id="clock">—</div>
 
<div class="info-grid">
  <div>offset serveur</div> <span id="offset">—</span>
  <div>RTT HTTP</div>       <span id="rtt">—</span>
  <div>drift/s</div>        <span id="drift">—</span>
  <div>syncs</div>          <span id="syncs">0</span>
</div>
 
<div id="markers-label">derniers markers LSL</div>
<div id="markers"><span style="color:#333;font-size:12px">en attente…</span></div>
 
<button onclick="forceSync()">↺ resync</button>
 
<script>
// ── State ──────────────────────────────────────────────────────
let offset      = null   // local_clock_flask - performance.now()/1000
let lastOffset  = null
let syncCount   = 0
let knownMarkers = []
 
// ── Clock sync (simplified NTP) ─────────────────────────────
async function sync() {
  try {
    const t0   = performance.now()
    const resp = await fetch('/api/lsl_sync')
    const t1   = performance.now()
    const data = await resp.json()
 
    const rtt_s  = (t1 - t0) / 1000
    const t_mid  = t0 / 1000 + rtt_s / 2
    const newOff = data.lsl_now - t_mid
 
    lastOffset = offset
    offset     = newOff
    syncCount++
 
    document.getElementById('rtt').textContent    = (rtt_s * 1000).toFixed(1) + ' ms'
    document.getElementById('offset').textContent = newOff.toFixed(6) + ' s'
    document.getElementById('syncs').textContent  = syncCount
 
    if (lastOffset !== null) {
      const drift = (newOff - lastOffset) / 10   // drift per second (sync every 10s)
      document.getElementById('drift').textContent =
        (drift * 1e6).toFixed(1) + ' µs/s'
    }
  } catch(e) {
    document.getElementById('offset').textContent = 'erreur réseau'
  }
}
 
function forceSync() { sync() }
 
// Initial sync + periodic every 10s
sync()
setInterval(sync, 10_000)
 
// ── Clock display (RAF = ~60fps) ─────────────────────────
const clockEl = document.getElementById('clock')
 
function tick() {
  if (offset !== null) {
    const lsl = performance.now() / 1000 + offset
    clockEl.textContent = lsl.toFixed(6)
  }
  requestAnimationFrame(tick)
}
requestAnimationFrame(tick)
 
// ── Poll markers every 500ms ────────────────────────────
async function pollMarkers() {
  try {
    const resp = await fetch('/api/last_markers?n=10')
    const data = await resp.json()
    const markers = data.markers || []
 
    // Only show the new ones
    const newOnes = markers.filter(m => !knownMarkers.includes(m.ts + m.val))
    if (newOnes.length === 0) return
 
    newOnes.forEach(m => knownMarkers.push(m.ts + m.val))
 
    const container = document.getElementById('markers')
    if (container.querySelector('span')) container.innerHTML = ''
 
    newOnes.forEach(m => {
      const row = document.createElement('div')
      row.className = 'marker-row'
      row.innerHTML = `<span class="marker-ts">${parseFloat(m.ts).toFixed(6)}</span>
                       <span class="marker-val">${m.val}</span>`
      container.prepend(row)
    })
 
    // Flash the clock on each new marker
    clockEl.classList.add('flash')
    setTimeout(() => clockEl.classList.remove('flash'), 150)
 
  } catch(e) {}
}
setInterval(pollMarkers, 500)
</script>
</body>
</html>
"""
 
 

# ═══════════════════════════════════════════════════════════
#  PREPROCESSING  (data/PXX → data_clean/Gesture_XX/PXX)
#  Background task: refused during a trial, and vice versa
#  (the Bluetooth thread must not compete with the CPU
#  of the resampling — same principle as the capture architecture).
# ═══════════════════════════════════════════════════════════
# VDATA / CLEAN_ROOT: DISPLAYED dataset (see _set_view), by default the active dataset

preproc = {
    "running": False, "kind": None, "cancel": threading.Event(),
    "done": 0, "total": 0, "current": "", "ok": 0, "skipped": 0, "errors": 0, "warnings": 0,
    "started": None, "finished": None, "message": "", "recent_errors": [], "summary": None, "notices": [],
    
}
preproc_lock = threading.Lock()


def _preproc_progress(p):
    last = p["last"]
    with preproc_lock:
        preproc["done"], preproc["total"], preproc["current"] = p["done"], p["total"], p["current"]
        if last["status"] == "ok":
            preproc["ok"] += 1
            preproc["warnings"] += len(last["warnings"])
        elif last["status"] == "skipped":
            preproc["skipped"] += 1
        else:
            preproc["errors"] += 1
            preproc["recent_errors"] = (preproc["recent_errors"] + [f'{last["key"]} — {last["error"]}'])[-8:]


def _preproc_worker(kind, params):
    try:
        out_root = params["out_root"]
        if kind == "clean":
            diff = pp.stream_differences(pp.stream_sets(VDATA, params["participants"]))
            if diff:                              # simple notification: the repetitions are cleaned anyway
                note = pp.format_stream_notice(diff)
                with preproc_lock:
                    preproc["notices"] = [note]
                log(f"⚠ {len(diff['odd'])} répétition(s) avec des flux différents (voir la page Dataset)")
            results = pp.run_batch(VDATA, out_root, params["participants"], params["freq_hz"],
                                   params["method"], params["overwrite"], params["keep_intermediate"],
                                   fix_slope=bool(params.get("fix_slope")),
                                   progress=_preproc_progress, cancel=preproc["cancel"],
                                   workers=params.get("workers"))
            summary = pp.write_report(results, out_root)
            msg = f'Terminé : {preproc["ok"]} traitée(s), {preproc["skipped"]} ignorée(s), {preproc["errors"]} erreur(s)'
        else:
            summary, msg = None, ""
        with preproc_lock:
            preproc["summary"], preproc["message"] = summary, msg
        log(msg)
    except pp.Cancelled as e:
        try:                                      # the work already done is recorded: report consistent with the files present
            if getattr(e, "results", None):
                pp.write_report(e.results, params["out_root"])
        except Exception as ex_:
            log(f"Rapport partiel non écrit : {ex_}")
        with preproc_lock:
            preproc["message"] = "Annulé"
        log("Prétraitement annulé")
    except Exception as e:
        with preproc_lock:
            preproc["message"] = f"Erreur : {e}"
            preproc["errors"] += 1
        log(f"Prétraitement : erreur {e}")
    finally:
        with preproc_lock:
            preproc["running"] = False
            preproc["finished"] = time.time()


def _start_preproc(kind, params):
    busy = SESSION.busy()
    if audit_job["running"]:
        return jsonify({"ok": False, "msg": "Un audit d'intégrité est en cours"})
    if clf.job["running"]:
        return jsonify({"ok": False, "msg": "Un entraînement de classifieur est en cours"})
    if busy:
        return jsonify({"ok": False, "msg": "Un essai est en cours — impossible de prétraiter maintenant"})
    with preproc_lock:
        if preproc["running"]:
            return jsonify({"ok": False, "msg": "Un prétraitement est déjà en cours"})
        preproc.update(running=True, kind=kind, done=0, total=0, current="", ok=0, skipped=0,
                       errors=0, warnings=0, started=time.time(), finished=None, message="",
                       recent_errors=[], summary=None, notices=[])
        preproc["cancel"].clear()
    if kind == "clean":
        _select_clean_version(params["version_slug"])        # the app follows the version being produced
    threading.Thread(target=_preproc_worker, args=(kind, params), daemon=True).start()
    return jsonify({"ok": True})


# ── Named versions of the cleaned dataset ──────────────────────────────────────
# A same raw dataset can have several clean versions (frequency, interpolation…), each in
# <data_clean/set>/<name>/ (top-level folder, version.json states its dataset) with its own statistics and models. Without a name, we write
# in the root of the cleaned dataset (original behaviour).
def _version_root(slug):
    return dt.version_root(CLEAN_BASE, slug)

def _select_clean_version(slug):
    global CLEAN_VERSION, CLEAN_ROOT
    CLEAN_VERSION, CLEAN_ROOT = slug, _version_root(slug)
    try:
        clf.reset_selected()              # the classifier follows the active version
    except Exception:
        pass

def _count_clean(root):
    n = 0
    for g in glob.glob(os.path.join(root, "Gesture_*")):
        n += len(glob.glob(os.path.join(g, "P*", "*_Clean.csv")))
    return n

def _list_clean_versions():
    out = [{"slug": "", "name": "Par défaut", "files": _count_clean(CLEAN_BASE)}]
    if True:
        for n, root in dt.list_versions(CLEAN_BASE):
            try:
                with open(os.path.join(root, "version.json"), encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
            out.append({"slug": n, "name": meta.get("name") or n, "files": _count_clean(root),
                        "freq_hz": meta.get("freq_hz"), "method": meta.get("method"), "clock_corrected": bool(meta.get("clock_corrected")), "fix_slope": bool(meta.get("fix_slope")), "updated": meta.get("updated")})
    return out

@app.route('/api/preprocess/versions')
def preprocess_versions():
    return jsonify({"active": CLEAN_VERSION, "versions": _list_clean_versions()})

@app.route('/api/preprocess/version/select', methods=['POST'])
def preprocess_version_select():
    slug = str((request.get_json(silent=True) or {}).get("slug") or "")
    busy = SESSION.busy()
    if busy or preproc["running"] or audit_job["running"] or clf.job["running"]:
        return jsonify({"ok": False, "msg": "Un essai, un prétraitement, un audit ou un entraînement est en cours"})
    if slug and slug not in dict(dt.list_versions(CLEAN_BASE)):
        return jsonify({"ok": False, "msg": "Version inconnue"})
    _select_clean_version(slug)
    log("Dataset nettoyé actif : " + (slug or "par défaut"))
    return jsonify({"ok": True})

@app.route('/api/preprocess/participants')
def preprocess_participants():
    parts = pp.list_participants(VDATA)
    vname = (request.args.get("version") or "").strip()      # name being typed (not yet created) → counters of this version
    root = _version_root(gs.slug(vname)) if vname else CLEAN_ROOT
    return jsonify({"participants": pp.count_pending(VDATA, root, parts),
                    "defaults": {"freq_hz": pp.DEFAULT_FREQ_HZ, "method": pp.DEFAULT_METHOD,
                                 "workers": pp.default_workers(), "cpus": os.cpu_count() or 1,
                                 "methods": list(pp.METHODS)}})


@app.route('/api/preprocess/start', methods=['POST'])
def preprocess_start():
    d = request.get_json(silent=True) or {}
    available = pp.list_participants(VDATA)
    wanted = d.get("participants") or available
    parts = [p for p in wanted if p in available]
    if not parts:
        return jsonify({"ok": False, "msg": "Aucun participant à traiter"})
    method = d.get("method", pp.DEFAULT_METHOD)
    if method not in pp.METHODS:
        return jsonify({"ok": False, "msg": f"Méthode inconnue : {method}"})
    try:
        freq = int(d.get("freq_hz", pp.DEFAULT_FREQ_HZ))
        assert 20 <= freq <= 1000
    except Exception:
        return jsonify({"ok": False, "msg": "Fréquence invalide (20–1000 Hz)"})
    vname = str(d.get("version") or "").strip()[:60]
    vslug = gs.slug(vname) if vname else CLEAN_VERSION          # empty name: the active version
    out_root = _version_root(vslug)
    if vslug and vslug in SETS.datasets(os.path.basename(os.path.dirname(CLEAN_BASE))):
        return jsonify({"ok": False, "msg": f"« {vslug} » est déjà le nom d'un dataset : choisis un autre nom de version"})
    if vslug and dt.owner_of(out_root) not in (None, os.path.normpath(CLEAN_BASE)) \
            and os.path.normcase(dt.owner_of(out_root)) != os.path.normcase(os.path.normpath(CLEAN_BASE)):
        return jsonify({"ok": False, "msg": f"« {vslug} » est déjà une version d'un autre dataset : choisis un autre nom"})
    if vslug and os.path.isdir(out_root) and not os.path.isfile(os.path.join(out_root, "version.json")) and os.listdir(out_root):
        return jsonify({"ok": False, "msg": f"Le dossier {out_root} existe déjà et n'est pas une version : choisis un autre nom"})
    fix_slope = bool(d.get("fix_slope"))
    if fix_slope and not vslug:
        return jsonify({"ok": False, "msg": "Pente a = 1 fixe : donne un nom de version (le dataset par défaut garde la pente ajustée)"})
    meta = {}
    if vslug:
        try:
            with open(os.path.join(out_root, "version.json"), encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
        if _count_clean(out_root) > 0 and not meta.get("clock_corrected") and not d.get("overwrite"):
            # files produced before the corrected clock became the only mode (arrival time): do not mix them
            return jsonify({"ok": False, "msg": "Cette version contient des fichiers produits avec l'ancien temps d'arrivée (sans horloge corrigée) : "
                                                "choisis un autre nom, ou coche « Retraiter les répétitions déjà nettoyées »"})
        if _count_clean(out_root) > 0 and bool(meta.get("fix_slope")) != fix_slope and not d.get("overwrite"):
            return jsonify({"ok": False, "msg": "Cette version contient déjà des fichiers avec l'autre réglage de pente (ajustée / a = 1) : "
                                                "choisis un autre nom, ou coche « Retraiter les répétitions déjà nettoyées »"})
    if vslug:
        os.makedirs(out_root, exist_ok=True)
        mpath = os.path.join(out_root, "version.json")
        if not meta:
            meta = {"created": datetime.now().isoformat(timespec="seconds")}
        meta.update(dataset=os.path.basename(os.path.normpath(CLEAN_BASE)), name=vname or meta.get("name") or vslug, freq_hz=freq, method=method, clock_corrected=True, fix_slope=fix_slope,
                    updated=datetime.now().isoformat(timespec="seconds"))
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return _start_preproc("clean", {"participants": parts, "freq_hz": freq, "method": method,
                                    "version_slug": vslug, "out_root": out_root,
                                    "overwrite": bool(d.get("overwrite")),
                                    "keep_intermediate": bool(d.get("keep_intermediate")),
                                    "fix_slope": fix_slope,
                                    "workers": max(1, min(int(d.get("workers") or pp.default_workers()), 16))})


@app.route('/api/preprocess/cancel', methods=['POST'])
def preprocess_cancel():
    preproc["cancel"].set()
    return jsonify({"ok": True})


@app.route('/api/preprocess/status')
def preprocess_status():
    with preproc_lock:
        s = {k: v for k, v in preproc.items() if k != "cancel"}
    s["elapsed"] = round((s["finished"] or time.time()) - s["started"], 1) if s["started"] else 0
    if s["summary"] is None:
        path = os.path.join(CLEAN_ROOT, "preprocessing_summary.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    s["summary"] = json.load(f)
            except Exception:
                pass
    return jsonify(s)


@app.route('/api/preprocess/report.csv')
def preprocess_report():
    path = os.path.join(CLEAN_ROOT, "preprocessing_report.csv")
    if not os.path.isfile(path):
        return jsonify({"ok": False, "msg": "Pas encore de rapport"}), 404
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name="preprocessing_report.csv")



# ═══════════════════════════════════════════════════════════
#  DATASET: inventory, 5-repetition check, move, preview
# ═══════════════════════════════════════════════════════════
audit_job = {"running": False, "cancel": threading.Event(), "done": 0, "total": 0, "current": "",
             "message": "", "started": None, "finished": None}
audit_lock = threading.Lock()


def _dataset_locked():
    rec = SESSION.busy()
    if rec:
        return "Un essai est en cours — réessayer après l'enregistrement"
    if preproc["running"]:
        return "Un prétraitement est en cours"
    if audit_job["running"]:
        return "Un audit d'intégrité est en cours"
    if clf.job["running"]:
        return "Un entraînement de classifieur est en cours"
    return None


def _audit_worker():
    try:
        def prog(p):
            with audit_lock:
                audit_job.update(done=p["done"], total=p["total"], current=p["current"])
        s = dt.audit_all(VDATA, CLEAN_ROOT, progress=prog, cancel=audit_job["cancel"])
        msg = f'Audit terminé : {s["audited"]} répétitions, {s["with_errors"]} avec erreur(s), {s["with_warnings_only"]} avec avertissement(s)'
    except dt.AuditCancelled:
        msg = "Audit annulé"
    except Exception as e:
        msg = f"Erreur d'audit : {e}"
    with audit_lock:
        audit_job.update(running=False, message=msg, finished=time.time())
    log(msg)


@app.route('/api/dataset/audit/start', methods=['POST'])
def audit_start():
    lock = _dataset_locked()
    if lock:
        return jsonify({"ok": False, "msg": lock})
    with audit_lock:
        audit_job.update(running=True, done=0, total=0, current="", message="", started=time.time(), finished=None)
        audit_job["cancel"].clear()
    threading.Thread(target=_audit_worker, daemon=True).start()
    return jsonify({"ok": True})


@app.route('/api/dataset/audit/cancel', methods=['POST'])
def audit_cancel():
    audit_job["cancel"].set()
    return jsonify({"ok": True})


@app.route('/api/dataset/audit/status')
def audit_status():
    with audit_lock:
        j = {k: v for k, v in audit_job.items() if k != "cancel"}
    j["summary"] = dt.load_audit_summary(CLEAN_ROOT)
    return jsonify(j)


@app.route('/api/dataset/audit/issues')
def audit_issues_route():
    limit = min(int(request.args.get('limit', 300) or 300), 5000)
    return jsonify({"items": dt.audit_issues(CLEAN_ROOT, request.args.get('cls', ''), request.args.get('code', ''),
                                             limit, request.args.get('severity', ''))})


@app.route('/api/dataset/audit/report.csv')
def audit_report():
    path = os.path.join(CLEAN_BASE, "integrity_issues.csv")
    if not os.path.isfile(path):
        return jsonify({"ok": False, "msg": "Pas encore d'audit"}), 404
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name="integrity_issues.csv")


@app.route('/api/dataset/overview')
def dataset_overview():
    return jsonify(dt.scan_overview(VDATA, CLEAN_ROOT, VGESTURES, VREPS))


@app.route('/api/dataset/cell')
def dataset_cell():
    try:
        return jsonify(dt.cell_detail(VDATA, CLEAN_ROOT, request.args.get('participant', ''),
                                      request.args.get('gesture', ''), VREPS))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/dataset/move', methods=['POST'])
def dataset_move():
    lock = _dataset_locked()
    if lock:
        return jsonify({"ok": False, "msg": lock})
    d = request.get_json(silent=True) or {}
    try:
        res = dt.move_repetitions(VDATA, CLEAN_ROOT, d.get("participant", ""),
                                  d.get("keys") or [], str(d.get("reason", ""))[:200])
    except ValueError as e:
        return jsonify({"ok": False, "msg": str(e)})
    except Exception as e:                                   # never an HTML 500: the page must be able to read the answer
        return jsonify({"ok": False, "msg": f"{type(e).__name__}: {str(e)[:200]}"})
    log(f"Dataset : {len(res['moved'])} répétition(s) déplacée(s) vers anomalie ({d.get('participant')})"
        + (f", {len(res['failed'])} en échec : {res['failed'][0]['error']}" if res['failed'] else ""))
    return jsonify({"ok": True, **res})


@app.route('/api/dataset/anomalies')
def dataset_anomalies():
    return jsonify({"items": dt.list_anomalies(VDATA, request.args.get('participant') or None)})


@app.route('/api/dataset/restore', methods=['POST'])
def dataset_restore():
    lock = _dataset_locked()
    if lock:
        return jsonify({"ok": False, "msg": lock})
    d = request.get_json(silent=True) or {}
    try:
        return jsonify({"ok": True, **dt.restore_repetition(VDATA, d.get("participant", ""), d.get("key", ""))})
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route('/api/dataset/stats')
def dataset_stats():
    return jsonify(dt.stats_rows(CLEAN_ROOT))


@app.route('/api/dataset/rep')
def dataset_rep():
    try:
        return jsonify(dt.rep_data(VDATA, CLEAN_ROOT, request.args.get('participant', ''),
                                   request.args.get('key', ''), request.args.get('source', 'raw')))
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 404


# ═══════════════════════════════════════════════════════════
#  CLASSIFIEUR  (page /classifier + /api/clf/*)
#  Same rule as preprocessing: never at the same time as a trial.
# ═══════════════════════════════════════════════════════════
def _clf_busy():
    rec = SESSION.busy()
    if rec:
        return "Un essai est en cours — impossible d'entraîner ou d'inférer maintenant"
    if preproc["running"]:
        return "Un prétraitement est en cours"
    if audit_job["running"]:
        return "Un audit d'intégrité est en cours"
    return None


clf.register(app, lambda: CLEAN_ROOT, BASE_DIR, busy=_clf_busy)


# ── Displayed dataset (independent of the recording one) ──────────────
@app.route('/api/view')
def view_get():
    sets = []
    for s_ in SETS.summary(ACTIVE_SET, ACTIVE_DATASET):
        if s_.get("error"):
            continue
        sets.append({"id": s_["id"], "name": s_["name"],
                     "datasets": [{"id": d["id"], "name": d["name"], "active": bool(d.get("active"))} for d in s_["datasets"]]})
    return jsonify({"sets": sets, "active": {"set": ACTIVE_SET, "dataset": ACTIVE_DATASET},
                    "view": {"set": VIEW_SET or ACTIVE_SET, "dataset": VIEW_DS or ACTIVE_DATASET, "following": VIEW_SET is None}})

@app.route('/api/view/select', methods=['POST'])
def view_select():
    d = request.get_json(silent=True) or {}
    if preproc["running"] or audit_job["running"] or clf.job["running"]:
        return jsonify({"ok": False, "msg": "Un prétraitement, un audit ou un entraînement est en cours : attends la fin ou annule"})
    try:
        _set_view(d.get("set") or None, d.get("dataset") or None)
    except ValueError as e:
        return jsonify({"ok": False, "msg": str(e)})
    try:
        clf.reset_selected()
    except Exception:
        pass
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════
#  JEUX DE GESTES  (onglet « Jeux de gestes » de l'examinateur)
# ═══════════════════════════════════════════════════════════
def _sets_busy():
    if SESSION.busy():
            return "Un essai est en cours — impossible de changer de jeu de gestes maintenant"
    if clf.job.get("running"):
        return "Un entraînement est en cours"
    return _clf_busy()

def _on_set_activated(sid, ds=None):
    _apply_set(sid, ds)
    try:
        clf.reset_selected()              # the classifier follows the active dataset again
    except Exception:
        pass
    SESSION.reset()                       # the current participant no longer makes sense in another set / dataset
    log(f"Jeu de gestes actif : {SETS.get(sid, ds)['name']} · dataset {ACTIVE_DATASET}")

gs.register(app, SETS, _sets_busy, lambda: ACTIVE_SET, _on_set_activated, os.path.join(BASE_DIR, "static", "videos"),
            lambda: ACTIVE_DATASET)


if __name__ == '__main__':
    print("=" * 52)
    print("  TapStrap Experiment — Flask + LSL (v4)")
    print("  Participant : http://localhost:5000")
    print("  Examinateur : http://localhost:5000/examiner")
    print("  Classifieur : http://localhost:5000/classifier")
    print("  English     : http://localhost:5000/en  (/en/examiner, /en/classifier)")
    print("=" * 52)
    print("\n  LSL streams disponibles dans LabRecorder dès")
    print("  que tu cliques 'Connecter' côté examinateur.\n")
    print("  Streams : " + ", ".join(f"TapStrap_{'Finger' if k.startswith('finger_') else 'IMU'}_{k.split('_', 1)[1]}" for k in DEVICES.stream_keys))
    print("            TapExp_Markers\n")
    print("  Prétraitement : onglet « Dataset » de l'examinateur\n")
    
   

    app.run(debug=False,host='0.0.0.0', port=5000, threaded=True)
