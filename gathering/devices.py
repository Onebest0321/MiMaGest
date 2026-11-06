"""MODULE 3 - Device.

Question: which sensors are present, and what do they deliver?

Owns: the device registry (slots, models, Bluetooth names), the known-taps memory, the Bluetooth
connection through tapsdk, the raw mode, the "model matches the packets" check, the status of
each device (connected / name / warning) and the conversion of tapsdk packets into samples.

It does NOT know who uses the samples. Every sample is handed to the registered sinks:

    sink(key, lsl_ts, hw_ts, data_type, sample, capturing)

    key        "finger_TAP1", "imu_TAP2", ...
    lsl_ts     time of reception on the common clock (taken in the Bluetooth thread)
    hw_ts      device clock (s)
    capturing  value of the `capture_flag()` callback AT RECEPTION TIME (as FAST_RECORDING in v3)

Everything external is injected (SDK class, input-mode class, clock, log), so the module can be
tested with a fake SDK and no Bluetooth.
"""
import json
import queue
import re
import threading
import time

# Streams produced by each device model in raw mode.
DEVICE_MODELS = {
    "TapStrap2": ("finger", "imu"),   # 5 finger accelerometers (15 channels) + thumb IMU (6 channels)
    "TapXR":     ("imu",),            # wristband IMU only (6 channels)
}

DEFAULT_DEVICE_REGISTRY = [
    {"slot": "TAP1", "model": "TapStrap2", "position": "right hand",    "name": ""},   # Bluetooth name to enter
    {"slot": "TAP2", "model": "TapXR",     "position": "right forearm", "name": ""},   # Bluetooth name to enter
]

# Only validated configuration: one TapStrap2 and one TapXR.
TESTED_CONFIGURATION = {"TapStrap2": 1, "TapXR": 1}
MODEL_CHECK_DELAY = 10.0   # seconds after connection before checking that the data match the declared model
RAW_MODE_SENSITIVITY = [0, 5, 4]


def normalize_registry(raw):
    """Validates a list of entries {slot?, model, position?, name?} and returns the full registry.
    A missing slot gets the first free name TAP1, TAP2… Bluetooth names are compared case-insensitively."""
    explicit = {str(d["slot"]) for d in raw if d.get("slot")}
    registry, used, names, nxt = [], set(), set(), 1
    for d in raw:
        model = d.get("model")
        if model not in DEVICE_MODELS:
            raise ValueError(f"Modèle inconnu « {model} » (connus : {', '.join(DEVICE_MODELS)})")
        slot = str(d["slot"]) if d.get("slot") else None
        if slot is None:
            while f"TAP{nxt}" in explicit or f"TAP{nxt}" in used:
                nxt += 1
            slot = f"TAP{nxt}"
        if not re.fullmatch(r"[A-Za-z0-9]+", slot):
            raise ValueError(f"Nom de slot invalide « {slot} » (lettres et chiffres seulement)")
        if slot in used:
            raise ValueError(f"Slot « {slot} » défini deux fois")
        name = str(d.get("name") or "").strip()
        if name and name.lower() in names:
            raise ValueError(f"Le nom Bluetooth « {name} » est déjà utilisé par un autre slot")
        used.add(slot)
        if name:
            names.add(name.lower())
        registry.append({"slot": slot, "model": model, "position": str(d.get("position") or ""),
                         "name": name, "streams": DEVICE_MODELS[model]})
    if not registry:
        raise ValueError("Il faut au moins un appareil")
    return registry


def stream_keys(registry):
    return [f"{st}_{d['slot']}" for d in registry for st in d["streams"]]


def packet_to_sample(data_type, raw_points):
    """tapsdk packet → (stream, sample). Finger: 15 values; IMU: 6 values, reordered Accel then Gyro.
    Returns (None, None) for an unknown packet type."""
    sample = []
    for p in raw_points:
        sample += [float(p.x), float(p.y), float(p.z)] if hasattr(p, 'x') else [0., 0., 0.]
    if 'Device' in data_type:
        while len(sample) < 15:
            sample.append(0.)
        return "finger", sample
    if 'IMU' in data_type:
        while len(sample) < 6:
            sample.append(0.)
        # Accel / Gyro swapped because the raw data is inverted compared to the header
        return "imu", sample[3:6] + sample[0:3]
    return None, None


class KnownTaps:
    """Taps already connected (permanent memory, independent of the session registry)."""

    def __init__(self, path, log=lambda m: None):
        self.path = path
        self.lock = threading.Lock()
        self.log = log

    def list(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return [{"name": str(d["name"]), "model": str(d.get("model") or "")} for d in data if d.get("name")]
        except Exception:
            return []

    def remember(self, name, model):
        if not name:
            return
        with self.lock:
            known = [d for d in self.list() if d["name"] != name]
            known.append({"name": name, "model": model})
            known.sort(key=lambda d: d["name"].lower())
            try:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(known, f, ensure_ascii=False, indent=2)
            except OSError as e:
                self.log(f"⚠ Impossible d'enregistrer known_taps.json : {e}")

    def forget(self, name):
        with self.lock:
            known = [d for d in self.list() if d["name"] != name]
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(known, f, ensure_ascii=False, indent=2)


def _empty_status():
    return {"connected": False, "id": None, "name": None, "warning": None}


class DeviceManager:
    def __init__(self, sdk_class, input_mode_class, clock, log=lambda m: None, known_taps=None,
                 timer=threading.Timer, model_check_delay=MODEL_CHECK_DELAY, retry_delay=3.0):
        self.sdk_class = sdk_class
        self.input_mode_class = input_mode_class
        self.clock = clock
        self.log = log
        self.known = known_taps
        self.timer = timer
        self.model_check_delay = model_check_delay
        self.retry_delay = retry_delay
        self.lock = threading.RLock()
        self.registry = []
        self.slots, self.slot_info, self.stream_keys = [], {}, []
        self.untested, self.untested_note = False, ""
        self.status = {}
        self.seen = {}
        self.slot_map = {}            # tapsdk identifier → slot (read without lock by the Bluetooth thread)
        self.sinks = []
        self.registry_listeners = []  # called after every registry change (e.g. Recording rebuilds its buffers)
        self.stop_listeners = []      # called when Bluetooth stops (e.g. Synchronisation closes its outlets)
        self.capture_flag = lambda: False
        self.scanning = False
        self.tap = None
        self.thread = None
        self.queue = queue.Queue()
        self.queue_active = False
        self.connected_ids = {}    # tapsdk identifier → Bluetooth name, for devices connected at SDK level
        self.set_registry([dict(d) for d in DEFAULT_DEVICE_REGISTRY])

    # ── registry ────────────────────────────────────────────
    def set_registry(self, entries):
        """Validates and replaces the registry. Raises ValueError (registry unchanged) if invalid."""
        registry = normalize_registry(entries)
        with self.lock:
            self.registry[:] = registry
            self.slots[:] = [d["slot"] for d in registry]
            self.slot_info.clear()
            self.slot_info.update({d["slot"]: d for d in registry})
            self.stream_keys[:] = stream_keys(registry)
            counts = {}
            for d in registry:
                counts[d["model"]] = counts.get(d["model"], 0) + 1
            self.untested = counts != TESTED_CONFIGURATION
            self.untested_note = ("configuration d'appareils NON TESTÉE : seule 1 TapStrap2 + 1 TapXR a été validée "
                                  f"(registre actuel : {', '.join(f'{n} {m}' for m, n in counts.items())}). "
                                  "Le prétraitement, l'audit et le classifieur ne sont pas adaptés à d'autres configurations.")
            old = self.status
            self.status = {s: old.get(s, _empty_status()) for s in self.slots}
            self.seen = {s: {"finger": False, "imu": False} for s in self.slots}
        for fn in self.registry_listeners:
            fn(self)

    def entries(self):
        return [{k: d[k] for k in ("slot", "model", "position", "name")} for d in self.registry]

    def missing_names(self):
        return [d["slot"] for d in self.registry if not d["name"]]

    def edit_error(self):
        if self.scanning:
            return "Arrête le scan (Reset) avant de modifier les appareils"
        return None

    # ── sinks ───────────────────────────────────────────────
    def add_sink(self, fn):
        self.sinks.append(fn)

    # ── status ──────────────────────────────────────────────
    def view(self):
        with self.lock:
            return {sl: dict(dv, model=self.slot_info[sl]["model"], position=self.slot_info[sl]["position"],
                             streams=list(self.slot_info[sl]["streams"]), pinned_name=self.slot_info[sl]["name"])
                    for sl, dv in self.status.items()}

    def is_connected(self, slot):
        return self.status.get(slot, {}).get("connected", False)

    def _reset_status(self):
        for dev in self.status.values():
            dev.update(_empty_status())

    def _warn(self, slot, msg):
        with self.lock:
            self.status[slot]["warning"] = msg
        self.log(f"⚠ {slot} : {msg}")

    # ── tapsdk callbacks ────────────────────────────────────
    def _slot_for(self, identifier, name):
        for slot, dev in self.status.items():
            if dev["id"] == identifier:
                return slot
        nm = (name or "").strip().lower()
        for d in self.registry:
            if d["name"] and d["name"].lower() == nm:
                return d["slot"]
        return None

    def on_connect(self, identifier, name, fw):
        with self.lock:
            slot = self._slot_for(identifier, name)
            if slot is None:
                self.log(f"Appareil ignoré : {name} — aucun slot ne porte ce nom Bluetooth "
                         f"(ajoute-le dans l'examinateur, avec son type, avant de scanner)")
                return
            dev = self.status[slot]
            if dev["id"] is not None and dev["id"] != identifier:
                self.slot_map.pop(dev["id"], None)      # the previous device of this slot is replaced
            dev.update(connected=True, id=identifier, name=name, warning=None)
            self.seen[slot] = {"finger": False, "imu": False}
            info = self.slot_info[slot]
            label = f"{info['model']} {slot} ({info['position']})"
        if self.known:
            self.known.remember(name, info["model"])
        self.slot_map[identifier] = slot
        # Small delay to solidify the Bluetooth connection before sending the raw mode command
        self.timer(1.0, lambda: self._set_mode(identifier, label, name)).start()
        self.timer(self.model_check_delay, lambda: self._check_model(slot, identifier)).start()

    def _set_mode(self, identifier, label, name):
        try:
            if self.tap:
                self.tap.set_input_mode(self.input_mode_class("raw", list(RAW_MODE_SENSITIVITY)), identifier)
        except Exception as e:
            self.log(f"⚠ Erreur set mode sur {label}: {e}")
        self.log(f"✓ {label} connecté : {name}")

    def _check_model(self, slot, identifier):
        with self.lock:
            dev = self.status.get(slot)
            if not dev or not dev["connected"] or dev["id"] != identifier:
                return
        expected = self.slot_info[slot]["streams"]
        missing = [st for st in expected if not self.seen[slot][st]]
        if missing:
            self._warn(slot, f"aucun paquet « {', '.join(missing)} » reçu en {self.model_check_delay:.0f} s alors que le slot est déclaré "
                             f"{self.slot_info[slot]['model']} — vérifie le modèle dans le registre, ou si l'appareil est bien en mode raw")

    def on_disconnect(self, identifier):
        with self.lock:
            for slot, dev in self.status.items():
                if dev["id"] == identifier:
                    dev.update(_empty_status())
                    self.log(f"⚠ {self.slot_info[slot]['model']} {slot} déconnecté")
                    break
        self.slot_map.pop(identifier, None)

    def on_raw_sensor_data(self, identifier, raw_sensor_data):
        """Called by the Bluetooth thread: takes the timestamps at once, blocks nothing, queues the packet."""
        lsl_ts = self.clock()
        hardware_ts = getattr(raw_sensor_data, 'timestamp', 0) / 1000.0
        if not raw_sensor_data.points:
            return
        slot = self.slot_map.get(identifier)
        if not slot:
            return
        self.queue.put((lsl_ts, hardware_ts, slot, str(raw_sensor_data.type), raw_sensor_data.points,
                        self.capture_flag()))

    # ── processing thread ───────────────────────────────────
    def process_item(self, item):
        """Turns one queued packet into a sample and hands it to every sink."""
        lsl_ts, hardware_ts, slot, data_type, raw_points, capturing = item
        stream, sample = packet_to_sample(data_type, raw_points)
        if stream is None:
            return
        seen = self.seen.get(slot)
        if seen is not None:
            if stream == "finger" and not seen["finger"]:
                seen["finger"] = True
                if "finger" not in self.slot_info[slot]["streams"]:
                    self._warn(slot, f"paquets doigts reçus alors que le slot {slot} est déclaré {self.slot_info[slot]['model']} (sans flux doigts) : "
                                     f"ces données sont ignorées — corrige le modèle dans le registre")
            elif stream == "imu":
                seen["imu"] = True
        key = f"{stream}_{slot}"
        for sink in self.sinks:
            sink(key, lsl_ts, hardware_ts, data_type, sample, capturing)

    def _process_queue(self):
        while self.queue_active:
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.process_item(item)
            finally:
                self.queue.task_done()

    def drain(self):
        """Processes everything already queued (used by tests and before stopping)."""
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                return
            try:
                self.process_item(item)
            finally:
                self.queue.task_done()

    # ── scan control ────────────────────────────────────────
    def begin_scan(self):
        """Checks a scan can start and marks it as started. Returns an error message or None."""
        with self.lock:
            if self.scanning:
                return "Déjà en écoute"
            missing = self.missing_names()
            if missing:
                return f"Saisis le nom Bluetooth des slots : {', '.join(missing)}"
            self.scanning = True
        return None

    # The Windows Tap SDK keeps ONE native manager per process. Creating a second SDK object after
    # a Reset (as v3 did) registers the callbacks a second time, so every packet arrives twice and
    # the counts double. v4 therefore creates the SDK object once, runs it once, and keeps it for
    # the whole life of the app. Reset only stops *using* its events; a new scan uses them again.

    def _on_connect_evt(self, identifier, name, fw):
        self.connected_ids[identifier] = name          # remembered even when not scanning
        self.log(f"SDK : connexion reçue de {name}" + ("" if self.scanning else " (scan arrêté, ignorée)"))
        if self.scanning:
            self.on_connect(identifier, name, fw)

    def _on_disconnect_evt(self, identifier):
        self.connected_ids.pop(identifier, None)
        if self.scanning:
            self.on_disconnect(identifier)

    def _on_raw_evt(self, identifier, data):
        if self.scanning:
            self.on_raw_sensor_data(identifier, data)

    def _init_tap(self):
        if not self.queue_active:
            self.queue_active = True
            threading.Thread(target=self._process_queue, daemon=True).start()
        tap = self.sdk_class()
        tap.register_connection_events(self._on_connect_evt)
        tap.register_disconnection_events(self._on_disconnect_evt)
        tap.register_raw_data_events(self._on_raw_evt)
        self.tap = tap

    def launch(self):
        """Starts listening (after begin_scan). As in v3, the SDK object is created here, in the calling
        (Flask) thread, before the background thread runs it. It is created and run only the first time;
        after a Reset, the devices that are still connected are attached again to their slots."""
        if self.tap is None:
            self._init_tap()
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        else:
            for identifier, name in list(self.connected_ids.items()):
                self.on_connect(identifier, name, None)

    def _run(self):
        """Runs the SDK once and keeps it alive for the whole life of the app (retries on error)."""
        retry_count = 0
        while True:
            try:
                if self.tap is None:
                    self._init_tap()
                self.tap.run()
                while True:          # keeps the thread (and the SDK object) alive
                    time.sleep(1)
            except Exception as e:
                self.log(f"⚠ SDK erreur : {e}")
                retry_count += 1
                self.log(f"🔄 Tentative de reconnexion dans 3 secondes... ({retry_count})")
                self.tap = None
                time.sleep(self.retry_delay)

    def reset(self):
        """"Reset" button: stops using the device events and marks every device as disconnected.
        The SDK object is kept, so a new scan does not duplicate the data."""
        with self.lock:
            self._reset_status()
            self.scanning = False
        self.slot_map.clear()
        for fn in self.stop_listeners:
            fn()
