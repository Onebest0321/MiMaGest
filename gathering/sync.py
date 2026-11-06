"""MODULE 4 - Synchronisation.

Question: how do all streams and events share one time base?

Owns: the common clock (LSL local_clock), the LSL outlets (one per device stream + the markers
stream), the protocol markers and the camera timestamps (first frame, system and hardware clock).

* Samples from Device arrive through `on_sample` (registered as a Device sink) and are pushed to
  LSL continuously, independently of the recording (LabRecorder chooses when it captures).
* Markers are pushed to LSL and then handed to the marker sinks (`add_marker_sink`); Recording
  uses this to write the markers.csv file of the current repetition.

The LSL library is injected (`pylsl` by default), so the module can be tested with a fake one.
"""
import threading

FINGER_SRATE = 200.0
IMU_SRATE = 208.0


class Timeline:
    def __init__(self, lsl=None, log=lambda m: None):
        if lsl is None:
            import pylsl as lsl
        self.lsl = lsl
        self.log = log
        self.lock = threading.Lock()
        self.outlets = {}          # "<stream>_<slot>" → outlet
        self.keepalive = {}        # strong references: avoids garbage collection of the LSL outlets
        self.markers = None
        self.active = False
        self.marker_sinks = []
        # camera sync (first frame of the phone video)
        self.cam_first_frame = 0.0
        self.cam_first_frame_hw = 0.0

    # ── clock ───────────────────────────────────────────────
    def now(self):
        return self.lsl.local_clock()

    # ── LSL outlets ─────────────────────────────────────────
    def _make_outlet(self, stream_type, slot):
        L = self.lsl
        if stream_type == "finger":
            info = L.StreamInfo(f"TapStrap_Finger_{slot}", "Finger", 15, FINGER_SRATE, "float32", f"tapstrap_finger_{slot.lower()}")
            chans = info.desc().append_child("channels")
            for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
                for ax in ["X", "Y", "Z"]:
                    ch = chans.append_child("channel")
                    ch.append_child_value("label", f"{finger}_{ax}")
                    ch.append_child_value("unit", "normalized")
        else:
            info = L.StreamInfo(f"TapStrap_IMU_{slot}", "IMU", 6, IMU_SRATE, "float32", f"tapstrap_imu_{slot.lower()}")
            chans = info.desc().append_child("channels")
            for label, unit in zip(["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"], ["m/s2"] * 3 + ["rad/s"] * 3):
                ch = chans.append_child("channel")
                ch.append_child_value("label", label)
                ch.append_child_value("unit", unit)
        return L.StreamOutlet(info, chunk_size=1, max_buffered=360)

    def open_outlets(self, registry):
        """Creates the LSL streams described by the device registry, plus the markers stream."""
        print("[LSL] Création des outlets...", flush=True)
        outlets = {}
        for d in registry:
            for st in d["streams"]:
                key = f"{st}_{d['slot']}"
                outlets[key] = self._make_outlet(st, d["slot"])
                print(f"[LSL] {key} OK", flush=True)
        info = self.lsl.StreamInfo("TapExp_Markers", "Markers", 1, 0, "string", "tapexp_markers")
        markers = self.lsl.StreamOutlet(info, chunk_size=1, max_buffered=360)
        print("[LSL] markers OK", flush=True)
        self.keepalive.clear()
        self.keepalive.update(outlets)
        with self.lock:
            self.markers = markers
            self.outlets.clear()
            self.outlets.update(outlets)
        self.active = True
        self.log(f"LSL : {len(outlets) + 1} streams actifs — visible dans LabRecorder")
        print("[LSL] Tous les outlets créés ✓", flush=True)

    def close_outlets(self):
        """Stops pushing device streams (the markers outlet stays, as in v3)."""
        with self.lock:
            self.outlets.clear()
        self.active = False

    # ── samples (Device sink) ───────────────────────────────
    def on_sample(self, key, lsl_ts, hw_ts, data_type, sample, capturing):
        with self.lock:
            o = self.outlets.get(key)
            if o:
                o.push_sample(sample, lsl_ts)

    # ── markers ─────────────────────────────────────────────
    def add_marker_sink(self, fn):
        self.marker_sinks.append(fn)

    def push_marker(self, marker_str, ts=None):
        marker_ts = self.now() if ts is None else ts
        with self.lock:
            if self.markers:
                self.markers.push_sample([marker_str], marker_ts)
        for sink in self.marker_sinks:
            sink(marker_ts, marker_str)
        return marker_ts

    # ── camera ──────────────────────────────────────────────
    def set_camera_first_frame(self, ts_sys, ts_hw):
        self.cam_first_frame, self.cam_first_frame_hw = ts_sys, ts_hw

    def take_camera_first_frame(self):
        """Returns (sys, hw) of the last first frame received and forgets it (0.0 if none)."""
        v = (self.cam_first_frame, self.cam_first_frame_hw)
        self.cam_first_frame = self.cam_first_frame_hw = 0.0
        return v
