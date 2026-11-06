"""MODULE 5 - Recording.

Question: how is one repetition stored without delaying reception?

Owns: the file names and folders of a repetition, the in-memory buffers (one per device stream),
the sample counters, the markers.csv file, the video files, and moving a rejected repetition to
anomalie/.

Reception is never delayed: `on_sample` (a Device sink, called from the processing thread, not the
Bluetooth thread) only appends to a list in RAM; the CSV files are written once, at the end of
the repetition (`write_and_close`). Same behaviour as v3.

Inputs are plain values (participant, session, gesture, repetition, data folder), so the module
can be tested without devices, LSL or Flask.
"""
import csv
import os
import shutil
import threading
from datetime import datetime

FINGER_HEADER = ["Timestamp_LSL", "Timestamp_HW", "Type",
                 "Thumb_X", "Thumb_Y", "Thumb_Z",
                 "Index_X", "Index_Y", "Index_Z",
                 "Middle_X", "Middle_Y", "Middle_Z",
                 "Ring_X", "Ring_Y", "Ring_Z",
                 "Pinky_X", "Pinky_Y", "Pinky_Z"]
IMU_HEADER = ["Timestamp_LSL", "Timestamp_HW", "Type",
              "Accel_X", "Accel_Y", "Accel_Z",
              "Gyro_X", "Gyro_Y", "Gyro_Z"]


def pid_label(pid):
    return "TEST" if str(pid).lower() == 'test' else f"P{int(pid):02d}"


class Recorder:
    def __init__(self, base_dir, log=lambda m: None, now=datetime.now):
        self.base_dir = base_dir
        self.log = log
        self.now = now
        self.buffers = {}
        self.paths = {}               # last repetition: stream key → CSV path
        self.marker_path = None       # markers.csv of the last repetition (later markers are appended to it)
        self.marker_lock = threading.Lock()
        self.capturing = False        # FAST_RECORDING of v3
        self.finger_count = 0
        self.imu_count = 0
        self.slot_counts = {}
        self.last_video_path = None

    # ── streams ─────────────────────────────────────────────
    def set_streams(self, keys, slots):
        """Called when the device registry changes."""
        self.buffers = {k: [] for k in keys}
        self.slot_counts = {s: {"finger": 0, "imu": 0} for s in slots}

    # ── one repetition ──────────────────────────────────────
    def prepare(self, pid, session, gesture, rep, data_root):
        """Generates the paths of the repetition (rep is 0-based), creates markers.csv, empties the buffers."""
        for k in self.buffers:
            self.buffers[k].clear()
        ts = self.now().strftime('%Y%m%d_%H%M%S')
        if str(pid).lower() == 'test':
            sdir = os.path.join(self.base_dir, "datatest", "test_P99")
            prefix = f"TEST_S{session:02d}_{gesture.replace(' ', '_')}_rep{rep + 1:02d}_{ts}"
        else:
            sdir = os.path.join(data_root, f"P{int(pid):02d}")
            prefix = f"P{int(pid):02d}_S{session:02d}_{gesture.replace(' ', '_')}_rep{rep + 1:02d}_{ts}"
        os.makedirs(sdir, exist_ok=True)
        paths = {k: os.path.join(sdir, f"{prefix}_{k}.csv") for k in self.buffers}
        marker_p = os.path.join(sdir, f"{prefix}_markers.csv")
        with open(marker_p, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["Timestamp_LSL", "Marker"])
        self.paths = paths
        with self.marker_lock:
            self.marker_path = marker_p
        print(f"[Marker] fichier cree: {marker_p}", flush=True)
        self.log(f"Marker log cree: {os.path.basename(marker_p)}")
        self.log(f"Enregistrement prêt (CSV en différé) : {prefix}")
        return dict(paths)

    def start_capture(self):
        self.finger_count = 0
        self.imu_count = 0
        for c in self.slot_counts.values():
            c["finger"] = 0
            c["imu"] = 0
        self.capturing = True

    def stop_capture(self):
        self.capturing = False

    def is_capturing(self):
        return self.capturing

    def counts(self):
        return self.finger_count, self.imu_count, {sl: dict(c) for sl, c in self.slot_counts.items()}

    # ── sinks ───────────────────────────────────────────────
    def on_sample(self, key, lsl_ts, hw_ts, data_type, sample, capturing):
        """Device sink. `capturing` is the flag at reception time."""
        if not capturing:
            return
        buf = self.buffers.get(key)          # absent if the device model does not have this stream
        if buf is None:
            return
        buf.append([lsl_ts, hw_ts, data_type] + sample)
        stream, slot = key.split("_", 1)
        if stream == "finger":
            self.finger_count += 1
        else:
            self.imu_count += 1
        if slot in self.slot_counts:
            self.slot_counts[slot][stream] += 1

    def on_marker(self, ts, marker_str):
        """Synchronisation marker sink: appends the marker to the current markers.csv."""
        with self.marker_lock:
            if self.marker_path:
                try:
                    with open(self.marker_path, 'a', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerow([f"{ts:.6f}", marker_str])
                except Exception as e:
                    print(f"[Marker Log Error] {e}")

    def write_and_close(self):
        """Writes all buffers to the files in one go."""
        for k, path in self.paths.items():
            if not path:
                continue
            hdr = FINGER_HEADER if k.startswith("finger_") else IMU_HEADER
            try:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(hdr)
                    w.writerows(self.buffers.get(k, []))
            except Exception as e:
                print(f"[CSV Error] {e}")
        for k in self.buffers:
            self.buffers[k].clear()

    # ── rejected repetition ─────────────────────────────────
    def anomaly_folder(self, pid, data_root):
        if str(pid).lower() == 'test':
            return os.path.join(self.base_dir, "datatest", "anomalie", "test_P99")
        return os.path.join(data_root, "anomalie", f"P{int(pid):02d}")

    def reject_last(self, pid, data_root):
        """Moves the files of the last repetition (CSV, markers, video) to anomalie/<participant>.
        Nothing is deleted. Returns {"csv": last moved CSV path or None, "video": moved video or None}."""
        folder = self.anomaly_folder(pid, data_root)
        os.makedirs(folder, exist_ok=True)
        files = list(self.paths.values())
        with self.marker_lock:
            if self.marker_path:
                files.append(self.marker_path)
        out = {"csv": None, "video": None}
        for p in files:
            if p and os.path.exists(p):
                dest = os.path.join(folder, os.path.basename(p))
                shutil.move(p, dest)
                out["csv"] = dest
        v = self.last_video_path
        if v and os.path.exists(v):
            dest = os.path.join(folder, os.path.basename(v))
            shutil.move(v, dest)
            out["video"] = dest
        return out

    # ── video of the phone ──────────────────────────────────
    def save_video(self, video, data_root, pid, session_num, gesture, rep, t1_sys, t1_hw, t2, t3, t4, anomaly=False):
        """Saves the uploaded video with its 5 timestamps in the file name."""
        f = lambda v: f"{v:.6f}".replace(".", "-")
        if anomaly:
            folder, anomaly_str = os.path.join(data_root, "anomalie", f"P{pid:02d}"), "_RETRY"
        else:
            folder, anomaly_str = os.path.join(data_root, f"P{pid:02d}", "videos"), ""
        os.makedirs(folder, exist_ok=True)
        filename = (
            f"P{pid:02d}_S{session_num:02d}"
            f"_{gesture.replace(' ', '_')}"
            f"_rep{int(rep):02d}"
            f"_t1sys{f(t1_sys)}"      # System.nanoTime() → LSL — NTP-synced clock
            f"_t1hw{f(t1_hw)}"        # sensor hardware → LSL — exact physical moment
            f"_t2flaskStart{f(t2)}"
            f"_t3flaskStop{f(t3)}"
            f"_t4camStop{f(t4)}"
            f"{anomaly_str}"
            f".mp4"
        )
        path = os.path.join(folder, filename)
        video.save(path)
        self.last_video_path = path
        return filename
