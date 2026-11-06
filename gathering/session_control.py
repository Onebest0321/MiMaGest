"""MODULE 2 - Session Control.

Question: how does one trial run, and for whom?

Owns: the participant, the session number, the current gesture and repetition, the phase
(idle → ready → countdown → recording → ready … → done), the countdown, the history of
repetitions, and the decisions "retry" and "skip".

It does not touch devices, LSL or files directly. It uses three injected interfaces:

    gestures   Gesture Set   order_for(pid), first_order(), duration(name), repetitions, data_root
    recorder   Recording     prepare(), start_capture(), stop_capture(), counts(), write_and_close(),
                             reject_last(), paths
    timeline   Sync          now(), push_marker()

plus `blocked()` (returns a message when preprocessing / audit / training is running) and `log`.
`sleep` and `timer` are injectable so that a whole session can be tested in milliseconds.
"""
import threading
import time
from datetime import datetime


COUNTDOWN_SECONDS = 3


def pid_label(pid):
    """'TEST' or 'P07', as written in the markers."""
    return "TEST" if str(pid).lower() == 'test' else f"P{int(pid):02d}"


class SessionController:
    def __init__(self, gestures, recorder, timeline, log=lambda m: None, blocked=lambda: None,
                 sleep=time.sleep, timer=threading.Timer, countdown_s=COUNTDOWN_SECONDS, done_delay_s=5.0):
        self.gestures = gestures
        self.rec = recorder
        self.tl = timeline
        self.log = log
        self.blocked = blocked
        self.sleep = sleep
        self.timer = timer
        self.countdown_s = countdown_s
        self.done_delay_s = done_delay_s
        self.lock = threading.RLock()
        self.state = {
            "participant_id":      1,
            "current_gesture_idx": 0,
            "current_repetition":  0,
            "recording":           False,
            "phase":               "idle",   # idle|ready|countdown|recording|done
            "gesture_order":       gestures.first_order(),
            "countdown":           0,
            "video_file":          "gesture_demo.mp4",  # fallback if no video per gesture
            "history":             [],
            "replay_requested":    False,
            "recording_start_time": None,
            "session_number":      1,
            "lsl_record_start":    0.0,
            "lsl_record_stop":     0.0,
            "lsl_countdown_start": 0.0,
        }

    # ── queries ─────────────────────────────────────────────
    def busy(self):
        """True while a trial is counting down or recording."""
        with self.lock:
            return self.state["recording"] or self.state["countdown"] > 0

    def current(self):
        """(gesture, 0-based repetition, participant) of the current step."""
        with self.lock:
            s = self.state
            return s["gesture_order"][s["current_gesture_idx"]], s["current_repetition"], s["participant_id"]

    def view(self):
        with self.lock:
            s = self.state
            order, idx = s["gesture_order"], s["current_gesture_idx"]
            cur = order[idx] if idx < len(order) else (order[-1] if order else None)
            return {
                "participant_id":      s["participant_id"],
                "phase":               s["phase"],
                "recording":           s["recording"],
                "current_gesture":     cur,
                "current_gesture_idx": idx,
                "current_repetition":  s["current_repetition"],
                "total_repetitions":   self.gestures.repetitions,
                "total_gestures":      len(order),
                "gesture_order":       order,
                "countdown":           s["countdown"],
                "video_file":          s["video_file"],
                "history":             s["history"],
                "replay_requested":    s["replay_requested"],
                "session_number":      s["session_number"],
                "recording_elapsed":   round(time.time() - s["recording_start_time"], 1) if s["recording_start_time"] else 0,
                "current_gesture_def": self.gestures.definition(cur) if cur else None,
            }

    def record_times(self):
        with self.lock:
            return self.state["lsl_record_start"], self.state["lsl_record_stop"]

    # ── setup ───────────────────────────────────────────────
    def set_session(self, num):
        with self.lock:
            self.state["session_number"] = num
        self.log(f"Session S{num:02d} définie")

    def set_repetition(self, rep):
        """rep: 1-based. Returns an error message or None."""
        with self.lock:
            if self.state["recording"] or self.state["countdown"] > 0:
                return "Impossible pendant l'enregistrement"
            self.state["current_repetition"] = max(0, min(rep - 1, self.gestures.repetitions - 1))
            if self.state["phase"] == "done":
                self.state["phase"] = "ready"
        self.log(f"Répétition fixée manuellement à la n°{rep}")
        return None

    def set_participant(self, raw_pid):
        with self.lock:
            pid = 'test' if str(raw_pid).lower() == 'test' else int(raw_pid)
            self.state["participant_id"] = pid
            self.state["gesture_order"] = self.gestures.order_for(pid)
            self.state["current_gesture_idx"] = 0
            self.state["current_repetition"] = 0
            self.state["phase"] = "ready"
            self.state["history"] = []
        self.log("Participant Test — ordre défini" if pid == 'test' else f"Participant P{pid:02d} — ordre défini")

    def reset(self):
        """New gesture set or dataset: back to participant 1, idle."""
        with self.lock:
            self.state.update(participant_id=1, gesture_order=self.gestures.first_order(), current_gesture_idx=0,
                              current_repetition=0, phase="idle", history=[])

    def set_video(self, filename):
        with self.lock:
            self.state["video_file"] = filename

    def request_replay(self):
        with self.lock:
            self.state["replay_requested"] = True
        self.log("↺ Replay vidéo")

    def ack_replay(self):
        with self.lock:
            self.state["replay_requested"] = False

    # ── one trial ───────────────────────────────────────────
    def start(self):
        """Starts the countdown then the recording, in a thread. Returns an error message or None."""
        msg = self.blocked()
        if msg:
            return msg
        with self.lock:
            if self.state["recording"] or self.state["countdown"] > 0:
                return "Déjà en cours"
            self.state["phase"] = "countdown"
            self.state["countdown"] = self.countdown_s
        threading.Thread(target=self._run_countdown, daemon=True).start()
        return None

    def _run_countdown(self):
        lsl_countdown_start = self.tl.now()
        with self.lock:
            self.state["lsl_countdown_start"] = lsl_countdown_start
            gesture, rep, pid, session = (self.state["gesture_order"][self.state["current_gesture_idx"]],
                                          self.state["current_repetition"], self.state["participant_id"],
                                          self.state["session_number"])
        self.rec.prepare(pid, session, gesture, rep, self.gestures.data_root)
        pid_str = pid_label(pid)
        self.tl.push_marker(f"COUNTDOWN_START|{pid_str}|{gesture}|rep{rep + 1:02d}|s={self.countdown_s}")

        for i in range(self.countdown_s, 0, -1):
            with self.lock:
                self.state["countdown"] = i
            self.sleep(1)

        t_start = time.time()
        lsl_record_start = self.tl.now()
        with self.lock:
            self.state["countdown"] = 0
            self.state["recording"] = True
            self.state["phase"] = "recording"
            self.state["recording_start_time"] = t_start
            self.state["lsl_record_start"] = lsl_record_start
            self.state["lsl_countdown_start"] = self.tl.now() - self.countdown_s
        self.rec.start_capture()

        self.tl.push_marker(f"COUNTDOWN_END|{pid_str}|{gesture}|rep{rep + 1:02d}", ts=lsl_record_start)
        self.tl.push_marker(f"START|{pid_str}|{gesture}|rep{rep + 1:02d}", ts=lsl_record_start)
        self.log(f"▶ START — {gesture} rep {rep + 1}")

        duration = self.gestures.duration(gesture)
        self.log(f"Enregistrement {gesture} pendant {duration}s")
        self.sleep(duration)
        self.stop()

    def stop(self):
        """Ends the current recording (automatically after the duration, or from the Stop button)."""
        self.rec.stop_capture()
        lsl_record_stop = self.tl.now()
        with self.lock:
            s = self.state
            if not s["recording"]:
                return {"ok": False, "msg": "Pas en cours"}
            s["recording"] = False
            s["recording_start_time"] = None
            s["lsl_record_stop"] = lsl_record_stop
            fc, ic, slot_counts = self.rec.counts()
            gesture = s["gesture_order"][s["current_gesture_idx"]]
            rep = s["current_repetition"]
            pid = s["participant_id"]
            s["history"].append({
                "gesture": gesture, "rep": rep + 1,
                "finger_count": fc, "imu_count": ic,
                "slot_counts": slot_counts,
                "imu_count_tap1": slot_counts.get("TAP1", {}).get("imu", 0),   # old fields (v2)
                "imu_count_tap2": slot_counts.get("TAP2", {}).get("imu", 0),
                "ts": datetime.now().strftime('%H:%M:%S'),
                "status": "ok", "csv_paths": dict(self.rec.paths),
            })
            s["current_repetition"] += 1
            if s["current_repetition"] >= self.gestures.repetitions:
                s["current_repetition"] = 0
                s["current_gesture_idx"] += 1
            is_last_step = s["current_gesture_idx"] >= len(s["gesture_order"])
            # Always "ready" first, even after the last gesture, so the phone sees recording → ready
            # and sends its video; "done" comes a few seconds later.
            s["phase"] = "ready"
        self.rec.write_and_close()
        self.tl.push_marker(f"STOP|{pid_label(pid)}|{gesture}|rep{rep + 1:02d}|f={fc}|imu={ic}", ts=lsl_record_stop)
        self.log(f"■ STOP — {fc} pts doigts, {ic} pts IMU")
        if is_last_step:
            self.timer(self.done_delay_s, self._finalize_done).start()
        return {"ok": True}

    def _finalize_done(self):
        with self.lock:
            if self.state["phase"] == "ready" and self.state["current_gesture_idx"] >= len(self.state["gesture_order"]):
                self.state["phase"] = "done"
        self.log("✓ Expérience terminée (phase 'done')")

    # ── failed trials ───────────────────────────────────────
    def retry(self):
        """Goes back one repetition and moves its files to anomalie/. Returns an error message or None."""
        with self.lock:
            s = self.state
            if s["recording"]:
                return "Stop d'abord"
            if s["phase"] not in ["ready", "done"]:
                return "Pas recommençable"
            if s["history"]:
                s["history"][-1]["status"] = "retried"
            if s["current_repetition"] == 0 and s["current_gesture_idx"] > 0:
                s["current_gesture_idx"] -= 1
                s["current_repetition"] = self.gestures.repetitions - 1
                s["phase"] = "ready"
            elif s["current_repetition"] > 0:
                s["current_repetition"] -= 1
            gesture = s["gesture_order"][s["current_gesture_idx"]]
            rep = s["current_repetition"]
            pid = s["participant_id"]
        moved = self.rec.reject_last(pid, self.gestures.data_root)
        with self.lock:
            if self.state["history"]:
                if moved["csv"]:
                    self.state["history"][-1]["csv_paths"]["anomaly"] = moved["csv"]
                if moved["video"]:
                    self.state["history"][-1]["video_path_anomaly"] = moved["video"]
        self.tl.push_marker(f"RETRY|{pid_label(pid)}|{gesture}|rep{rep + 1:02d}")
        self.log(f"↩ RETRY — {gesture} rep {rep + 1}")
        return None

    def skip(self):
        """Skips the current gesture. Returns an error message or None."""
        with self.lock:
            s = self.state
            if s["recording"]:
                return "Stop d'abord"
            gesture = s["gesture_order"][s["current_gesture_idx"]]
            pid = s["participant_id"]
            s["history"].append({
                "gesture": gesture, "rep": s["current_repetition"] + 1,
                "finger_count": 0, "imu_count": 0,
                "ts": datetime.now().strftime('%H:%M:%S'),
                "status": "skipped", "csv_paths": {},
            })
            s["current_repetition"] = 0
            s["current_gesture_idx"] += 1
            s["phase"] = "done" if s["current_gesture_idx"] >= len(s["gesture_order"]) else "ready"
        self.tl.push_marker(f"SKIP|{pid_label(pid)}|{gesture}")
        self.log(f"⏭ SKIP — {gesture}")
        return None
