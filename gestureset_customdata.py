"""
Gesture sets and datasets.

Two levels:

    GESTURE SET     the list of gestures, the number of repetitions and the Latin square.
                    Several people can each have their own set.
    DATASET         a recording campaign of a set. A set can have as many as needed,
                    they never overwrite each other (same gestures, independent participants P01…).

Folder layout (identical for ALL sets, including the "default" set):

    gesture_sets/
        active.json                    active set + dataset (and the last dataset of each set)
        <set>.json  <set>.latin.json   definitions (the "default" set is in app_v2.py / latin_square.json)
    data/<set>/<dataset>/              raw recordings   : P01/…  anomalie/…  dataset.json
    data_clean/<set>/<dataset>/        cleaned data     : Gesture_01/P01/…  models/  classification/  reports…

The classifier's models, runs and evaluations live in data_clean/<set>/<dataset>/: they therefore
automatically follow the dataset they were trained on.

Gesture identifier = "Gesture NN" (position in the set). The readable name goes in `label`.
Set / dataset identifier = a "slug" (a-z, 0-9, - and _), fixed at creation.
"""
import os, re, json, glob, shutil
from datetime import datetime

DEFAULT_ID = "default"
DEFAULT_DATASET = "principal"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
MAX_GESTURES = 60
VIDEO_EXT = {".mp4", ".webm", ".mov", ".m4v"}
META_FILE = "dataset.json"


def slug(s):
    s = (s or "").lower()
    for a, b in (("é", "e"), ("è", "e"), ("ê", "e"), ("à", "a"), ("ç", "c"), ("ù", "u"), ("ô", "o"), ("î", "i")):
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:40] or "jeu"


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _has_files(root, ignore=()):
    """True if at least one file (outside `ignore`) exists under `root`."""
    for _r, _d, files in os.walk(root):
        if any(f not in ignore for f in files):
            return True
    return False


class GestureSets:
    def __init__(self, base_dir, builtin_defs, builtin_reps, build_latin, verify_latin):
        self.base = base_dir
        self.dir = os.path.join(base_dir, "gesture_sets")
        self.data_base = os.path.join(base_dir, "data")
        self.clean_base = os.path.join(base_dir, "data_clean")
        self.builtin_defs = builtin_defs
        self.builtin_reps = builtin_reps
        self.build_latin = build_latin
        self.verify_latin = verify_latin
        self.active_file = os.path.join(self.dir, "active.json")
        self.migrated_file = os.path.join(self.dir, "migrated_datasets.json")

    # ── chemins ─────────────────────────────────────────────
    def roots(self, sid, ds):
        """(raw folder, cleaned folder) of a dataset."""
        return os.path.join(self.data_base, sid, ds), os.path.join(self.clean_base, sid, ds)

    def _latin_path(self, sid):
        return os.path.join(self.base, "latin_square.json") if sid == DEFAULT_ID else os.path.join(self.dir, sid + ".latin.json")

    def _set_path(self, sid):
        return os.path.join(self.dir, sid + ".json")

    def ids(self):
        out = [DEFAULT_ID]
        for p in sorted(glob.glob(os.path.join(self.dir, "*.json"))):
            n = os.path.basename(p)[:-5]
            if n not in ("active", "migrated_datasets") and not n.endswith(".latin") and ID_RE.match(n):
                out.append(n)
        return out

    # ── datasets of a set ───────────────────────────────────
    def datasets(self, sid):
        """Identifiers of the set's datasets ("principal" first)."""
        names = set()
        for base in (self.data_base, self.clean_base):
            d = os.path.join(base, sid)
            if os.path.isdir(d):
                names |= {n for n in os.listdir(d) if ID_RE.match(n) and os.path.isdir(os.path.join(d, n))
                          and not (base == self.clean_base and os.path.isfile(os.path.join(d, n, "version.json")))}   # a cleaned version is not a dataset
        return sorted(names, key=lambda n: (n != DEFAULT_DATASET, n))

    def _meta_path(self, sid, ds):
        return os.path.join(self.roots(sid, ds)[0], META_FILE)

    def _make_dataset(self, sid, ds, name, notes=""):
        d, c = self.roots(sid, ds)
        os.makedirs(d, exist_ok=True)
        os.makedirs(c, exist_ok=True)
        if not os.path.isfile(self._meta_path(sid, ds)):
            _write_json(self._meta_path(sid, ds), {"name": name, "notes": notes, "created": datetime.now().isoformat(timespec="seconds")})

    def resolve(self, sid, ds=None):
        """Dataset to use for `sid`: `ds` if given (error if it does not exist), otherwise the last one
        used for this set, otherwise the first. Creates "principal" if the set has no dataset yet."""
        lst = self.datasets(sid)
        if not lst:
            self._make_dataset(sid, DEFAULT_DATASET, "Principal")
            lst = [DEFAULT_DATASET]
        if ds:
            if ds not in lst:
                raise ValueError("Dataset introuvable")
            return ds
        last = (_read_json(self.active_file, {}) or {}).get("last", {}).get(sid)
        return last if last in lst else lst[0]

    def dataset_info(self, sid, ds):
        meta = _read_json(self._meta_path(sid, ds), {}) or {}
        d, c = self.roots(sid, ds)
        rel = lambda p: os.path.relpath(p, self.base).replace(os.sep, "/")
        return {"id": ds, "name": meta.get("name") or ds, "notes": meta.get("notes", ""), "created": meta.get("created", ""),
                "has_data": self.has_data(sid, ds), "data_dir": rel(d), "clean_dir": rel(c)}

    def create_dataset(self, sid, name, notes=""):
        self.get(sid)
        name = str(name or "").strip()[:60]
        if not name:
            raise ValueError("Donne un nom au dataset")
        base = slug(name); ds = base; k = 2
        while ds in self.datasets(sid):
            ds = f"{base}-{k}"; k += 1
        self._make_dataset(sid, ds, name, str(notes or "").strip()[:300])
        return ds

    def rename_dataset(self, sid, ds, name, notes=None):
        """Changes the displayed name (the folder keeps its identifier: no path moves)."""
        self.resolve(sid, ds)
        name = str(name or "").strip()[:60]
        if not name:
            raise ValueError("Donne un nom au dataset")
        meta = _read_json(self._meta_path(sid, ds), {}) or {}
        meta["name"] = name
        if notes is not None:
            meta["notes"] = str(notes).strip()[:300]
        meta.setdefault("created", datetime.now().isoformat(timespec="seconds"))
        _write_json(self._meta_path(sid, ds), meta)

    def is_empty(self, sid, ds):
        """No file (outside dataset.json) in the raw and cleaned folders: safe to delete."""
        d, c = self.roots(sid, ds)
        import dataset_tools as dt
        if any(_has_files(r) for _n, r in dt.list_versions(c)):            # cleaned versions of this dataset
            return False
        return not (_has_files(d, ignore=(META_FILE,)) or _has_files(c))

    def delete_dataset(self, sid, ds, active):
        """`active` = (active set, active dataset)."""
        self.resolve(sid, ds)
        if (sid, ds) == tuple(active):
            raise ValueError("Active un autre dataset avant de supprimer celui-ci")
        if not self.is_empty(sid, ds):
            d, c = self.roots(sid, ds)
            raise ValueError(f"Ce dataset contient des fichiers ({os.path.relpath(d, self.base)}/, {os.path.relpath(c, self.base)}/). "
                             "Supprime ces dossiers toi-même si tu es sûr.")
        for r in self.roots(sid, ds):
            shutil.rmtree(r, ignore_errors=True)

    def has_data(self, sid, ds=None):
        """Recordings (P*) or cleaned data (Gesture_*): in `ds`, or in any dataset of the set."""
        for d_ in ([ds] if ds else self.datasets(sid)):
            for root in self.roots(sid, d_):
                if glob.glob(os.path.join(root, "P*")) or glob.glob(os.path.join(root, "Gesture_*")):
                    return True
        return False

    # ── reading a set ────────────────────────────────────
    def get(self, sid, ds=None):
        if sid == DEFAULT_ID:
            s = {"id": DEFAULT_ID, "name": "Jeu d'origine", "builtin": True, "reps": self.builtin_reps,
                 "gestures": [dict(g) for g in self.builtin_defs]}
        else:
            if not ID_RE.match(sid or "") or not os.path.isfile(self._set_path(sid)):
                raise ValueError("Jeu de gestes introuvable")
            with open(self._set_path(sid), encoding="utf-8") as f:
                s = json.load(f)
            s["id"] = sid; s["builtin"] = False
        s["dataset"] = self.resolve(sid, ds)
        s["data_root"], s["clean_root"] = self.roots(sid, s["dataset"])
        s["latin"] = self.latin(s)
        return s

    # ── active set / dataset ────────────────────────────────
    def active_id(self):
        sid = (_read_json(self.active_file, {}) or {}).get("id", DEFAULT_ID)
        return sid if sid in self.ids() else DEFAULT_ID

    def active_dataset(self, sid=None):
        sid = sid or self.active_id()
        a = _read_json(self.active_file, {}) or {}
        if a.get("id") == sid and a.get("dataset") in self.datasets(sid):
            return a["dataset"]
        return self.resolve(sid)

    def set_active(self, sid, ds=None):
        if sid not in self.ids():
            raise ValueError("Jeu de gestes introuvable")
        ds = self.resolve(sid, ds)
        last = dict((_read_json(self.active_file, {}) or {}).get("last", {}))
        last[sid] = ds
        _write_json(self.active_file, {"id": sid, "dataset": ds, "last": last})
        return ds

    def summary(self, active, active_ds):
        out = []
        for sid in self.ids():
            try:
                s = self.get(sid)
            except Exception as e:
                out.append({"id": sid, "name": sid, "error": str(e), "active": sid == active}); continue
            dss = [dict(self.dataset_info(sid, d), active=(sid == active and d == active_ds)) for d in self.datasets(sid)]
            out.append({"id": sid, "name": s["name"], "builtin": s["builtin"], "n": len(s["gestures"]), "reps": s["reps"],
                        "has_data": any(d["has_data"] for d in dss), "active": sid == active,
                        "latin_ok": bool(s["latin"]) and len(s["latin"]) == len(s["gestures"]),
                        "datasets": dss})
        return out

    # ── the set's own Latin square (shared by its datasets) ─
    def latin(self, s, force_new=False):
        names = [g["name"] for g in s["gestures"]]
        path = self._latin_path(s["id"])
        if os.path.exists(path) and not force_new:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("gestures") != names:
                print(f"[Latin Square] ⚠ Les gestes du jeu « {s['id']} » ont changé depuis la création de {path}")
            return data["square"]
        if len(names) < 2:
            raise ValueError("Il faut au moins 2 gestes pour un carré latin")
        square = self.build_latin(names)
        assert self.verify_latin(square), "Carré latin invalide !"
        _write_json(path, {"created": datetime.now().isoformat(), "gestures": names, "n": len(names), "square": square})
        print(f"[Latin Square] ✓ {len(names)}×{len(names)} sauvegardé : {path}")
        return square

    # ── creating / editing a set ────────────────────────────
    @staticmethod
    def _clean_gesture(i, g):
        def t(k, mx=600): return str(g.get(k) or "").strip()[:mx]
        out = {"name": f"Gesture {i + 1:02d}", "label": t("label", 80), "video": os.path.basename(t("video", 200)),
               "limb": "main", "limb_fr": t("limb_fr", 40) or "Main", "limb_en": t("limb_en", 40) or "Hand",
               "start_fr": t("start_fr"), "start_en": t("start_en"), "desc_fr": t("desc_fr"), "desc_en": t("desc_en")}
        d = g.get("duration")
        if d not in (None, ""):
            try:
                d = float(d)
            except (TypeError, ValueError):
                raise ValueError(f"Durée invalide pour le geste {i + 1}")
            if not 0.5 <= d <= 60: raise ValueError(f"Durée du geste {i + 1} : entre 0,5 et 60 s")
            out["duration"] = d
        return out

    def save(self, d):
        """Creates (no `id`) or edits (with `id`) a set. A new set gets a first "principal" dataset.
        Returns the id."""
        name = str(d.get("name") or "").strip()[:60]
        if not name: raise ValueError("Donne un nom au jeu")
        gl = d.get("gestures")
        if not isinstance(gl, list) or not 2 <= len(gl) <= MAX_GESTURES:
            raise ValueError(f"Un jeu contient entre 2 et {MAX_GESTURES} gestes")
        try:
            reps = int(d.get("reps", 5))
        except (TypeError, ValueError):
            raise ValueError("Nombre de répétitions invalide")
        if not 1 <= reps <= 50: raise ValueError("Répétitions : entre 1 et 50")
        gestures = [self._clean_gesture(i, g) for i, g in enumerate(gl)]

        sid = d.get("id")
        new = not sid
        if sid:
            if sid == DEFAULT_ID: raise ValueError("Le jeu d'origine est défini dans app_v2.py : duplique-le pour le modifier")
            old = self.get(sid)
            locked = self.has_data(sid)                 # n'importe lequel de ses datasets
            if locked and (len(gestures) != len(old["gestures"]) or reps != old["reps"]):
                raise ValueError("Ce jeu contient déjà des enregistrements dans au moins un dataset : on ne peut plus changer "
                                 "le nombre de gestes ni de répétitions (textes, noms et vidéos restent modifiables)")
            n_changed = len(gestures) != len(old["gestures"])
        else:
            base = slug(name); sid = base; k = 2
            while sid in self.ids() or sid == DEFAULT_ID:
                sid = f"{base}-{k}"; k += 1
            n_changed = True
        _write_json(self._set_path(sid), {"name": name, "reps": reps, "gestures": gestures,
                                           "created": d.get("created") or datetime.now().isoformat()})
        if n_changed:
            self.latin({"id": sid, "gestures": gestures}, force_new=True)       # new number of gestures → new square
        if new:
            self._make_dataset(sid, DEFAULT_DATASET, "Principal")
        return sid

    def regenerate_latin(self, sid):
        if sid == DEFAULT_ID: raise ValueError("Le carré du jeu d'origine est protégé (latin_square.json)")
        if self.has_data(sid): raise ValueError("Ce jeu contient déjà des enregistrements : le carré ne peut plus changer")
        self.latin(self.get(sid), force_new=True)

    def delete(self, sid, active):
        if sid == DEFAULT_ID: raise ValueError("Le jeu d'origine ne peut pas être supprimé")
        if sid == active: raise ValueError("Active un autre jeu avant de supprimer celui-ci")
        self.get(sid)
        for ds in self.datasets(sid):
            if not self.is_empty(sid, ds):
                raise ValueError(f"Ce jeu contient des fichiers (data/{sid}/{ds}/ ou data_clean/{sid}/{ds}/). "
                                 "Supprime ces dossiers toi-même si tu es sûr.")
        for p in (self._set_path(sid), self._latin_path(sid)):
            if os.path.isfile(p): os.remove(p)
        for b in (self.data_base, self.clean_base):
            shutil.rmtree(os.path.join(b, sid), ignore_errors=True)

    # ── one-time migration of the old folder layout ─────────
    def migrate_legacy(self):
        """Old → new, only once (marker gesture_sets/migrated_datasets.json). Only moves
        (os.rename on the same disk): nothing is copied or deleted.

            data/P*, data/anomalie            → data/default/principal/
            data_clean/*                      → data_clean/default/principal/
            data_sets/<set>/data[_clean]      → data/<set>/principal/, data_clean/<set>/principal/
        """
        if os.path.isfile(self.migrated_file):
            return []
        log = []
        set_ids = set(self.ids())

        def move(src, dst_dir):
            dst = os.path.join(dst_dir, os.path.basename(src))
            if os.path.exists(dst):
                log.append(f"⚠ ignoré (existe déjà) : {dst}")
                return
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, dst)
            log.append(f"{os.path.relpath(src, self.base)} → {os.path.relpath(dst, self.base)}")

        d_new, c_new = self.roots(DEFAULT_ID, DEFAULT_DATASET)
        if os.path.isdir(self.data_base):
            for n in sorted(os.listdir(self.data_base)):
                p = os.path.join(self.data_base, n)
                if os.path.isdir(p) and (re.fullmatch(r"P\d+", n) or n == "anomalie"):
                    move(p, d_new)
        if os.path.isdir(self.clean_base):
            for n in sorted(os.listdir(self.clean_base)):
                if n not in set_ids:                     # everything except set folders already in the new format
                    move(os.path.join(self.clean_base, n), c_new)

        legacy = os.path.join(self.base, "data_sets")
        if os.path.isdir(legacy):
            for sid in sorted(os.listdir(legacy)):
                src = os.path.join(legacy, sid)
                if not os.path.isdir(src) or sid not in set_ids or sid == DEFAULT_ID:
                    continue
                d_dst, c_dst = self.roots(sid, DEFAULT_DATASET)
                for sub, dst in (("data", d_dst), ("data_clean", c_dst)):
                    s = os.path.join(src, sub)
                    if os.path.isdir(s):
                        for n in sorted(os.listdir(s)):
                            move(os.path.join(s, n), dst)
                        try: os.rmdir(s)
                        except OSError: pass
                try: os.rmdir(src)
                except OSError: pass
            try: os.rmdir(legacy)
            except OSError: pass

        for sid in self.ids():                          # every set has at least one dataset (with its dataset.json)
            ds = self.datasets(sid)
            if DEFAULT_DATASET in ds or not ds:
                self._make_dataset(sid, DEFAULT_DATASET, "Principal")
        _write_json(self.migrated_file, {"done": datetime.now().isoformat(timespec="seconds"), "moves": log})
        return log


def register(app, mgr, busy, get_active, activate, videos_dir, get_dataset):
    """Routes /api/sets/*.
    busy() → message or None · get_active() → active set · get_dataset() → active dataset ·
    activate(sid, ds) applies the set and the dataset in app_v2.py."""
    from flask import jsonify, request
    from werkzeug.utils import secure_filename

    def guard():
        m = busy()
        return (jsonify({"ok": False, "msg": m}), 409) if m else None

    def body():
        return request.get_json(silent=True) or {}

    @app.route("/api/sets")
    def sets_list():
        a, ds = get_active(), get_dataset()
        return jsonify({"active": a, "dataset": ds, "sets": mgr.summary(a, ds)})

    @app.route("/api/sets/get")
    def sets_get():
        try:
            s = mgr.get(request.args.get("id", ""))
        except ValueError as e:
            return jsonify({"ok": False, "msg": str(e)}), 404
        return jsonify({"ok": True, "id": s["id"], "name": s["name"], "builtin": s["builtin"], "reps": s["reps"],
                        "gestures": s["gestures"], "has_data": mgr.has_data(s["id"]),
                        "latin": s["latin"][:3], "latin_n": len(s["latin"]),
                        "datasets": [mgr.dataset_info(s["id"], d) for d in mgr.datasets(s["id"])]})

    @app.route("/api/sets/activate", methods=["POST"])
    def sets_activate():
        """{id, dataset?}: activates a set (and one of its datasets, by default the last one used)."""
        g = guard()
        if g: return g
        try:
            d = body(); sid = d.get("id", "")
            mgr.get(sid)
            ds = mgr.set_active(sid, d.get("dataset") or None)
            activate(sid, ds)
        except (ValueError, AssertionError) as e:
            return jsonify({"ok": False, "msg": str(e)}), 400
        return jsonify({"ok": True, "active": sid, "dataset": ds})

    @app.route("/api/sets/save", methods=["POST"])
    def sets_save():
        g = guard()
        if g: return g
        try:
            sid = mgr.save(body())
            if sid == get_active(): activate(sid, get_dataset())       # the active set changed: reload its definitions
        except (ValueError, AssertionError) as e:
            return jsonify({"ok": False, "msg": str(e)}), 400
        return jsonify({"ok": True, "id": sid})

    @app.route("/api/sets/delete", methods=["POST"])
    def sets_delete():
        g = guard()
        if g: return g
        try:
            mgr.delete(body().get("id", ""), get_active())
        except ValueError as e:
            return jsonify({"ok": False, "msg": str(e)}), 400
        return jsonify({"ok": True})

    @app.route("/api/sets/latin", methods=["POST"])
    def sets_latin():
        g = guard()
        if g: return g
        sid = body().get("id", "")
        try:
            mgr.regenerate_latin(sid)
            if sid == get_active(): activate(sid, get_dataset())
        except (ValueError, AssertionError) as e:
            return jsonify({"ok": False, "msg": str(e)}), 400
        return jsonify({"ok": True})

    @app.route("/api/sets/latin_view")
    def sets_latin_view():
        try:
            s = mgr.get(request.args.get("id", ""))
        except ValueError as e:
            return jsonify({"ok": False, "msg": str(e)}), 404
        return jsonify({"ok": True, "square": s["latin"], "gestures": [g["name"] for g in s["gestures"]]})

    # ── datasets of a set ───────────────────────────────────
    @app.route("/api/sets/datasets")
    def datasets_list():
        sid = request.args.get("id", "")
        try:
            mgr.get(sid)
        except ValueError as e:
            return jsonify({"ok": False, "msg": str(e)}), 404
        act = (get_active(), get_dataset())
        return jsonify({"ok": True, "datasets": [dict(mgr.dataset_info(sid, d), active=(sid, d) == act) for d in mgr.datasets(sid)]})

    @app.route("/api/sets/datasets/create", methods=["POST"])
    def datasets_create():
        """{id, name, notes?, activate?}: new empty dataset in the set `id`."""
        g = guard()
        if g: return g
        d = body(); sid = d.get("id", "")
        try:
            ds = mgr.create_dataset(sid, d.get("name"), d.get("notes", ""))
            if d.get("activate"):
                mgr.set_active(sid, ds); activate(sid, ds)
        except (ValueError, AssertionError) as e:
            return jsonify({"ok": False, "msg": str(e)}), 400
        return jsonify({"ok": True, "dataset": mgr.dataset_info(sid, ds)})

    @app.route("/api/sets/datasets/rename", methods=["POST"])
    def datasets_rename():
        g = guard()
        if g: return g
        d = body()
        try:
            mgr.rename_dataset(d.get("id", ""), d.get("dataset", ""), d.get("name"), d.get("notes"))
        except ValueError as e:
            return jsonify({"ok": False, "msg": str(e)}), 400
        return jsonify({"ok": True})

    @app.route("/api/sets/datasets/delete", methods=["POST"])
    def datasets_delete():
        g = guard()
        if g: return g
        d = body()
        try:
            mgr.delete_dataset(d.get("id", ""), d.get("dataset", ""), (get_active(), get_dataset()))
        except ValueError as e:
            return jsonify({"ok": False, "msg": str(e)}), 400
        return jsonify({"ok": True})

    @app.route("/api/sets/video", methods=["POST"])
    def sets_video():
        f = request.files.get("file")
        if not f or not f.filename: return jsonify({"ok": False, "msg": "Aucun fichier"}), 400
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in VIDEO_EXT: return jsonify({"ok": False, "msg": "Vidéo .mp4, .webm, .mov ou .m4v uniquement"}), 400
        base = secure_filename(os.path.splitext(f.filename)[0]) or "video"
        name = f"set_{base}{ext}"; k = 2
        os.makedirs(videos_dir, exist_ok=True)
        while os.path.exists(os.path.join(videos_dir, name)):      # never overwrite an existing video
            name = f"set_{base}_{k}{ext}"; k += 1
        f.save(os.path.join(videos_dir, name))
        return jsonify({"ok": True, "video": name})
