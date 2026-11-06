"""Checks the number of *_Clean.csv files per gesture and per participant in data_clean/<set>/<dataset>/Gesture_XX/PYY/.

One repetition (rep01…rep05) = one _Clean.csv file. The script warns if there are more or fewer than 5, if a participant P1…P20 is missing, or if a repetition exists twice.

    python compter_fichiers.py
    python compter_fichiers.py "C:\\path\\data_clean\\default\\newclock" --participants 20 --reps 5
"""
import argparse, os, re, sys
from collections import defaultdict

DEFAULT_ROOT = r"C:\Users\vanissum.OASIS\Desktop\Tap Software\appv2\data_clean\default\principal"
NAME_RE = re.compile(r"^P(\d+)_S\d+_Gesture_(\d+)_rep(\d+)_(\d{8}_\d{6})_Clean\.csv$", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    ap.add_argument("--participants", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gestures", type=int, default=0, help="nombre de gestes attendus (défaut : ceux trouvés dans les données)")
    a = ap.parse_args()
    if not os.path.isdir(a.root):
        print("Dossier introuvable :", a.root); return 2

    # data[participant][gesture][rep] = set of timestamps (several files for the same rep = duplicate)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    ignored = []
    for r, _d, files in os.walk(a.root):
        for f in files:
            if not f.lower().endswith("_clean.csv"):
                continue
            m = NAME_RE.match(f)
            if not m:
                ignored.append(os.path.join(r, f)); continue
            p, g, rep, ts = int(m[1]), int(m[2]), int(m[3]), m[4]
            data[p][g][rep].add(ts)

    gestures = list(range(1, a.gestures + 1)) if a.gestures else sorted({g for p in data.values() for g in p})
    problems = 0
    print(f"Dossier : {a.root}\nGestes : {len(gestures)} · répétitions attendues par geste : {a.reps}\n")
    for p in range(1, a.participants + 1):
        if p not in data:
            print(f"P{p:02d}  ✗ AUCUN fichier"); problems += 1; continue
        msgs = []
        for g in gestures:
            reps = data[p].get(g, {})
            n = len(reps)
            if n != a.reps:
                expected = set(range(1, a.reps + 1))
                miss = sorted(expected - set(reps)); extra = sorted(set(reps) - expected)
                detail = f" (manque rep{', rep'.join(f'{x:02d}' for x in miss)})" if miss else ""
                if extra: detail += f" (en trop : rep{', rep'.join(f'{x:02d}' for x in extra)})"
                msgs.append(f"Gesture_{g:02d} : {n} fichier(s) au lieu de {a.reps}{detail}")
            dup = [rp for rp, ts in reps.items() if len(ts) > 1]
            if dup:
                msgs.append(f"Gesture_{g:02d} : rep{', rep'.join(f'{x:02d}' for x in sorted(dup))} présente(s) en plusieurs fichiers")
        extra_g = sorted(set(data[p]) - set(gestures))
        if extra_g:
            msgs.append("gestes inattendus : " + ", ".join(f"Gesture_{g:02d}" for g in extra_g))
        if msgs:
            problems += 1
            print(f"P{p:02d}  ✗")
            for m in msgs: print("      -", m)
        else:
            print(f"P{p:02d}  ✓ {len(gestures)} gestes × {a.reps} répétitions")
    unknown = sorted(set(data) - set(range(1, a.participants + 1)))
    if unknown:
        print("\nParticipants hors plage P1–P%d : %s" % (a.participants, ", ".join(f"P{p:02d}" for p in unknown)))
    if ignored:
        print(f"\n{len(ignored)} fichier(s) .csv _Clean.csv au nom non reconnu (ignorés), par ex. : {ignored[0]}")
    print(f"\n{'⚠ ' + str(problems) + ' participant(s) à vérifier' if problems else '✓ Tout est conforme'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
