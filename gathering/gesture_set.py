"""MODULE 1 - Gesture Set.

Question: what is studied, how many times, and in what order?

Owns: the built-in gesture list (the "default" set), the number of repetitions, the gesture
durations, the Latin square and the gesture order of each participant. Other sets are stored by
gestureset_customdata.GestureSets (unchanged from v3); this module only reads them through
GestureSet.apply().

No Flask, no device, no file writing: everything here can be tested from its inputs alone.
"""
import random

DEFAULT_REPETITIONS = 5     # REPETITIONS_PER_GESTURE of v3
DEFAULT_DURATION_S = 3      # RECORDING_DURATION of v3 (used when a gesture has no "duration")

# Built-in gesture set (identical to GESTURE_DEFS of app_v3.py)
BUILTIN_GESTURES = [
    

  {
    "name": "Gesture 01",
    "video": "gesture_1.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras posé sur l'accoudoir, main dans le prolongement de l'avant-bras, paume vers le bas, doigts détendus et légèrement écartés.",
    "start_en": "Arm resting on the armrest, hand in line with the forearm, palm facing down, fingers relaxed and slightly apart.",
    "desc_fr": "Tous les doigts se referment simultanément vers la paume jusqu'à former un poing serré.",
    "desc_en": "All fingers curl inward simultaneously toward the palm until a tight fist is formed."
  },
  {
    "name": "Gesture 02",
    "video": "gesture_2.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras le long du corps, main dans le prolongement de l'avant-bras, paume vers soi, doigts refermés.",
    "start_en": "Arm along the body, hand in line with the forearm, palm facing toward oneself, fingers closed.",
    "desc_fr": "Les cinq doigts s'étendent complètement et s'écartent les uns des autres.",
    "desc_en": "All five fingers extend fully and spread apart from each other."
  },
  {
    "name": "Gesture 03",
    "video": "gesture_3.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras posé sur l'accoudoir, main dans le prolongement de l'avant-bras, paume vers le bas, doigts détendus.",
    "start_en": "Arm resting on the armrest, hand in line with the forearm, palm facing down, fingers relaxed.",
    "desc_fr": "L'index est placé contre la base du pouce ; l'index glisse le long du pouce, puis est relâché et s'étend complètement.",
    "desc_en": "The index finger is placed at the base of the thumb; the index slides along the thumb, then is released and fully extends."
  },
  {
    "name": "Gesture 04",
    "video": "gesture_4.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras le long du corps, main détendue, doigts dans le prolongement naturel de l'avant-bras.",
    "start_en": "Arm along the body, hand relaxed, fingers in the natural extension of the forearm.",
    "desc_fr": "L'index et le majeur s'étendent vers le haut ; l'annulaire, l'auriculaire et le pouce restent repliés.",
    "desc_en": "The index and middle fingers extend upward; the ring finger, pinky, and thumb remain closed."
  },
  {
    "name": "Gesture 05",
    "video": "gesture_5.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras posé sur l'accoudoir, paume orientée face à l'écran, pouce et index écartés en forme de L, majeur aligné avec l'index.",
    "start_en": "Arm resting on the armrest, palm facing the screen, thumb and index spread open in an L-shape, middle finger aligned with the index.",
    "desc_fr": "Le pouce, l'index et le majeur (index et majeur collés) convergent l'un vers l'autre en un mouvement de pincement, comme pour comprimer un contenu sur un écran.",
    "desc_en": "The thumb, index, and middle finger (index and middle kept together) converge toward each other in a pinching motion, as if compressing content on a screen."
  },
  {
    "name": "Gesture 06",
    "video": "gesture_6.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras posé sur l'accoudoir, paume orientée face à l'écran, pouce, index et majeur rapprochés en pince (index et majeur collés).",
    "start_en": "Arm resting on the armrest, palm facing the screen, thumb, index, and middle finger pinched together (index and middle kept together).",
    "desc_fr": "Le pouce, l'index et le majeur (index et majeur collés) s'écartent vers l'extérieur en un mouvement d'ouverture fluide, imitant le geste de zoom par pincement.",
    "desc_en": "The thumb, index, and middle finger (index and middle kept together) spread apart in a smooth outward motion, mimicking a pinch-to-zoom gesture."
  },
  {
    "name": "Gesture 07",
    "video": "gesture_7.mp4",
    "limb": "poignet",
    "limb_fr": "Poignet",
    "limb_en": "Wrist",
    "start_fr": "Bras posé sur l'accoudoir, avant-bras à plat, paume vers le bas, doigts détendus.",
    "start_en": "Arm resting on the armrest, forearm flat, palm facing down, fingers relaxed.",
    "desc_fr": "La main se fléchit vers le haut au niveau du poignet, soulevant le dos de la main en direction de l'avant-bras. L'avant-bras reste immobile.",
    "desc_en": "The hand bends upward at the wrist, raising the back of the hand toward the forearm. The forearm remains stationary."
  },
  {
    "name": "Gesture 08",
    "video": "gesture_8.mp4",
    "limb": "poignet",
    "limb_fr": "Poignet",
    "limb_en": "Wrist",
    "start_fr": "Bras posé sur l'accoudoir, avant-bras à plat, paume vers le bas, doigts détendus.",
    "start_en": "Arm resting on the armrest, forearm flat, palm facing down, fingers relaxed.",
    "desc_fr": "La main se fléchit vers le bas au niveau du poignet, rapprochant la paume de l'avant-bras. L'avant-bras reste immobile.",
    "desc_en": "The hand bends downward at the wrist, moving the palm toward the forearm. The forearm remains stationary."
  },
  {
    "name": "Gesture 09",
    "video": "gesture_9.mp4",
    "limb": "poignet",
    "limb_fr": "Poignet",
    "limb_en": "Wrist",
    "start_fr": "Bras posé sur l'accoudoir, avant-bras à plat, paume orientée vers la gauche (pouce vers le haut).",
    "start_en": "Arm resting on the armrest, forearm flat, palm facing left (thumb up).",
    "desc_fr": "Le poignet dévie latéralement vers l'extérieur dans le plan horizontal, éloignant la main du milieu du corps.",
    "desc_en": "The wrist deviates laterally outward in the horizontal plane, displacing the hand away from the body midline."
  },
  {
    "name": "Gesture 10",
    "video": "gesture_10.mp4",
    "limb": "poignet",
    "limb_fr": "Poignet",
    "limb_en": "Wrist",
    "start_fr": "Bras posé sur l'accoudoir, avant-bras à plat, paume orientée vers la gauche (pouce vers le haut).",
    "start_en": "Arm resting on the armrest, forearm flat, palm facing left (thumb up).",
    "desc_fr": "Le poignet dévie médialement dans le plan horizontal, rapprochant la main du milieu du corps.",
    "desc_en": "The wrist deviates medially in the horizontal plane, displacing the hand toward the body midline."
  },
  {
    "name": "Gesture 11",
    "video": "gesture_11.mp4",
    "limb": "poignet",
    "limb_fr": "Poignet",
    "limb_en": "Wrist",
    "start_fr": "Bras posé sur l'accoudoir, avant-bras à plat, paume orientée vers le bas, doigts légèrement fléchis.",
    "start_en": "Arm resting on the armrest, forearm flat, palm facing down , fingers slightly curled.",
    "desc_fr": "L'avant-bras effectue une rotation de façon à ce que la paume se retourne vers le haut. Le bras reste collé à l'accoudoir pendant tout le mouvement.",
    "desc_en": "The forearm rotates so that the palm turns upward."
  },
  {
    "name": "Gesture 12",
    "video": "gesture_12.mp4",
    "limb": "poignet",
    "limb_fr": "Poignet",
    "limb_en": "Wrist",
    "start_fr": "Bras posé sur l'accoudoir, coude fléchi, avant-bras vertical, main dans le prolongement, paume vers le haut.",
    "start_en": "Arm resting on the armrest, elbow bent, forearm vertical, hand in line with the forearm, palm facing up.",
    "desc_fr": "L'avant-bras effectue une rotation, retournant la paume vers le bas. Le bras reste collé à l'accoudoir pendant tout le mouvement.",
    "desc_en": "The forearm rotates, turning the palm downward. The arm remains in contact with the armrest throughout the movement."
  },
  {
    "name": "Gesture 13",
    "video": "gesture_13.mp4",
    "limb": "coude",
    "limb_fr": "Coude",
    "limb_en": "Elbow",
    "start_fr": "Bras posé sur l'accoudoir, coude fléchi, avant-bras vertical, main en forme de poing.",
    "start_en": "Arm resting on the armrest, elbow bent, forearm vertical, hand in a fist.",
    "desc_fr": "L'avant-bras s'étend lentement jusqu'à ce que le coude soit complètement déplié.",
    "desc_en": "The forearm slowly straightens until the elbow is fully extended."
  },
  {
    "name": "Gesture 14",
    "video": "gesture_14.mp4",
    "limb": "coude",
    "limb_fr": "Coude",
    "limb_en": "Elbow",
    "duration": 5,
    "start_fr": "Bras posé sur l'accoudoir, coude à 90°, avant-bras horizontal, paume orientée vers soi.",
    "start_en": "Arm resting on the armrest, elbow at 90°, forearm horizontal, palm facing toward oneself.",
    "desc_fr": "Flexion-extension répétée du coude uniquement (2 fois) ; l'avant-bras restant horizontal et la paume orientée vers l'intérieur. On amène le bras vers soi, puis on revient en position de départ, puis on ramène le bras vers soi.",
    "desc_en": "Repeated flexion-extension of the elbow only (2 times), keeping the forearm horizontal and the palm facing inward. The arm is pulled toward oneself, then returned to the starting position, then pulled toward oneself again."
  },
  {
    "name": "Gesture 15",
    "video": "gesture_15.mp4",
    "limb": "coude",
    "limb_fr": "Coude",
    "limb_en": "Elbow",
    "start_fr": "Avant-bras entièrement étendu vers la droite, main dans le prolongement, paume vers le bas.",
    "start_en": "Forearm fully extended to the right, hand in line with the forearm, palm facing down.",
    "desc_fr": "En gardant le bras supérieur immobile, le coude se fléchit pour ramener l'avant-bras vers le corps, s'arrêtant quand le bras est à 90°.",
    "desc_en": "Keeping the upper arm still, the elbow bends to bring the forearm inward toward the body, stopping when the arm reaches 90°."
  },
  {
    "name": "Gesture 16",
    "video": "gesture_16.mp4",
    "limb": "coude",
    "limb_fr": "Coude",
    "limb_en": "Elbow",
    "duration": 5,
    "start_fr": "Coude sur l'accoudoir, bras fléchi vers le haut, doigts formant un angle droit par rapport à la main.",
    "start_en": "Elbow on the armrest, arm bent upward, fingers forming a right angle relative to the hand.",
    "desc_fr": "Tracé d'un Z en trois temps : trait 1 : mouvement latéral de l'épaule formant la barre horizontale du haut ; trait 2 : flexion/extension coordonnées du coude traçant la diagonale ; trait 3 : mouvement latéral du coude formant la barre horizontale du bas.",
    "desc_en": "Tracing a Z in three strokes: stroke 1 — shoulder-driven lateral motion forms the top horizontal bar; stroke 2 — coordinated elbow flexion/extension traces the diagonal; stroke 3 — elbow-driven lateral motion forms the bottom horizontal bar."
  },
  {
    "name": "Gesture 17",
    "video": "gesture_17.mp4",
    "limb": "coude",
    "limb_fr": "Coude",
    "limb_en": "Elbow",
    "start_fr": "Bras posé sur l'accoudoir, paume vers le bas.",
    "start_en": "Arm resting on the armrest, palm facing down.",
    "desc_fr": "L'avant-bras se lève pour se poser contre le bras, comme pour lever la main, mais le coude reste sur l'accoudoir.",
    "desc_en": "The forearm lifts up against the upper arm, as if raising the hand, but the elbow remains on the armrest."
  },
  {
    "name": "Gesture 18",
    "video": "gesture_18.mp4",
    "limb": "coude",
    "limb_fr": "Coude",
    "limb_en": "Elbow",
    "start_fr": "Sans accoudoir, bras posé comme s'il était sur l'accoudoir, paume vers le bas.",
    "start_en": "Without armrest, arm held as if resting on an armrest, palm facing down.",
    "desc_fr": "L'avant-bras se dirige vers le bas, comme si on laissait pendre son bras. Le bras supérieur reste immobile.",
    "desc_en": "The forearm moves downward, as if letting the arm hang. The upper arm remains stationary."
  },
  {
    "name": "Gesture 19",
    "video": "gesture_19.mp4",
    "limb": "epaule",
    "limb_fr": "Épaule",
    "limb_en": "Shoulder",
    "start_fr": "Bras le long du corps, main détendue, doigts dans le prolongement naturel de l'avant-bras.",
    "start_en": "Arm along the body, hand relaxed, fingers in the natural extension of the forearm.",
    "desc_fr": "Le bras se soulève vers l'avant, depuis la position basse jusqu'à l'élévation maximale.",
    "desc_en": "The arm lifts forward, from the lowered position to maximum elevation."
  },
  {
    "name": "Gesture 20",
    "video": "gesture_20.mp4",
    "limb": "epaule",
    "limb_fr": "Épaule",
    "limb_en": "Shoulder",
    "start_fr": "Bras le long du corps, main détendue, doigts dans le prolongement naturel de l'avant-bras.",
    "start_en": "Arm along the body, hand relaxed, fingers in the natural extension of the forearm.",
    "desc_fr": "Le bras s'élève latéralement, s'éloignant du milieu du corps, jusqu'à l'élévation maximale.",
    "desc_en": "The arm raises laterally, moving away from the body midline, up to maximum elevation."
  },
  {
    "name": "Gesture 21",
    "video": "gesture_21.mp4",
    "limb": "epaule",
    "limb_fr": "Épaule",
    "limb_en": "Shoulder",
    "start_fr": "Bras étendu, paume vers l'écran.",
    "start_en": "Arm extended, palm facing the screen.",
    "desc_fr": "Le bras balaye horizontalement vers la droite en un seul mouvement, sans arriver à l'amplitude maximale.",
    "desc_en": "The arm sweeps horizontally to the right in a single motion, without reaching maximum range."
  },
  {
    "name": "Gesture 22",
    "video": "gesture_22.mp4",
    "limb": "epaule",
    "limb_fr": "Épaule",
    "limb_en": "Shoulder",
    "start_fr": "Bras étendu horizontalement sur le côté, main dans le prolongement, doigts détendus.",
    "start_en": "Arm extended horizontally to the side, hand in line with the forearm, fingers relaxed.",
    "desc_fr": "Le bras descend en arc de cercle contrôlé dans le plan frontal jusqu'à se retrouver le long du corps.",
    "desc_en": "The arm sweeps downward in a controlled arc through the frontal plane until it hangs straight down along the body."
  },
  {
    "name": "Gesture 23",
    "video": "gesture_23.mp4",
    "limb": "epaule",
    "limb_fr": "Épaule",
    "limb_en": "Shoulder",
    "start_fr": "Bras étendu, paume vers l'écran.",
    "start_en": "Arm extended, palm facing the screen.",
    "desc_fr": "Le bras balaye horizontalement vers la gauche en un seul mouvement, sans arriver à l'amplitude maximale.",
    "desc_en": "The arm sweeps horizontally to the left in a single motion, without reaching maximum range."
  },
  {
    "name": "Gesture 24",
    "video": "gesture_24.mp4",
    "limb": "epaule",
    "limb_fr": "Épaule",
    "limb_en": "Shoulder",
    "start_fr": "Bras étendu, poing fermé.",
    "start_en": "Arm extended, fist closed.",
    "duration": 5,
    "desc_fr": "Le bras trace un huit couché continu : depuis le centre, boucle vers la droite, retour par le centre, boucle vers la gauche, puis retour au centre.",
    "desc_en": "The arm traces a continuous sideways figure-of-eight: from center, loop right, cross back through center, loop left, and return to center."
  },
{
    "name": "Gesture 25",
    "video": "gesture_25.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Coude posé sur l'accoudoir, l'avant-bras vers le haut, paume vers le coté, doigts detendus  .",
    "start_en": "Elbow resting on the armrest, forearm pointing upward, palm on the side, fingers relaxed.",
    "desc_fr": "Les doigts se referment en poing pendant que le pouce se lève vers le haut (geste « pouce en l'air »).",
    "desc_en": "The fingers close into a fist while the thumb rises upward (thumbs-up gesture)."
  },
  {
    "name": "Gesture 26",
    "video": "gesture_26.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras posé sur l'accoudoir, main dans le prolongement de l'avant-bras, paume vers le bas, doigts refermés.",
    "start_en": "Arm resting on the armrest, hand in line with the forearm, palm facing down, fingers closed.",
    "desc_fr": "L'index s'étend vers le haut comme pour pointer quelque chose, les autres doigts restent repliés.",
    "desc_en": "The index finger extends upward as if pointing at something, the other fingers remain folded."
  },
  {
    "name": "Gesture 27",
    "video": "gesture_27.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras posé sur l'accoudoir, main dans le prolongement de l'avant-bras, paume vers le bas, doigts refermés.",
    "start_en": "Arm resting on the armrest, hand in line with the forearm, palm facing down, fingers closed.",
    "desc_fr": "L'index et le majeur s'étendent et s'écartent en forme de V, les autres doigts et le pouce restent repliés.",
    "desc_en": "The index and middle fingers extend and spread apart in a V shape, the other fingers and thumb remain folded."
  },
  {
    "name": "Gesture 28",
    "video": "gesture_28.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Coude posé sur l'accoudoir, l'avant-bras vers le haut, paume vers l'écran, doigts refermés.",
    "start_en": "Elbow resting on the armrest, forearm pointing upward, palm facing the screen, fingers closed.",
    "desc_fr": "Le pouce et l'index se rejoignent en formant un cercle, les autres doigts s'étendent naturellement.",
    "desc_en": "The thumb and index finger come together to form a circle, the other fingers extend naturally."
  },
  {
    "name": "Gesture 29",
    "video": "gesture_29.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Coude posé sur l'accoudoir, l'avant-bras vers le haut, paume vers l'écran, doigts refermés.",
    "start_en": "Elbow resting on the armrest, forearm pointing upward, palm facing the screen, fingers closed.",
    "desc_fr": "L'auriculaire s'étend vers le haut, les autres doigts restent repliés.",
    "desc_en": "The pinky finger extends upward, the other fingers remain folded."
  },
  {
    "name": "Gesture 30",
    "video": "gesture_30.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Coude posé sur l'accoudoir, l'avant-bras vers le haut, paume vers l'écran, doigts refermés.",
    "start_en": "Elbow resting on the armrest, forearm pointing upward, palm facing the screen, fingers closed.",
    "desc_fr": "Les quatre doigts se fléchissent simultanément en griffes, les pointes se rapprochant de la paume sans former un poing complet.",
    "desc_en": "All four fingers curl simultaneously into a claw shape, fingertips approaching the palm without forming a full fist."
  },
  {
    "name": "Gesture 31",
    "video": "gesture_31.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Coude posé sur l'accoudoir, l'avant-bras vers le haut, paume vers l'écran, doigts refermés.",
    "start_en": "Elbow resting on the armrest, forearm pointing upward, palm facing the screen, fingers closed.",
    "desc_fr": "Tous les doigts s'étendent complètement.",
    "desc_en": "All fingers extend fully."
  },
  {
    "name": "Gesture 32",
    "video": "gesture_32.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Bras posé sur l'accoudoir, l'avant-bras vers le haut, paume vers l'écran, doigts en position comme pour faire un 2.",
    "start_en": "Arm resting on the armrest, hand in line with the forearm, palm facing the screen, fingers positioned as if making the number 2.",
    "desc_fr": "L'index et le majeur s'écartent simultanément.",
    "desc_en": "The index and middle fingers spread apart simultaneously."
  },
  {
    "name": "Gesture 33",
    "video": "gesture_33.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "start_fr": "Coude posé sur l'accoudoir, l'avant-bras vers le haut, paume vers l'écran, poing fermé.",
    "start_en": "Elbow resting on the armrest, forearm pointing upward, palm facing the screen, fist closed.",
    "desc_fr": "L'index passe par-dessus le majeur, les deux doigts restant étendus, les autres doigts restent repliés.",
    "desc_en": "The index finger crosses over the middle finger, both fingers remaining extended, the other fingers remain folded."
  },
  {
    "name": "Gesture 34",
    "video": "gesture_34.mp4",
    "limb": "main",
    "limb_fr": "Main",
    "limb_en": "Hand",
    "duration": 9,
    "start_fr": "Avant-bras reposant sur l'accoudoir, paume vers le haut.",
    "start_en": "Forearm resting on the armrest, palm facing upward.",
    "desc_fr": "Le pouce touche successivement chacun des doigts selon les 10 positions du score de Kapandji : du bout de l'index jusqu'à la base de l'auriculaire, en passant par les positions intermédiaires de chaque doigt.",
    "desc_en": "The thumb touches each finger successively according to the 10 positions of the Kapandji score: from the tip of the index finger to the base of the pinky, passing through intermediate positions of each finger."
  }
]


def build_random_latin_square(items):
    """Random n×n Latin square, valid BY CONSTRUCTION (never fails or falls back).

    We start from the cyclic square L[r][c] = (r + c) mod n then randomly permute the rows,
    the columns and the symbols: these three operations preserve the property "each
    symbol exactly once per row and per column".

    Limitation to be aware of: a Latin square balances the POSITION of each gesture, not its
    neighbours. To also balance which gesture precedes which other (carry-over effects), a
    Williams square (n even) would be needed."""
    rng = random.Random()                  # random seed
    n = len(items)
    rows = list(range(n)); cols = list(range(n)); syms = list(items)
    rng.shuffle(rows); rng.shuffle(cols); rng.shuffle(syms)
    return [[syms[(r + c) % n] for c in cols] for r in rows]


def verify_latin_square(square):
    n = len(square); items = set(square[0])
    for row in square:
        if set(row) != items: return False
    for col in range(n):
        if {square[r][col] for r in range(n)} != items: return False
    return True


def order_for(latin, participant):
    """Gesture order of a participant: row (pid - 1) mod n of the Latin square; 'test' → row 0."""
    if str(participant).lower() == "test":
        return latin[0]
    return latin[(int(participant) - 1) % len(latin)]


class GestureSet:
    """The ACTIVE gesture set: gestures, repetitions, Latin square and data folders of its dataset."""

    def __init__(self, store):
        """store: a gestureset_customdata.GestureSets (built with the functions of this module)."""
        self.store = store
        self.set_id = None
        self.dataset = None
        self.definitions = list(BUILTIN_GESTURES)
        self.by_name = {g["name"]: g for g in self.definitions}
        self.names = [g["name"] for g in self.definitions]
        self.repetitions = DEFAULT_REPETITIONS
        self.latin = []
        self.data_root = None
        self.clean_root = None
        self.name = ""

    @classmethod
    def create(cls, base_dir):
        import gestureset_customdata as gs
        return cls(gs.GestureSets(base_dir, BUILTIN_GESTURES, DEFAULT_REPETITIONS,
                                  build_random_latin_square, verify_latin_square))

    def apply(self, sid, ds=None):
        """Makes `sid` active with its dataset `ds` (default: the last one used for this set)."""
        st = self.store.get(sid, ds)
        self.definitions = st["gestures"]
        self.by_name = {g["name"]: g for g in self.definitions}
        self.names = [g["name"] for g in self.definitions]
        self.repetitions = int(st["reps"])
        self.latin = st["latin"]
        self.data_root, self.clean_root = st["data_root"], st["clean_root"]
        self.set_id, self.dataset, self.name = sid, st["dataset"], st["name"]
        return st

    def order_for(self, participant):
        return order_for(self.latin, participant)

    def first_order(self):
        return self.latin[0]

    def duration(self, gesture_name):
        g = self.by_name.get(gesture_name)
        return g.get("duration", DEFAULT_DURATION_S) if g else DEFAULT_DURATION_S

    def definition(self, gesture_name):
        return self.by_name.get(gesture_name)

    def labels(self):
        return {g["name"]: g.get("label", "") for g in self.definitions}
