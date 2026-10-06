"""Ligne du temps partagée entre l'image (render.py) et le son (audio.py).

Tout ce qui doit être synchronisé (coupes, battements de cœur, pannes de
courant, événements-chocs) est défini ici une seule fois.
"""
import math

DUR = 60.0
FPS = 24
W, H = 1280, 720

# (nom, début, fin, vitesse) en secondes. « vitesse » : temps interne de la
# scène par seconde réelle (permet de resserrer une scène sans tout retoucher).
SCENES4 = [
    ("titre",    0.0,  4.5, 5.0 / 4.5),
    ("arrivee",  4.5, 10.5, 7.0 / 6.0),
    ("murale",  10.5, 15.0, 1.0),
    ("corridor", 15.0, 20.5, 6.5 / 5.5),
    ("maths",    20.5, 26.5, 7.0 / 6.0),
    ("machine",  26.5, 31.0, 5.0 / 4.5),
    ("pause2",   31.0, 33.0, 1.0),
    ("diner",    33.0, 38.5, 6.0 / 5.5),
    ("examen",   38.5, 43.5, 6.5 / 5.0),
    ("cloche",   43.5, 49.0, 6.0 / 5.5),
    ("rewind",   49.0, 55.0, 1.0),
    ("fin",      55.0, 60.0, 1.0),
]
SCENES = [(n, a, b) for n, a, b, _ in SCENES4]
START = {n: a for n, a, b, _ in SCENES4}
SPEED = {n: v for n, a, b, v in SCENES4}
CUTS = [s[1] for s in SCENES[1:]]


def scene_at(t):
    """-> (nom, temps interne de la scène, durée interne)"""
    for name, a, b, v in SCENES4:
        if a <= t < b:
            return name, (t - a) * v, (b - a) * v
    name, a, b, v = SCENES4[-1]
    return name, (t - a) * v, (b - a) * v


def ev(name, internal):
    """Temps absolu d'un événement donné en temps interne de scène."""
    return START[name] + internal / SPEED[name]


# Pannes de courant dans le corridor (temps interne) : la silhouette avance
CORR_BLACKOUTS = [2.0, 3.6, 4.8, 5.8]
CORR_BLACK_LEN = 0.14
# distance de la silhouette après chaque panne
CORR_FIG_Z = [26.0, 17.0, 10.0, 5.2, 2.4]

# Les élèves se retournent (temps interne de la scène maths)
MATHS_TURN = 3.0
# Les micro-ondes : bips du décompte puis le long BIP
DINER_BEEPS = [2.2, 2.9, 3.6]
DINER_LONG = 4.2
# Examen : le tampon « À REPRENDRE »
EXAM_STAMP = 5.0
# Machine : la spirale tourne et l'objet tombe
MACHINE_DROP = (2.0, 3.0)
# Cloche : la porte passe de SORTIE à ENTRÉE
CLOCHE_EXIT_FLIP = 4.2
# Murale : les fenêtres s'allument, SORTIE devient ENTRÉE, les lettres pourrissent
MURALE_LIGHTS = 1.0
MURALE_EXIT = 2.4
MURALE_ROT = 3.0

# Événements absolus qui font monter le niveau de « glitch »
EVENTS = [
    (ev("arrivee", 1.5), 1.6), (ev("arrivee", 4.6), 0.5), (ev("arrivee", 6.2), 0.4),
    (ev("murale", MURALE_LIGHTS), 0.6), (ev("murale", MURALE_EXIT), 0.8),
    (ev("maths", MATHS_TURN), 1.4),
    (ev("diner", DINER_LONG), 1.0),
    (ev("examen", EXAM_STAMP), 1.0),
    (ev("machine", MACHINE_DROP[1]), 0.7),
    (START["cloche"], 1.6), (ev("cloche", CLOCHE_EXIT_FLIP), 1.0),
    (55.0, 2.0),
] + [(ev("corridor", b), 1.2) for b in CORR_BLACKOUTS]


def glitch_level(t):
    g = 0.08 + 0.25 * (t / DUR)
    for c in CUTS:
        g += 1.0 * math.exp(-((t - c) / 0.07) ** 2)
    for et, amp in EVENTS:
        g += amp * math.exp(-((t - et) / 0.12) ** 2)
    if 49.0 <= t < 55.0:
        g += 0.15 + 0.45 * ((t - 49.0) / 6.0) ** 2
    if t >= 55.3:
        g *= 0.5
    return g


# Battements de cœur : le tempo accélère tout au long de la journée
def heartbeat_times():
    beats, t = [], 0.6
    while t < 55.0:
        beats.append(t)
        bpm = 52 + 110 * (t / 55.0) ** 1.6
        t += 60.0 / bpm
    return beats


BEATS = heartbeat_times()


def heart_pulse(t):
    """0..1, pic juste après chaque battement (pour faire pulser l'image)."""
    v = 0.0
    for b in BEATS:
        if b <= t < b + 0.6:
            v = max(v, math.exp(-(t - b) * 9))
            v = max(v, 0.6 * math.exp(-max(0.0, t - b - 0.18) * 9) if t > b + 0.18 else 0)
    return v


# Horloge affichée dans le coin (heure du « jour »).
# Horaire réel : début 8 h 25, pauses 9 h 25 et 10 h 45 (10 min), dîner
# jusqu'à 12 h 45, pauses 13 h 45 et 14 h 55.
CLOCK_START = {
    "arrivee": (8, 24, 12), "murale": (8, 25, 0), "corridor": (8, 41, 3), "maths": (9, 5, 0),
    "machine": (9, 25, 0), "pause2": (10, 45, 0), "diner": (12, 0, 0), "examen": (13, 15, 0),
}


def osd_clock(name, local):
    if name == "rewind":
        # 14:55 -> 08:24 à reculons
        a = 14 * 3600 + 55 * 60
        b = 8 * 3600 + 24 * 60
        k = min(1.0, (local / 5.6) ** 2)
        s = int(a + (b - a) * k)
    elif name == "cloche":
        # la cloche de 13 h 45... puis c'est déjà 14 h 55
        s = (13 * 3600 + 44 * 60 + 58 + int(local)) if local < 1.9 else (14 * 3600 + 55 * 60 + int(local - 1.9))
    elif name in CLOCK_START:
        h, m, s0 = CLOCK_START[name]
        s = h * 3600 + m * 60 + s0 + int(local)
    else:
        return "--:--:--"
    return f"{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}"


# Répliques parlées : (clé, début, qui parle, texte) — sous-titrées à l'écran.
# Le texte est aussi celui que lit la voix de synthèse (voices.py).
VO_LINES = [
    ("intro", 1.6, "INTERPHONE", "Bonjour à tous. Bienvenue au CFGA de la Jonquière. Une autre belle journée."),
    ("trace", 9.2, "UNE VOIX", "Laissez votre trace. Elle restera ici. Avec vous."),
    ("corr", 15.6, "INTERPHONE", "Rappel. Le corridor du deuxième étage est un peu plus long aujourd'hui."),
    ("maths", 20.9, "L'ENSEIGNANT", "Qui peut me dire combien font deux plus deux? Personne? Encore."),
    ("froid", 28.6, "INTERPHONE", "Température extérieure: moins quarante. Restez en dedans."),
    ("diner", 34.1, "INTERPHONE", "C'est l'heure du dîner. Bon appétit, là."),
    ("exam", 38.7, "UNE VOIX", "Vous avez tout le temps. Tout. Le. Temps."),
    ("cloche", 45.0, "INTERPHONE", "C'est la pause. Merci de ne pas quitter l'établissement."),
    ("hier", 49.5, "INTERPHONE", "Comme hier. Comme demain. Comme hier."),
    ("demain", 57.2, "UNE VOIX", "À demain."),
]
# facteur de ralentissement appliqué à la voix dans audio.py (pour la durée affichée)
VO_SLOW = {"trace": 0.93, "maths": 0.9, "exam": 0.92, "demain": 0.72}
