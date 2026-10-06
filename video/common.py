"""Ligne du temps partagée entre l'image (render.py) et le son (audio.py).

Tout ce qui doit être synchronisé (coupes, battements de cœur, pannes de
courant, événements-chocs) est défini ici une seule fois.
"""
import math

DUR = 60.0
FPS = 24
W, H = 1280, 720

# (nom, début, fin) en secondes
SCENES = [
    ("titre",    0.0,  5.0),
    ("arrivee",  5.0, 12.0),
    ("corridor", 12.0, 18.5),
    ("maths",    18.5, 25.5),
    ("diner",    25.5, 31.5),
    ("examen",   31.5, 38.0),
    ("machine",  38.0, 43.0),
    ("cloche",   43.0, 49.0),
    ("rewind",   49.0, 55.0),
    ("fin",      55.0, 60.0),
]
CUTS = [s[1] for s in SCENES[1:]]


def scene_at(t):
    for name, a, b in SCENES:
        if a <= t < b:
            return name, t - a, b - a
    name, a, b = SCENES[-1]
    return name, t - a, b - a


# Pannes de courant dans le corridor (temps local) : la silhouette avance
CORR_BLACKOUTS = [2.0, 3.6, 4.8, 5.8]
CORR_BLACK_LEN = 0.14
# distance de la silhouette après chaque panne
CORR_FIG_Z = [26.0, 17.0, 10.0, 5.2, 2.4]

# Les élèves se retournent (temps local de la scène maths)
MATHS_TURN = 3.0
# Les micro-ondes : bips du décompte (temps local) puis le long BIP
DINER_BEEPS = [2.2, 2.9, 3.6]
DINER_LONG = 4.2
# Examen : le tampon « À REPRENDRE »
EXAM_STAMP = 5.0
# Machine : la spirale tourne et l'objet tombe
MACHINE_DROP = (2.0, 3.0)
# Cloche : la porte passe de SORTIE à ENTRÉE
CLOCHE_EXIT_FLIP = 4.2

# Événements absolus qui font monter le niveau de « glitch »
EVENTS = [
    (5.0 + 2.5, 0.5), (5.0 + 4.6, 0.5), (5.0 + 6.2, 0.4),
    (18.5 + MATHS_TURN, 1.4),
    (25.5 + DINER_LONG, 1.0),
    (31.5 + EXAM_STAMP, 1.0),
    (38.0 + MACHINE_DROP[1], 0.7),
    (43.0, 1.6), (43.0 + CLOCHE_EXIT_FLIP, 1.0),
    (55.0, 2.0),
] + [(12.0 + b, 1.2) for b in CORR_BLACKOUTS]


def glitch_level(t):
    g = 0.12 + 0.35 * (t / DUR)
    for c in CUTS:
        g += 1.3 * math.exp(-((t - c) / 0.10) ** 2)
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


# Horloge affichée dans le coin (heure du « jour »)
CLOCK_START = {
    "arrivee": (7, 59, 12), "corridor": (8, 15, 3), "maths": (9, 0, 0),
    "diner": (12, 0, 0), "examen": (13, 15, 0), "machine": (14, 30, 0),
    "cloche": (15, 59, 54),
}


def osd_clock(name, local):
    if name == "rewind":
        # 16:00:00 -> 07:59:00 à reculons
        a = 16 * 3600
        b = 7 * 3600 + 59 * 60
        k = min(1.0, (local / 5.6) ** 2)
        s = int(a + (b - a) * k)
    elif name in CLOCK_START:
        h, m, s0 = CLOCK_START[name]
        s = h * 3600 + m * 60 + s0 + int(local)
    else:
        return "--:--:--"
    return f"{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}"
