"""Trame sonore entièrement synthétisée (numpy/scipy) + voix MBROLA fr-CA traitées.

Sortie : build/audio.wav (48 kHz, stéréo)
"""
import os
import wave

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from common import (BEATS, CLOCHE_EXIT_FLIP, CORR_BLACKOUTS, CUTS, DINER_BEEPS,
                    DINER_LONG, DUR, EXAM_STAMP, MACHINE_DROP, MATHS_TURN,
                    MURALE_LIGHTS, MURALE_EXIT, SCENES, ev)

SR = 48000
N = int(DUR * SR)
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1974)
L = np.zeros(N)
R = np.zeros(N)
T = np.arange(N) / SR
S = {name: a for name, a, b in SCENES}


def add(sig, start, gain=1.0, pan=0.0):
    """Ajoute un signal mono (ou (n,2)) à partir de `start` secondes."""
    i = int(start * SR)
    if i >= N:
        return
    if sig.ndim == 1:
        n = min(len(sig), N - i)
        gl, gr = gain * np.sqrt((1 - pan) / 2), gain * np.sqrt((1 + pan) / 2)
        L[i:i + n] += sig[:n] * gl * 1.41
        R[i:i + n] += sig[:n] * gr * 1.41
    else:
        n = min(len(sig), N - i)
        L[i:i + n] += sig[:n, 0] * gain
        R[i:i + n] += sig[:n, 1] * gain


def filt(x, kind, f, order=4):
    sos = butter(order, f, btype=kind, fs=SR, output="sos")
    return sosfilt(sos, x)


def env_ad(n, a, d):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / d)


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def norm(x):
    m = np.max(np.abs(x)) + 1e-9
    return x / m


def reverb(x, dur=2.0, wet=0.4, seed=0):
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    ir = r.normal(0, 1, n) * np.exp(-np.arange(n) / SR * (6.0 / dur))
    ir = filt(ir, "lowpass", 5000, 2)
    y = fftconvolve(x, ir)[:len(x) + n]
    y = norm(y) * np.max(np.abs(x))
    out = np.zeros(len(x) + n)
    out[:len(x)] += x * (1 - wet)
    out[:len(y)] += y * wet
    return out


def resample(x, factor):
    """factor < 1 => plus lent et plus grave"""
    n = int(len(x) / factor)
    return np.interp(np.arange(n) * factor, np.arange(len(x)), x)


def load_vo(key):
    w = wave.open(os.path.join(HERE, "build", f"vo_{key}.wav"))
    sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    return norm(resample(x, sr / SR))


# --------------------------------------------------------------- VOIX ------
def intercom(x, seed=0):
    y = filt(x, "bandpass", [320, 3600])
    y = np.tanh(y * 4) * 0.6
    low = filt(resample(x, 0.86), "bandpass", [200, 2500]) * 0.32  # double plus lent qui « décroche »
    out = np.zeros(len(low))
    out[:len(y)] += y
    out += low
    return reverb(out, 1.8, 0.35, seed)


def chime(start, gain=0.5):
    for k, (f, dt) in enumerate(((659.25, 0.0), (523.25, 0.55))):
        d = tt(1.6)
        s = sum(np.sin(2 * np.pi * f * h * d) * a for h, a in ((1, 1), (2.01, 0.4), (3.0, 0.15)))
        s *= env_ad(len(d), 0.005, 0.5)
        add(reverb(s, 1.2, 0.3), start + dt, gain)


VO = [  # (clé, temps, gain, chime?)
    ("intro", 1.6, 0.85, True),
    ("corr", 15.6, 0.8, True),
    ("froid", 28.6, 0.8, True),
    ("diner", 34.1, 0.8, True),
    ("cloche", 45.0, 0.85, False),
]
for key, st, g, ch in VO:
    if ch:
        chime(st - 0.8)
    add(intercom(load_vo(key), len(key)), st, g)

# murale : « Laissez votre trace » — chuchoté, tout près, sans interphone
x = load_vo("trace")
x2 = resample(x, 0.93)
env = np.abs(filt(np.abs(x2), "lowpass", 40))
v = x2 * 0.5 + filt(rng.normal(0, 1, len(x2)), "bandpass", [900, 7000]) * env * 3.5
add(reverb(v, 0.8, 0.2, 9), 9.2, 0.8, 0.35)

# maths : l'enseignant, grave, proche, avec un murmure doublé
x = load_vo("maths")
low = resample(x, 0.9)
y = reverb(np.tanh(filt(low, "lowpass", 3000) * 2) * 0.7, 1.0, 0.25, 3)
add(y, 20.9, 0.9, -0.15)
wh = filt(rng.normal(0, 1, len(low)), "bandpass", [1500, 6000]) * np.abs(filt(low, "lowpass", 30)) * 3
add(wh, 21.05, 0.5, 0.6)

# examen : chuchotement qui passe d'une oreille à l'autre
x = load_vo("exam")
x2 = resample(x, 0.92)
env = np.abs(filt(np.abs(x2), "lowpass", 40))
whisper = filt(rng.normal(0, 1, len(x2)), "bandpass", [900, 7000]) * env * 4
v = x2 * 0.45 + whisper * 0.8
pan = np.sin(np.arange(len(v)) / SR * 2.2)
st = np.stack([v * np.sqrt((1 - pan) / 2), v * np.sqrt((1 + pan) / 2)], 1) * 1.3
st = st[:int((S["cloche"] - 38.7) * SR)]
st[-2000:] *= np.linspace(1, 0, 2000)[:, None]
add(st, 38.7, 0.9)

# rewind : « Comme hier. Comme demain. » en canon déformé
x = load_vo("hier")
for k, (f, g, p, d) in enumerate(((1.0, 0.8, 0, 0.0), (0.75, 0.5, -0.8, 0.25), (1.3, 0.35, 0.8, 0.5),
                                  (0.6, 0.45, 0.3, 0.9))):
    y = intercom(resample(x, f), k + 20)
    add(y, 49.5 + d, g, p)
y = intercom(load_vo("intro")[::-1], 40)
add(y[:int(5.3 * SR)], 49.6, 0.35, -0.4)

# fin : « À demain. » — grave, sec, tout près de l'oreille
x = resample(load_vo("demain"), 0.72)
x = np.tanh(filt(x, "lowpass", 4500) * 1.5)
add(reverb(x, 0.6, 0.12), 57.2, 1.0)
add(filt(rng.normal(0, 1, len(x)), "bandpass", [2000, 7000]) * np.abs(filt(x, "lowpass", 40)) * 2, 57.25, 0.4, 0.7)

# ----------------------------------------------------------- AMBIANCES -----
def drone_level(t):
    lvl = np.interp(t, [0, 1.2, 5, 49, 54.98, 55.0, 55.4, 60], [0.0, 0.25, 0.22, 0.42, 0.75, 0.0, 0.05, 0.08])
    return lvl


f0 = 43.65
dr = np.zeros(N)
for f, a in ((f0, 1.0), (f0 * 1.006, 0.9), (f0 * 1.498, 0.5), (f0 * 1.414, 0.0)):
    for h in range(1, 8):
        dr += np.sin(2 * np.pi * f * h * T + rng.uniform(0, 6)) * a / h
trit = sum(np.sin(2 * np.pi * f0 * 1.414 * h * T) / h for h in range(1, 6))
dr += trit * np.clip((T - 30) / 20, 0, 1)
dr = filt(dr, "lowpass", 380, 2) * (0.75 + 0.25 * np.sin(2 * np.pi * 0.07 * T))
dr = dr / np.max(np.abs(dr)) * drone_level(T)
L += dr
R += np.roll(dr, 300)

# vent / souffle (dehors : plus fort)
wind = filt(rng.normal(0, 1, N), "bandpass", [150, 1200], 2)
wind2 = filt(rng.normal(0, 1, N), "bandpass", [150, 1200], 2)
wl = np.interp(T, [0, 5, 5.3, 11.8, 12.0, 60], [0.05, 0.08, 0.3, 0.35, 0.06, 0.06])
wl *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.13 * T)
wl[T >= 55.0] *= 0.3
L += wind / np.max(np.abs(wind)) * wl
R += wind2 / np.max(np.abs(wind2)) * wl

# bourdonnement des néons (60 Hz, réseau d'Hydro-Québec)
hum = sum(np.sin(2 * np.pi * 60 * h * T) * a for h, a in ((1, 0.5), (2, 1.0), (3, 0.6), (4, 0.4), (6, 0.25)))
hum = np.tanh(hum * 1.5)
hl = np.zeros(N)
for a, b, lv in ((S["murale"], S["corridor"], 0.07), (S["corridor"], S["maths"], 0.16), (S["maths"], S["machine"], 0.10),
                  (S["machine"], S["diner"], 0.09), (S["diner"], S["examen"], 0.05), (S["examen"], S["cloche"], 0.11),
                  (S["cloche"], S["rewind"], 0.06)):
    hl[(T >= a) & (T < b)] = lv
crack = (rng.random(N // 480) < 0.06).repeat(480)
crack = np.pad(crack, (0, N - len(crack)))
hl *= np.where(crack, 0.15, 1.0)
for b in CORR_BLACKOUTS:  # la panne : silence total du néon
    hl[(T >= ev("corridor", b)) & (T < ev("corridor", b) + 0.12)] = 0
L += hum * hl
R += hum * hl

# ------------------------------------------------------- BATTEMENTS --------
def heartbeat():
    d = tt(0.35)
    f = 55 * np.exp(-d * 6) + 32
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * env_ad(len(d), 0.004, 0.08) + filt(rng.normal(0, 1, len(d)), "lowpass", 200) * env_ad(len(d), 0.001, 0.02) * 0.4


hb = heartbeat()
for b in BEATS:
    g = 0.25 + 0.55 * (b / 55)
    add(hb, b, g)
    add(hb * 0.7, b + 0.18, g)

# ------------------------------------------------- BOÎTE À MUSIQUE ---------
# « Daisy Bell » (Harry Dacre, 1892 — domaine public), la chanson que chante
# l'ordinateur HAL 9000 quand on le débranche. (demi-tons depuis do, temps)
DAISY_1 = [(7, 3), (4, 3), (0, 3), (-5, 3), (-3, 1), (-1, 1), (0, 1), (-3, 2), (0, 1), (-5, 6)]
DAISY_2 = [(2, 3), (7, 3), (4, 3), (0, 3), (-3, 1), (-1, 1), (0, 1), (2, 2), (4, 1), (2, 6)]
DAISY_3 = [(4, 1), (5, 1), (4, 1), (2, 1), (7, 2), (4, 1), (2, 1), (0, 5), (2, 1), (4, 2), (0, 1), (-3, 2), (0, 1),
           (-3, 1), (-5, 5)]
DAISY_4 = [(-5, 1), (0, 2), (4, 1), (2, 2), (-5, 1), (0, 2), (4, 1), (2, 1), (4, 1), (5, 1), (7, 1), (4, 1), (0, 1),
           (2, 2), (-5, 1), (0, 6)]


def musicbox_note(semi, detune=0.0, dur=1.8):
    f = 1046.5 * 2 ** ((semi + detune) / 12)
    d = tt(dur)
    s = np.sin(2 * np.pi * f * d) + 0.3 * np.sin(2 * np.pi * f * 4.02 * d) * np.exp(-d * 12) + 0.15 * np.sin(2 * np.pi * f * 2.0 * d)
    return s * env_ad(len(d), 0.002, 0.45)


def play_musicbox(notes, start, beat, slow, detune_amt, gain, seed=0, droop=0.0, end=DUR):
    r = np.random.default_rng(seed)
    t = start
    for k, (semi, beats) in enumerate(notes):
        if t >= end:
            break
        nn = musicbox_note(semi, r.normal(0, detune_amt) - droop * (t - start))
        add(reverb(nn, 1.5, 0.35, k), t, gain, r.uniform(-0.5, 0.5))
        t += beat * beats * (1 + slow * (t - start))
    return t


play_musicbox(DAISY_1, 0.95, 0.15, 0.02, 0.12, 0.24, seed=1)
# pause de 9 h 25... puis exactement la même mélodie, au même endroit, à 10 h 45
play_musicbox(DAISY_2 + DAISY_3, S["machine"] + 0.1, 0.17, 0.05, 0.35, 0.24, seed=2, droop=0.08, end=S["pause2"])
play_musicbox(DAISY_2, S["pause2"], 0.17, 0.05, 0.35, 0.26, seed=2, droop=0.08, end=S["diner"])
# rewind : à l'envers, de plus en plus vite
t_ = S["rewind"] + 0.2
for k, (semi, beats) in enumerate((DAISY_1 + DAISY_2)[::-1]):
    add(musicbox_note(semi, -0.3), t_, 0.12, 0.6 if k % 2 else -0.6)
    t_ += 0.11 * beats * max(0.25, 1 - k / 22)
# fin : « ...built for two », qui ralentit et se désaccorde comme une machine qu'on débranche
play_musicbox(DAISY_4[7:], 55.5, 0.2, 0.12, 0.25, 0.26, seed=4, droop=0.35)

# ---------------------------------------------------------- EFFETS --------
def glitch_burst(dur=0.18):
    n = int(dur * SR)
    x = rng.normal(0, 1, n)
    hold = rng.integers(8, 60)
    x = x[::hold].repeat(hold)[:n]
    sq = np.sign(np.sin(2 * np.pi * rng.uniform(200, 1800) * np.arange(n) / SR))
    y = np.where(np.arange(n) % 2000 < 1000, x, sq) * env_ad(n, 0.001, dur / 2)
    return np.round(y * 6) / 6


for c in CUTS:
    add(glitch_burst(0.2), c - 0.05, 0.35, rng.uniform(-0.6, 0.6))

# statique du début
st_ = rng.normal(0, 1, int(1.2 * SR)) * np.linspace(0.6, 0.15, int(1.2 * SR))
add(filt(st_, "highpass", 800), 0.0, 0.5)
add(np.sin(2 * np.pi * 1000 * tt(1.0)) * 0.3, 0.05, 0.25)  # tonalité de mire


def impact(dur=1.5, f0_=60):
    d = tt(dur)
    f = f0_ * np.exp(-d * 3) + 28
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-d * 2.5)
    s += filt(rng.normal(0, 1, len(d)), "lowpass", 2500) * np.exp(-d * 12) * 0.8
    return np.tanh(s * 2)


def reverse_swell(dur=0.8):
    d = tt(dur)
    s = filt(rng.normal(0, 1, len(d)), "bandpass", [400, 6000]) * (d / dur) ** 3
    return s


# corridor : chaque panne = la silhouette avance (souffle inversé + impact)
for b in CORR_BLACKOUTS:
    a = ev("corridor", b)
    add(reverse_swell(0.6), a - 0.6, 0.35)
    add(impact(1.2, 70), a, 0.7)

# maths : craie (tic-tic) puis grincement, puis le craquement des nuques
for t0 in [ev("maths", u) for u in np.arange(0.05, 4.8, 1 / 9)]:
    d = tt(0.03)
    add(filt(rng.normal(0, 1, len(d)), "bandpass", [2500, 6000]) * env_ad(len(d), 0.001, 0.008), t0, 0.25, 0.3)
for t0 in (ev("maths", 1.4), ev("maths", 5.0)):
    d = tt(0.7)
    f = 3200 + 900 * np.sin(2 * np.pi * 23 * d) + 600 * d
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * (0.6 + 0.4 * np.sin(2 * np.pi * 7 * d)) * np.sin(np.pi * d / 0.7)
    add(np.tanh(s * 2) * 0.6, t0, 0.22, 0.2)
turn = ev("maths", MATHS_TURN)
for k in range(22):
    d = tt(0.02)
    c = filt(rng.normal(0, 1, len(d)), "bandpass", [1200, 5000]) * env_ad(len(d), 0.0005, 0.004)
    add(c, turn - 0.1 + k * 0.012 + rng.uniform(0, 0.01), 0.9, rng.uniform(-0.9, 0.9))
add(impact(1.6, 55), turn, 0.85)

# murale : les fenêtres s'allument (bourdonnement), SORTIE -> ENTRÉE (impact)
add(reverse_swell(0.5), ev("murale", MURALE_LIGHTS) - 0.5, 0.3)
add(impact(0.9, 80), ev("murale", MURALE_EXIT), 0.6)

# dîner : micro-ondes, bips, puis le long BIP dissonant
mw = filt(rng.normal(0, 1, int(5.5 * SR)), "lowpass", 400) * 0.3 + np.sin(2 * np.pi * 120 * tt(5.5)) * 0.4
add(mw, S["diner"], 0.18)
for b in DINER_BEEPS:
    add(np.sign(np.sin(2 * np.pi * 2000 * tt(0.12))) * 0.3, ev("diner", b), 0.4)
lb = sum(np.sign(np.sin(2 * np.pi * f * tt(1.6))) for f in (1990, 2020, 2117, 1500, 1588))
lb = lb / 5 * np.minimum(1, tt(1.6) / 0.01)
add(filt(lb, "lowpass", 5000) * np.linspace(1, 0.6, len(lb)), ev("diner", DINER_LONG), 0.4)
add(impact(1.2, 50), ev("diner", DINER_LONG), 0.6)

# examen : tic-tac lourd, crayon, tampon
for k, t0 in enumerate(np.arange(S["examen"], S["cloche"], 0.5)):
    d = tt(0.05)
    f = 2400 if k % 2 == 0 else 1800
    s = np.sin(2 * np.pi * f * d) * env_ad(len(d), 0.0005, 0.01)
    add(reverb(s, 0.5, 0.3), t0, 0.5, -0.3 if k % 2 else 0.3)
for t0, dur in ((0.3, 2.6), (3.4, 1.0), (4.0, 1.4), (4.3, 1.0)):
    n = int(dur * SR)
    sc = filt(rng.normal(0, 1, n), "bandpass", [1500, 7000]) * (0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 7 * np.arange(n) / SR)))
    add(sc * 0.4, ev("examen", t0), 0.25, 0.35)
add(impact(1.0, 90), ev("examen", EXAM_STAMP), 0.9)

# machine : compresseur, moteur de la spirale, chute de l'objet
add(np.tanh(sum(np.sin(2 * np.pi * 58 * h * tt(6.5)) / h for h in range(1, 6)) * 2) * 0.25, S["machine"], 0.3)
d = tt((MACHINE_DROP[1] - MACHINE_DROP[0]) / 1.1)
mot = np.sign(np.sin(2 * np.pi * (90 + 30 * np.sin(2 * np.pi * 3 * d)) * d)) * 0.3
add(filt(mot, "lowpass", 900), ev("machine", MACHINE_DROP[0]), 0.4)
add(impact(0.8, 120), ev("machine", MACHINE_DROP[1]) - 0.05, 0.7)

# la cloche de 15:59 : sonnerie électrique saturée
bd = 2.6
d = tt(bd)
car = sum(np.sin(2 * np.pi * 1150 * p * d) * a for p, a in ((1, 1), (2.32, 0.6), (4.25, 0.35), (5.4, 0.2)))
clap = np.zeros(len(d))
for k in range(int(bd * 24)):
    i = int(k / 24 * SR)
    seg = np.exp(-np.arange(len(d) - i) / SR * 60)
    clap[i:] += seg
bell = np.tanh(car * clap * 2.5) * np.minimum(1, (bd - d) / 0.2)
add(bell, S["cloche"], 0.55, -0.1)
add(bell, S["cloche"] + 0.013, 0.45, 0.4)
add(impact(1.6, 45), ev("cloche", CLOCHE_EXIT_FLIP), 0.6)

# rewind : tic-tac qui s'emballe, montée (riser), puis claque
t0, step = S["rewind"], 0.25
while t0 < 55.0:
    d = tt(0.03)
    add(np.sin(2 * np.pi * 2600 * d) * env_ad(len(d), 0.0005, 0.006), t0, 0.45, rng.uniform(-0.7, 0.7))
    t0 += step
    step = max(0.028, step * 0.93)
rd = 6.0
d = tt(rd)
noise = rng.normal(0, 1, len(d))
riser = np.zeros(len(d))
for k in range(5):
    lo = 200 * 2 ** k
    riser += filt(noise, "bandpass", [lo, lo * 1.6], 2) * (d / rd) ** (1 + k * 0.4)
sh = sum(np.sin(2 * np.pi * np.cumsum(f * 2 ** (d / rd * 2)) / SR) for f in (110, 220, 440, 880)) * (d / rd) ** 2
add(np.tanh(riser * 0.5 + sh * 0.25) * 0.8, S["rewind"], 0.6)
add(impact(3.0, 50), 55.0, 1.0)
add(glitch_burst(0.35), 55.0, 0.6)
# inserts du rewind : petits éclats
from render import INSERTS  # noqa: E402

for it in INSERTS:
    add(glitch_burst(0.08), S["rewind"] + it, 0.3, rng.uniform(-0.8, 0.8))

# fin : les yeux s'ouvrent — un dernier souffle, puis coupure
add(reverse_swell(0.5), 59.25 - 0.5, 0.25)
add(impact(0.6, 35), 59.3, 0.5)

# ------------------------------------------------------------ MASTER -------
mix_ = np.stack([L, R], 1)
mix_ = filt(mix_.T, "highpass", 25, 2).T
mix_ = np.tanh(mix_ * 1.4) / np.tanh(1.4)
mix_ = mix_ / np.max(np.abs(mix_)) * 0.93
mix_[int(59.92 * SR):] = 0
fade = int(0.02 * SR)
mix_[int(59.92 * SR) - fade:int(59.92 * SR)] *= np.linspace(1, 0, fade)[:, None]
out = os.path.join(HERE, "build", "audio.wav")
with wave.open(out, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix_ * 32767).astype(np.int16).tobytes())
print("audio :", out)
