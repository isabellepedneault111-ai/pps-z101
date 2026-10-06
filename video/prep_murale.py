"""Extrait les images de la murale (sources/IMG_8538.mov) et calcule le suivi de caméra."""
import glob
import os
import subprocess

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "build", "murale")
os.makedirs(out, exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "0.6", "-t", "4.6", "-i",
                os.path.join(HERE, "sources", "IMG_8538.mov"), "-vf", "fps=24,scale=1280:720",
                os.path.join(out, "m_%03d.png")], check=True)
fs = sorted(glob.glob(os.path.join(out, "m_*.png")))


def gray(f):
    return np.asarray(Image.open(f).convert("L").resize((640, 360)), np.float32)


a0 = gray(fs[0])
F0 = np.fft.fft2(a0 - a0.mean())
offs = []
for f in fs:
    a = gray(f)
    R = np.fft.fft2(a - a.mean()) * np.conj(F0)
    c = np.real(np.fft.ifft2(R / (np.abs(R) + 1e-9)))
    y, x = np.unravel_index(np.argmax(c), c.shape)
    offs.append(((x - 640 if x > 320 else x) * 2, (y - 360 if y > 180 else y) * 2))
np.save(os.path.join(out, "offs.npy"), np.array(offs))
print(len(fs), "images")
