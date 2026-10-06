"""Génère les répliques avec Piper (voix neuronales françaises, hors ligne).

Voix : fr-siwis-medium (interphone, voix) et fr-gilles-low (l'enseignant),
téléchargées depuis https://github.com/rhasspy/piper/releases/tag/v0.0.2
et décompressées dans build/piper/.
"""
import os
import subprocess
import wave

from common import VO_LINES

HERE = os.path.dirname(os.path.abspath(__file__))
PIPER = os.path.join(HERE, "build", "piper")
MODELS = {
    "siwis": os.path.join(PIPER, "voice-fr-siwis-medium", "fr-siwis-medium.onnx"),
    "gilles": os.path.join(PIPER, "voice-fr-gilles-low", "fr-gilles-low.onnx"),
}
# clé -> (voix, lenteur)  (lenteur > 1 = plus lent, plus posé)
STYLE = {"maths": ("siwis", 1.0), "trace": ("siwis", 1.25), "exam": ("siwis", 1.3), "demain": ("siwis", 1.5)}

os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
for key, start, who, txt in VO_LINES:
    voice, ls = STYLE.get(key, ("siwis", 1.05))
    out = os.path.join(HERE, "build", f"vo_{key}.wav")
    subprocess.run(["python3", "-m", "piper", "-m", MODELS[voice], "-f", out, "--length-scale", str(ls),
                    "--sentence-silence", "0.25"], input=txt.encode(), check=True, capture_output=True)
    w = wave.open(out)
    print(f"{key:7s} {start:5.1f} -> {start + w.getnframes() / w.getframerate():5.1f}  ({w.getframerate()} Hz)")
