"""Génère les répliques (voix québécoises MBROLA ca1/ca2) dans build/vo_*.wav"""
import subprocess, wave, os
LINES = {
 "intro":  ("mb-ca1", 120, 35, "Bonjour à tous. Bienvenue au CFGA de la Jonquière. Une autre belle journée."),
 "trace":  ("mb-ca2", 115, 30, "Laissez votre trace. Elle restera ici. Avec vous."),
 "hier":   ("mb-ca1", 100, 25, "Comme hier. Comme demain. Comme hier."),
 "froid":  ("mb-ca2", 135, 40, "Température extérieure: moins quarante. Restez en dedans."),
 "corr":   ("mb-ca1", 125, 30, "Rappel. Le corridor du deuxième étage est un peu plus long aujourd'hui."),
 "maths":  ("mb-ca2", 115, 45, "Qui peut me dire combien font deux plus deux? Personne? Encore."),
 "diner":  ("mb-ca1", 125, 40, "C'est l'heure du dîner. Bon appétit, là."),
 "exam":   ("mb-ca2", 110, 30, "Vous avez tout le temps. Tout. Le. Temps."),
 "cloche": ("mb-ca1", 135, 35, "C'est la pause. Merci de ne pas quitter l'établissement."),
 "demain": ("mb-ca2", 100, 30, "À demain."),
}
os.makedirs("build", exist_ok=True)
for k,(v,s,p,txt) in LINES.items():
    out=f"build/vo_{k}.wav"
    subprocess.run(["espeak-ng","-v",v,"-s",str(s),"-p",str(p),"-g","4",txt,"-w",out],check=True)
    w=wave.open(out); print(k, round(w.getnframes()/w.getframerate(),2), w.getframerate())
