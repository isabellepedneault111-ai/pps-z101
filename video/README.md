# Une journée ordinaire au CFGA de la Jonquière

Vidéo d'horreur surréaliste de 60 secondes (1280×720, 24 i/s), **entièrement générée par du code** :
aucune image, aucun son ni aucune séquence préenregistrée. Œuvre de fiction.

▶ **`cfga_journee_ordinaire.mp4`**

## La journée (et sa boucle)

| Heure | Scène | Ce qui cloche |
|---|---|---|
| — | Mire « PAS DE SIGNAL » → titre | « ORDINAIRE » devient « ÉTERNELLE » |
| 07:59 | Arrivée | Le soleil est un œil, la neige tombe vers le haut, les fenêtres clignent ensemble, cheminées d'Arvida |
| 08:15 | Corridor B | Couloir infini, toutes les portes sont « B-104 », la silhouette avance à chaque panne de courant |
| 09:00 | Mathématiques | 2 + 2 = 5, l'enseignant est trop grand, tous les élèves se retournent d'un coup |
| 12:00 | Dîner | La file des micro-ondes ; le décompte affiche FAIM / RESTE / TOI |
| 13:15 | Examen | « Depuis combien de temps êtes-vous ici ? » — tampon « À REPRENDRE » |
| 14:30 | Pause | Machine distributrice : « Ton diplôme — ÉPUISÉ », « Encore un lundi — GRATUIT » |
| 15:59 | La cloche | SORTIE devient ENTRÉE ; dans les vitres de la porte : le stationnement, encore |
| ??:?? | Retour | L'horloge recule (le 12 est un 13), JOUR 04812 → 04813 |
| — | Fin | « Inscriptions ouvertes en tout temps. Pour toujours. » |

## Comment c'est fait

- `common.py` : ligne du temps partagée image ⇄ son (coupes, pannes, battements de cœur qui accélèrent).
- `render.py` : chaque scène est une fonction Python (Pillow + NumPy) ; post-traitement « VHS maudite »
  (bloom, déchirures, aberration chromatique, grain, barre de tracking) ; rendu parallèle envoyé à `ffmpeg`.
- `audio.py` : trame sonore synthétisée (drone avec triton, néons à 60 Hz, cœur, cloche électrique,
  *Frère Jacques* en fa mineur à la boîte à musique désaccordée, micro-ondes, tic-tac, grincement de craie).
- `voices.py` : voix d'interphone en **français québécois** (eSpeak NG + voix MBROLA `ca1`/`ca2`),
  ensuite traitées (filtre téléphone, double ralenti, réverbération, chuchotement, canon inversé).

### Regénérer

```bash
sudo apt-get install espeak-ng mbrola mbrola-ca1 mbrola-ca2 ffmpeg
pip install numpy pillow scipy
python3 voices.py && python3 audio.py && python3 render.py
ffmpeg -i build/video_seule.mp4 -i build/audio.wav -c:v libx264 -b:v 6M -c:a aac -shortest cfga_journee_ordinaire.mp4
python3 render.py --preview 8 20 44   # images-test dans build/preview/
```

Polices : Anton, VT323, Creepster, Special Elite, Reenie Beanie (Google Fonts, licences OFL / Apache).
