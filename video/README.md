# Une journée ordinaire au CFGA de la Jonquière

Vidéo d'horreur surréaliste de 60 secondes (1280×720, 24 i/s), **entièrement générée par du code** :
aucune image, aucun son ni aucune séquence préenregistrée. Œuvre de fiction.

▶ **`cfga_journee_ordinaire.mp4`**

## La journée (et sa boucle) — selon le vrai horaire

| Heure | Scène | Ce qui cloche |
|---|---|---|
| — | Mire « PAS DE SIGNAL » → titre | « ORDINAIRE » devient « ÉTERNELLE » |
| 08:24 | Arrivée | Le bâtiment sur pilotis : le stationnement est dessous, on voit les voitures entre les piliers. Le soleil est un œil, la neige tombe vers le haut |
| 08:25 | La murale « JE LAISSE MA TRACE » (**vraies images**) | Les fenêtres du bâtiment peint s'allument, des phares s'allument dessous, SORTIE devient ENTRÉE, une nouvelle trace s'écrit : « … et elle me garde. » Les lettres vertes pourrissent en rouge |
| 08:41 | Corridor, 2e étage | Toutes les portes sont « B-104 », la silhouette avance à chaque panne |
| 09:05 | Mathématiques | 2 + 2 = 5, tous les élèves se retournent d'un coup |
| 09:25 | Pause — salle V.I.P. | Machine distributrice ; l'enseigne V.I.P. devient « Vous · Ici · Pour toujours » |
| 10:45 | Pause — salle V.I.P. | Exactement la même pause, la même image, la même mélodie |
| 12:00 | Dîner — salle V.I.P. | « Le dîner finit à 12 h 45. Pour les autres. » |
| 13:15 | Examen | « Depuis combien de temps êtes-vous ici ? » — À REPRENDRE |
| 13:45 → 14:55 | La cloche | L'horloge saute d'une pause à l'autre ; SORTIE devient ENTRÉE |
| ??:?? | Retour | L'horloge recule jusqu'à 08:24 ; JOUR 04812 → 04813 |
| — | Fin | « Inscriptions ouvertes en tout temps. Pour toujours. » |

Musique : *Daisy Bell* (Harry Dacre, 1892, domaine public), en boîte à musique désaccordée —
la chanson que chante HAL 9000 pendant qu'on le débranche.

## Comment c'est fait

- `common.py` : ligne du temps partagée image ⇄ son (coupes, pannes, battements de cœur qui accélèrent).
- `render.py` : chaque scène est une fonction Python (Pillow + NumPy) ; post-traitement « VHS maudite »
  (bloom, déchirures, aberration chromatique, grain, barre de tracking) ; rendu parallèle envoyé à `ffmpeg`.
- `audio.py` : trame sonore synthétisée (drone avec triton, néons à 60 Hz, cœur, cloche électrique,
  *Daisy Bell* à la boîte à musique désaccordée, rejouée à l'envers au retour, micro-ondes, tic-tac, grincement de craie).
- `voices.py` : voix d'interphone en **français québécois** (eSpeak NG + voix MBROLA `ca1`/`ca2`),
  ensuite traitées (filtre téléphone, double ralenti, réverbération, chuchotement, canon inversé).

La scène de la murale utilise la vidéo `IMG_8538.mov` (fournie) : extraire 4,6 s à partir de 0,6 s
en PNG 1280×720 / 24 i/s dans `build/murale/`, puis calculer le suivi de caméra (`offs.npy`, corrélation de phase).

### Regénérer

```bash
sudo apt-get install espeak-ng mbrola mbrola-ca1 mbrola-ca2 ffmpeg
pip install numpy pillow scipy
python3 prep_murale.py && python3 voices.py && python3 audio.py && python3 render.py
ffmpeg -i build/video_seule.mp4 -i build/audio.wav -c:v libx264 -b:v 6M -c:a aac -shortest cfga_journee_ordinaire.mp4
python3 render.py --preview 8 20 44   # images-test dans build/preview/
```

Polices : Anton, VT323, Creepster, Special Elite, Reenie Beanie (Google Fonts, licences OFL / Apache).
