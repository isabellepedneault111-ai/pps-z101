# Une journée ordinaire au CFGA de la Jonquière

Vidéo d'horreur surréaliste de 60 secondes (1280×720, 24 i/s), **entièrement générée par du code** :
aucune image, aucun son ni aucune séquence préenregistrée. Œuvre de fiction.

▶ **`cfga_journee_ordinaire.mp4`**

## La journée (et sa boucle) — selon le vrai horaire

| Heure | Scène | Ce qui cloche |
|---|---|---|
| — | Mire « PAS DE SIGNAL » → titre | « ORDINAIRE » devient « ÉTERNELLE » |
| 08:24 | Arrivée | D'abord la **vraie photo** du bâtiment (caméra de surveillance), puis sa reconstitution dessinée qui bascule : ciel rouge, soleil-œil, visages empilés dans les fenêtres, phares rouges entre les piliers du stationnement, une passante de plus en plus proche |
| 08:25 | La murale « JE LAISSE MA TRACE » (**vraies images**) | Les fenêtres du bâtiment peint s'allument, des phares s'allument dessous, SORTIE devient ENTRÉE, une nouvelle trace s'écrit : « … et elle me garde. » Les lettres vertes pourrissent en rouge |
| 08:41 | Corridor, 2e étage | Toutes les portes sont « B-104 », la silhouette avance à chaque panne |
| 09:05 | Mathématiques | 2 + 2 = 5, tous les élèves se retournent d'un coup |
| 09:25 | Pause — salle V.I.P. | Machine à café (espresso, latte, cappuccino, moka…) : « 100 % … 100 % … », le café déborde, Espresso devient « Espoir — ÉPUISÉ » ; l'enseigne V.I.P. devient « Vous · Ici · Pour toujours » |
| 10:45 | Pause — salle V.I.P. | Exactement la même pause, la même image, la même mélodie |
| 12:00 | Dîner — salle V.I.P. | « Le dîner finit à 12 h 45. Pour les autres. » |
| 13:15 | Examen | « Depuis combien de temps êtes-vous ici ? » — À REPRENDRE |
| 15:55 | Fin des cours | La cloche sonne ; SORTIE devient ENTRÉE ; personne ne sort |
| ??:?? | Retour | L'horloge recule jusqu'à 08:24 ; JOUR 04812 → 04813 |
| — | Fin | Le logo redessiné : son cercle vert, ouvert, se referme et rougit. « Inscriptions ouvertes en tout temps. Pour toujours. » |

Musique : *Daisy Bell* (Harry Dacre, 1892, domaine public), en boîte à musique désaccordée —
la chanson que chante HAL 9000 pendant qu'on le débranche.

## Lisibilité

- Toutes les paroles sont **sous-titrées** (qui parle + texte), dessinées après les effets VHS pour rester nettes.
- Les voix sont mixées sur une piste à part : le reste de la trame baisse automatiquement quand quelqu'un parle.

## Comment c'est fait

- `common.py` : ligne du temps partagée image ⇄ son (coupes, pannes, battements de cœur qui accélèrent).
- `render.py` : chaque scène est une fonction Python (Pillow + NumPy) ; post-traitement « VHS maudite »
  (bloom, déchirures, aberration chromatique, grain, barre de tracking) ; rendu parallèle envoyé à `ffmpeg`.
- `audio.py` : trame sonore synthétisée (drone avec triton, néons à 60 Hz, cœur, cloche électrique,
  *Daisy Bell* à la boîte à musique désaccordée, rejouée à l'envers au retour, micro-ondes, tic-tac, grincement de craie).
- `voices.py` : voix neuronales **Piper** (hors ligne, voix `fr-siwis-medium` téléchargée depuis les
  versions GitHub de rhasspy/piper, dans `build/piper/`), légèrement traitées en interphone.
  Les textes des répliques sont dans `common.py` (`VO_LINES`) et sont sous-titrés à l'écran.

La scène de la murale utilise la vidéo `IMG_8538.mov` (fournie) : extraire 4,6 s à partir de 0,6 s
en PNG 1280×720 / 24 i/s dans `build/murale/`, puis calculer le suivi de caméra (`offs.npy`, corrélation de phase).

### Regénérer

```bash
sudo apt-get install espeak-ng ffmpeg
pip install numpy pillow scipy piper-tts
python3 prep_murale.py && python3 voices.py && python3 audio.py && python3 render.py
ffmpeg -i build/video_seule.mp4 -i build/audio.wav -c:v libx264 -b:v 6M -c:a aac -shortest cfga_journee_ordinaire.mp4
python3 render.py --preview 8 20 44   # images-test dans build/preview/
```

Polices : Anton, VT323, Creepster, Special Elite, Reenie Beanie (Google Fonts, licences OFL / Apache).
