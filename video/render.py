"""Une journée ordinaire au CFGA de la Jonquière — rendu image par image.

Chaque scène est une fonction (t_local, fi, rng) -> PIL.Image 1280x720.
Les images passent ensuite par une chaîne d'effets « VHS maudite »
(bloom, déchirures, aberration chromatique, grain, lignes de balayage).

Usage : python3 render.py [--preview SECONDES...] [--out fichier.mp4]
"""
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from common import VO_LINES, VO_SLOW
from common import ev as ev_abs
from common import (MURALE_EXIT, MURALE_LIGHTS, MURALE_ROT, CLOCHE_EXIT_FLIP, CORR_BLACK_LEN, CORR_BLACKOUTS,
                    CORR_FIG_Z, DINER_BEEPS, DINER_LONG, DUR, EXAM_STAMP, FPS,
                    MACHINE_DROP, MATHS_TURN, H, W, glitch_level, heart_pulse,
                    osd_clock, scene_at)

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(HERE, "fonts")
FONTS = {
    "anton": FD + "/Anton-Regular.ttf",
    "vt": FD + "/VT323-Regular.ttf",
    "creep": FD + "/Creepster-Regular.ttf",
    "type": FD + "/SpecialElite-Regular.ttf",
    "hand": FD + "/ReenieBeanie-Regular.ttf",
    "sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "inter": "/usr/share/fonts/opentype/inter/Inter-Medium.otf",
}
_fc = {}


def F(name, size):
    key = (name, int(size))
    if key not in _fc:
        _fc[key] = ImageFont.truetype(FONTS[name], max(4, int(size)))
    return _fc[key]


_cache = {}


def cached(key, builder):
    if key not in _cache:
        _cache[key] = builder()
    return _cache[key]


def lerp(a, b, k):
    return a + (b - a) * k


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def mix(c1, c2, k):
    return tuple(int(lerp(a, b, k)) for a, b in zip(c1, c2))


def vgrad(w, h, stops):
    """Dégradé vertical. stops = [(pos 0..1, (r,g,b)), ...]"""
    ys = np.linspace(0, 1, h)
    out = np.zeros((h, w, 3), np.float32)
    for ch in range(3):
        col = np.interp(ys, [s[0] for s in stops], [s[1][ch] for s in stops])
        out[:, :, ch] = col[:, None]
    return out


def to_img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def shaky_text(d, xy, text, font, fill, rng, amp=3, anchor="mm", stroke=0,
               stroke_fill=(0, 0, 0), split=0):
    """Texte dont chaque lettre tremble. split>0 : double image rouge/cyan."""
    total = font.getlength(text)
    x0, y0 = xy
    if anchor[0] == "m":
        x0 -= total / 2
    elif anchor[0] == "r":
        x0 -= total
    x = x0
    for ch in text:
        dx, dy = rng.normal(0, amp, 2)
        if split:
            d.text((x + dx - split, y0 + dy), ch, font=font, fill=(255, 0, 40),
                   anchor="l" + anchor[1])
            d.text((x + dx + split, y0 + dy), ch, font=font, fill=(0, 230, 255),
                   anchor="l" + anchor[1])
        d.text((x + dx, y0 + dy), ch, font=font, fill=fill, anchor="l" + anchor[1],
               stroke_width=stroke, stroke_fill=stroke_fill)
        x += font.getlength(ch)


def paste_rot_text(img, xy, text, font, fill, angle):
    tw = int(font.getlength(text)) + 20
    th = int(font.size * 1.6)
    t = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    ImageDraw.Draw(t).text((10, th // 2), text, font=font, fill=fill, anchor="lm")
    t = t.rotate(angle, expand=True, resample=Image.BICUBIC)
    img.paste(t, (int(xy[0] - t.width / 2), int(xy[1] - t.height / 2)), t)


def silhouette(d, cx, base, h, col=(5, 3, 6), arms=1.0, head_dx=0):
    """Silhouette humaine trop longue (pieds à `base`, hauteur h)."""
    hw = h * 0.11
    head_r = h * 0.07
    top = base - h
    d.ellipse((cx - head_r + head_dx, top, cx + head_r + head_dx, top + 2.3 * head_r), fill=col)
    neck = top + 2.1 * head_r
    d.polygon([(cx - hw, neck + h * 0.05), (cx + hw, neck + h * 0.05),
               (cx + hw * 0.75, base - h * 0.42), (cx + hw * 0.5, base),
               (cx - hw * 0.5, base), (cx - hw * 0.75, base - h * 0.42)], fill=col)
    d.ellipse((cx - hw * 1.1, neck, cx + hw * 1.1, neck + h * 0.12), fill=col)
    # bras qui descendent jusqu'aux genoux
    for s in (-1, 1):
        d.line([(cx + s * hw, neck + h * 0.07), (cx + s * hw * 1.25, base - h * 0.45),
                (cx + s * hw * 1.15, base - h * 0.45 + h * 0.25 * arms)],
               fill=col, width=max(2, int(h * 0.035)))


# ---------------------------------------------------------------- TITRE ----
def s_titre(t, fi, rng):
    if t < 1.15:
        img = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(img)
        cols = [(192, 192, 192), (192, 192, 0), (0, 192, 192), (0, 192, 0),
                (192, 0, 192), (192, 0, 0), (0, 0, 192)]
        bw = W / 7
        for i, c in enumerate(cols):
            d.rectangle((i * bw, 0, (i + 1) * bw, H * 0.67), fill=c)
        for i, c in enumerate([(0, 0, 192), (19, 19, 19), (192, 0, 192), (19, 19, 19),
                               (0, 192, 192), (19, 19, 19), (192, 192, 192)]):
            d.rectangle((i * bw, H * 0.67, (i + 1) * bw, H * 0.75), fill=c)
        d.rectangle((0, H * 0.75, W, H), fill=(10, 10, 20))
        d.text((W / 2, H * 0.86), "PAS DE SIGNAL", font=F("vt", 84),
               fill=(255, 255, 255), anchor="mm")
        a = np.asarray(img, np.float32)
        a = np.roll(a, int(rng.normal(0, 40)), axis=1)
        n = rng.random((H // 3 + 1, W // 3 + 1)).repeat(3, 0).repeat(3, 1)[:H, :W, None] * 255
        k = 0.85 - 0.6 * (t / 1.15)
        a = a * (1 - k) + n * k
        return to_img(a)

    tt = t - 1.15
    pulse = heart_pulse(t)

    def bg():
        a = vgrad(W, H, [(0, (6, 0, 10)), (0.55, (40, 0, 18)), (1, (8, 0, 4))])
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.hypot((xx - W / 2) / W, (yy - H / 2) / H)
        a += (np.exp(-r * 5) * 120)[..., None] * np.array([1.0, 0.05, 0.12])
        # casiers fantômes
        for x in range(0, W, 64):
            a[:, x:x + 2] *= 0.55
        return a

    a = cached("titre_bg", bg).copy()
    a *= 1 + 0.6 * pulse
    img = to_img(a)
    d = ImageDraw.Draw(img)

    glitchy = (2.2 < tt < 2.4) or (3.0 < tt < 3.12) or (tt > 3.45 and rng.random() < 0.55)
    word = "ÉTERNELLE" if glitchy else "ORDINAIRE"
    shaky_text(d, (W / 2, 175), "UNE JOURNÉE", F("anton", 118), (255, 255, 255),
               rng, amp=1.5 + 3 * glitchy, split=5)
    shaky_text(d, (W / 2, 325), word, F("anton", 172),
               (255, 40, 60) if glitchy else (255, 245, 210), rng,
               amp=2 + 7 * glitchy, split=8)
    sub = "au CFGA de la Jonquière"
    n = int(clamp(tt / 1.0) * len(sub))
    d.text((W / 2, 455), sub[:n], font=F("type", 54), fill=(230, 220, 200), anchor="mm")
    if tt > 1.2:
        s2 = "Saguenay · un lundi comme les autres"
        n2 = int(clamp((tt - 1.2) / 1.1) * len(s2))
        d.text((W / 2, 515), s2[:n2] + ("_" if int(t * 4) % 2 else ""), font=F("vt", 44),
               fill=(120, 255, 140), anchor="mm")
    return img


# -------------------------------------------------------------- ARRIVÉE ----
def eye_img(r, look, lid, iris=(170, 15, 25)):
    s = 2 * r
    e = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(e)
    d.ellipse((0, 0, s, s), fill=(255, 235, 200, 255))
    for k in range(6):
        a = k * 1.05
        d.line([(r + math.cos(a) * r, r + math.sin(a) * r),
                (r + math.cos(a) * r * 0.55, r + math.sin(a) * r * 0.6)], fill=(200, 40, 40, 255), width=2)
    ix, iy = r + look[0], r + look[1]
    d.ellipse((ix - r * 0.6, iy - r * 0.6, ix + r * 0.6, iy + r * 0.6), fill=iris + (255,))
    pr = r * 0.25
    d.ellipse((ix - pr, iy - pr, ix + pr, iy + pr), fill=(0, 0, 0, 255))
    d.ellipse((ix - pr * 1.2, iy - pr * 1.3, ix - pr * 0.6, iy - pr * 0.7), fill=(255, 255, 255, 230))
    if lid > 0:
        d.rectangle((0, 0, s, s * lid), fill=(90, 5, 20, 255))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, s, s), fill=255)
    e.putalpha(Image.fromarray(np.minimum(np.asarray(e.getchannel("A")), np.asarray(mask))))
    return e


def blink(t, times, dur=0.28):
    v = 0.0
    for b in times:
        if b <= t < b + dur:
            v = max(v, math.sin(math.pi * (t - b) / dur))
    return v


# Le vrai bâtiment (d'après la photo) : long bloc crème de trois étages, hautes
# fenêtres sombres verticales, aile basse en pierre à gauche, rez-de-chaussée en
# retrait sur piliers — le stationnement est dessous, on voit les voitures entre les piliers.
BX0, BX1, BTOP, BMID, GROUND = 250, 1170, 168, 468, 566
STRIPS = [300 + i * 103 for i in range(9)]          # fenêtres verticales
RPIL = [330 + i * 92 for i in range(10)]             # piliers du rez-de-chaussée
PARKED = [(x, 540, 0.42) for x in range(360, 1140, 92)]
PHOTO = os.path.join(HERE, "sources", "batiment.jpg")
PHOTO_END = 1.5   # temps interne : la vraie photo, puis la version dessinée


def draw_logo(img, cx, cy, size, close=0.0, lit=0.0, t=0.0):
    """Logo du CFGA redessiné en code (canevas d'origine 447 px).
    close : 0 = cercle ouvert comme le vrai logo, 1 = le cercle se referme (rouge)."""
    sc = size / 447

    def P(x, y):
        return (cx + (x - 225) * sc, cy + (y - 225) * sc)

    d = ImageDraw.Draw(img)
    R = 195 * sc
    col = mix((122, 178, 62), (220, 10, 20), clamp(close * 1.3))
    end = 210 + 75 * close
    d.arc((cx - R, cy - R, cx + R, cy + R), start=-75, end=end, fill=col, width=max(2, int(12 * sc)))
    blue = (28, 78, 150)
    d.rectangle((*P(50, 145), *P(332, 255)), fill=blue)
    for x0, x1 in ((50, 78), (135, 160), (215, 240), (305, 332)):
        d.rectangle((*P(x0, 250), *P(x1, 290)), fill=blue)
    for r, y in enumerate((160, 205)):
        for c in range(7):
            x = 68 + c * 38
            box = (*P(x, y), *P(x + 22, y + 34))
            if lit > 0 and hsh(r, c, 5) < lit:
                d.rounded_rectangle(box, radius=max(1, int(4 * sc)), fill=(235, 255, 90))
                hx, hy = P(x + 11, y + 11)
                d.ellipse((hx - 4 * sc, hy - 5 * sc, hx + 4 * sc, hy + 5 * sc), fill=(10, 10, 5))
                d.rectangle((*P(x + 5, y + 20), *P(x + 17, y + 34)), fill=(10, 10, 5))
            else:
                d.rounded_rectangle(box, radius=max(1, int(4 * sc)), fill=(255, 255, 255))
    d.text(P(248, 105), "CFGA", font=F("sans", 70 * sc), fill=(25, 25, 25), anchor="mm")
    d.text(P(185, 322), "DE LA JONQUIÈRE", font=F("inter", 34 * sc), fill=(40, 40, 40), anchor="mm")


def _strip_rect(x):
    return (x, 196, x + 40, 446)


def arrivee_layers():
    rng = np.random.default_rng(3)
    # ciel de jour (comme sur la photo) et ciel « d'après »
    day = vgrad(W, H, [(0, (70, 120, 205)), (0.45, (140, 180, 230)), (0.8, (200, 215, 235)), (1, (200, 215, 235))])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cl = (np.sin(xx * 0.011 + np.sin(yy * 0.05) * 2) * np.sin(yy * 0.03 + xx * 0.002) > 0.55) & (yy < 300)
    day[cl] = day[cl] * 0.4 + 255 * 0.6
    blood = vgrad(W, H, [(0, (25, 0, 22)), (0.25, (90, 5, 30)), (0.6, (200, 40, 25)), (0.8, (255, 130, 30)),
                         (1, (255, 160, 60))])
    # le décor (bâtiment, pelouse, arbres, lampadaires) sur calque transparent
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    # pelouse et allée
    d.rectangle((0, GROUND - 4, W, H), fill=(92, 140, 60, 255))
    for _ in range(900):
        x, y = rng.uniform(0, W), rng.uniform(GROUND, H)
        d.line([(x, y), (x + rng.uniform(-2, 2), y - rng.uniform(3, 9))], fill=(70, 115, 45, 255), width=1)
    road = [(0, 650), (300, 628), (640, 612), (1000, 604), (1280, 600), (1280, 640), (1000, 646), (640, 656),
            (300, 672), (0, 700)]
    d.polygon(road, fill=(150, 150, 148, 255))
    d.line(road[:5], fill=(190, 190, 185, 255), width=3)
    d.rectangle((0, GROUND - 6, W, GROUND + 2), fill=(170, 168, 160, 255))
    # aile basse en pierre à gauche
    d.rectangle((120, 400, BX0 + 40, GROUND), fill=(176, 160, 138, 255))
    for y in range(404, GROUND, 14):
        d.line([(120, y), (BX0 + 40, y)], fill=(150, 135, 115, 255), width=1)
    d.rectangle((110, 392, BX0 + 50, 404), fill=(205, 205, 200, 255))
    for x in (140, 190):
        d.rectangle((x, 440, x + 34, 520), fill=(35, 40, 50, 255))
    # rez-de-chaussée en retrait : le stationnement intérieur
    d.rectangle((BX0 + 40, BMID, BX1, GROUND), fill=(28, 28, 34, 255))
    d.rectangle((640, 488, 740, GROUND - 6), fill=(60, 70, 80, 255))
    for x, y, s in PARKED:
        w, h = 150 * s, 62 * s
        d.rounded_rectangle((x - w / 2, y - h / 2, x + w / 2, y + h / 2), radius=int(14 * s), fill=(150, 145, 135, 255))
        d.polygon([(x - w * 0.35, y - h / 2), (x + w * 0.35, y - h / 2), (x + w * 0.26, y - h * 1.15),
                   (x - w * 0.26, y - h * 1.15)], fill=(30, 30, 40, 255))
    # les étages : façade crème en panneaux horizontaux
    d.rectangle((BX0, BTOP, BX1, BMID), fill=(232, 228, 214, 255))
    for y in range(BTOP + 30, BMID, 38):
        d.line([(BX0, y), (BX1, y)], fill=(205, 200, 186, 255), width=2)
    d.rectangle((BX0 - 6, BTOP - 14, BX1 + 6, BTOP + 2), fill=(165, 165, 160, 255))
    d.rectangle((BX0 + 40, BMID - 4, BX1, BMID + 22), fill=(215, 210, 196, 255))   # avant-toit
    lay_arr = np.asarray(lay, np.float32).copy()
    sh = np.linspace(1.05, 0.88, BX1 - BX0)[None, :, None]   # lumière rasante
    lay_arr[BTOP:BMID, BX0:BX1, :3] *= sh
    lay = Image.fromarray(np.clip(lay_arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(lay)
    for x in STRIPS:
        x0, y0, x1, y1 = _strip_rect(x)
        d.rectangle((x0, y0, x1, y1), fill=(42, 52, 66, 255))
        for y in range(y0 + 62, y1, 62):
            d.line([(x0, y), (x1, y)], fill=(120, 125, 130, 255), width=3)
        d.polygon([(x0 + 4, y0 + 4), (x0 + 18, y0 + 4), (x0 + 4, y0 + 80)], fill=(90, 110, 140, 255))
    # enseigne blanche sous l'avant-toit, avec le logo
    d.rectangle((560, BMID - 2, 720, BMID + 40), fill=(245, 245, 245, 255), outline=(120, 120, 120, 255), width=2)
    draw_logo(lay, 588, BMID + 19, 40)
    d = ImageDraw.Draw(lay)
    d.text((660, BMID + 13), "CFGA", font=F("sans", 16), fill=(25, 25, 25, 255), anchor="mm")
    d.text((660, BMID + 29), "de la Jonquière", font=F("inter", 11), fill=(40, 40, 40, 255), anchor="mm")
    # arbres
    for cx, cy, r in ((70, 330, 150), (170, 260, 110), (40, 470, 120), (1240, 360, 120), (1290, 260, 110)):
        for k in range(26):
            a = rng.uniform(0, 2 * math.pi)
            rr = rng.uniform(0, r)
            px, py = cx + math.cos(a) * rr, cy + math.sin(a) * rr * 0.9
            s = rng.uniform(25, 55)
            g = int(rng.uniform(60, 110))
            d.ellipse((px - s, py - s, px + s, py + s), fill=(30, g, 30, 255))
    d.rectangle((60, 420, 78, GROUND + 10), fill=(70, 50, 35, 255))
    d.rectangle((1235, 430, 1252, GROUND + 10), fill=(70, 50, 35, 255))
    return day, blood, lay


def lamp(d, x, top, col_head):
    d.rectangle((x - 4, top, x + 4, 610), fill=(120, 120, 125))
    d.line([(x, top), (x + 46, top - 10)], fill=(120, 120, 125), width=5)
    d.ellipse((x + 34, top - 18, x + 64, top - 2), fill=col_head)


def s_arrivee(t, fi, rng):
    if t < PHOTO_END:
        # la vraie photo, comme une image de caméra de surveillance
        def photo():
            p = Image.open(PHOTO).convert("RGB")
            w, h = p.size
            ch = w * 9 / 16
            p = p.crop((0, (h - ch) / 2, w, (h + ch) / 2)).resize((W, H), Image.LANCZOS)
            return p.filter(ImageFilter.UnsharpMask(3, 120, 2))
        p = cached("photo", photo)
        s = 1 + 0.05 * t / PHOTO_END
        cw, ch = W / s, H / s
        return p.crop((int(640 - cw / 2), int(360 - ch / 2), int(640 + cw / 2), int(360 + ch / 2))).resize((W, H), Image.BILINEAR)

    day, blood, lay = cached("arr_layers", arrivee_layers)
    k = clamp((t - 2.6) / 2.6)          # 0 = jour ordinaire, 1 = l'horreur
    sky = day * (1 - k) + blood * k
    img = to_img(sky)
    d = ImageDraw.Draw(img)
    lid = blink(t, [4.6, 6.2])
    # le soleil devient un œil
    if k > 0.2:
        r = int(30 + 42 * k)
        sun = eye_img(r, (-14 + 6 * math.sin(t), 10), max(lid, 1 - clamp(k * 1.6)))
        img.paste(sun, (980 - r, 95 - r), sun)
    img.paste(lay, (0, 0), lay)
    d = ImageDraw.Draw(img)
    # nuages qui défilent (jour) / panache rouge (après)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    do = ImageDraw.Draw(ov)
    for i in range(6):
        y = 50 + i * 22
        x = (i * 300 + t * (20 + 8 * i)) % (W + 600) - 300
        c = mix((255, 255, 255), (40, 0, 20), k)
        do.ellipse((x, y, x + 380, y + 24), fill=c + (90,))
    img.paste(ov, (0, 0), ov)
    d = ImageDraw.Draw(img)
    # dans chaque fenêtre verticale, des visages empilés apparaissent
    nf = int(clamp((t - 3.0) / 2.0) * 4)
    for i, x in enumerate(STRIPS):
        x0, y0, x1, y1 = _strip_rect(x)
        for j in range(nf):
            fy = y0 + 30 + j * 62
            fx = (x0 + x1) / 2 + (4 if t > 5.2 else 0)
            d.ellipse((fx - 11, fy - 14, fx + 11, fy + 14), fill=(225, 215, 195))
            d.ellipse((fx - 7, fy - 5, fx - 2, fy), fill=(0, 0, 0))
            d.ellipse((fx + 2, fy - 5, fx + 7, fy), fill=(0, 0, 0))
        if lid > 0:
            d.rectangle((x0, y0, x1, y0 + (y1 - y0) * lid), fill=(10, 0, 4))
    # le stationnement : phares entre les piliers (rouges à la fin)
    if t > 2.2:
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        do = ImageDraw.Draw(ov)
        hl = (255, 240, 150) if t < 5.0 else (255, 20, 20)
        for x, y, sc in PARKED:
            for sg in (-1, 1):
                cx, cy = x + sg * 52 * sc, y - 2
                for rr, al in ((34 * sc, 50), (18 * sc, 110), (8 * sc, 255)):
                    do.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=hl + (al,))
        img.paste(ov, (0, 0), ov)
        d = ImageDraw.Draw(img)
    for x in RPIL:
        d.rectangle((x - 10, BMID + 22, x + 10, GROUND), fill=(205, 200, 190))
        d.rectangle((x + 4, BMID + 22, x + 10, GROUND), fill=(160, 155, 145))
    # porte d'entrée : la lueur rouge respire
    if k > 0.3:
        p = heart_pulse(t / (7.0 / 6.0) + 4.5)
        d.rectangle((650, 492, 730, GROUND - 8), fill=(int(120 + 135 * p), 0, 10))
    # lampadaires (ils penchent un peu plus à chaque clignement)
    head = mix((235, 235, 230), (255, 20, 20), k)
    lamp(d, 470, 150, head)
    lamp(d, 1010, 150, head)
    # la passante : immobile... puis de plus en plus près
    steps = [(0.0, 905, 600, 46), (4.6, 820, 640, 90), (6.2, 700, 700, 170)]
    px, py, ph = steps[0][1:]
    for st, x_, y_, h_ in steps:
        if t >= st:
            px, py, ph = x_, y_, h_
    silhouette(d, px, py, ph, col=mix((40, 40, 60), (3, 2, 4), k), arms=1.0 + 0.4 * k)
    if t > 4.6:
        for sg in (-1, 1):
            ex, ey = px + sg * ph * 0.03, py - ph + ph * 0.07
            d.ellipse((ex - 2, ey - 2, ex + 2, ey + 2), fill=(255, 255, 255))
    # la neige tombe vers le haut
    if k > 0:
        prng = np.random.default_rng(11)
        n = int(420 * k)
        pxs, pys = prng.random(420) * W, prng.random(420) * H
        sp, sz = 30 + prng.random(420) * 90, 1 + prng.random(420) * 2.5
        for i in range(n):
            y = (pys[i] - sp[i] * t) % H
            x = pxs[i] + 12 * math.sin(t * 1.5 + i)
            d.ellipse((x - sz[i], y - sz[i], x + sz[i], y + sz[i]), fill=(255, 235, 235))
    # assombrissement / virage au rouge de tout le décor
    if k > 0:
        a = np.asarray(img, np.float32)
        a = a * (1 - 0.35 * k) * np.array([1.0, 1 - 0.3 * k, 1 - 0.3 * k])
        img = to_img(a)
    s = 1 + 0.1 * clamp((t - PHOTO_END) / 5.5)
    cw, ch = W / s, H / s
    cx, cy = 640 + rng.normal(0, 1.2 * k), 380 + rng.normal(0, 1.2 * k)
    return img.crop((int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2))).resize((W, H), Image.BILINEAR)


# --------------------------------------------------------------- MURALE ----
# Vraies images (vidéo IMG_8538 fournie) : la murale « JE LAISSE MA TRACE ».
MUR_DIR = os.path.join(HERE, "build", "murale")
MUR_WIN_X = [392, 422, 453, 485, 516, 546, 576, 608, 641, 673, 705]
MUR_WIN_Y = [(485, 517), (522, 550)]
MUR_GAPS = [(382, 450), (467, 535), (552, 622), (640, 710)]


def mur_offsets():
    o = np.load(os.path.join(MUR_DIR, "offs.npy")).astype(np.float32)
    k = np.ones(7) / 7
    pad = np.pad(o, ((3, 3), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, "valid") for i in range(2)], 1)


def s_murale(t, fi, rng):
    files = cached("mur_files", lambda: sorted(f for f in os.listdir(MUR_DIR) if f.endswith(".png")))
    i = min(len(files) - 1, int(t * 24))
    img = Image.open(os.path.join(MUR_DIR, files[i])).convert("RGB")
    ox, oy = cached("mur_offs", mur_offsets)[i]
    a = np.asarray(img, np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    green = (g - b > 50) & (g >= r - 5) & (g > 90)
    # étalonnage : néons verdâtres, noirs écrasés, murs blancs éteints
    a = np.clip((a - 30) * 1.1, 0, None) * 0.72
    a = a * np.array([0.85, 1.0, 0.9])
    # les lettres vertes pourrissent en rouge sang
    rot = clamp((t - MURALE_ROT) / 1.2) * (0.6 + 0.4 * heart_pulse(ev_abs("murale", t)))
    if rot > 0:
        lum = a[green].mean(-1, keepdims=True)
        a[green] = a[green] * (1 - rot) + lum * np.array([1.9, 0.12, 0.18]) * rot
    img = to_img(a)
    d = ImageDraw.Draw(img)
    # les fenêtres du bâtiment peint s'allument... et quelqu'un y attend
    if t >= MURALE_LIGHTS:
        lid = blink(t, [2.2, 3.5], 0.3)
        for y0, y1 in MUR_WIN_Y:
            for x in MUR_WIN_X:
                x0, x1 = x - 8 + ox, x + 8 + ox
                yy0, yy1 = y0 + oy, y1 + oy
                d.rectangle((x0, yy0, x1, yy1), fill=(230, 255, 90))
                cx = (x0 + x1) / 2
                d.ellipse((cx - 4, yy0 + 6, cx + 4, yy0 + 15), fill=(10, 10, 5))
                d.rectangle((cx - 6, yy0 + 17, cx + 6, yy1), fill=(10, 10, 5))
                if lid > 0:
                    d.rectangle((x0, yy0, x1, yy0 + (yy1 - yy0) * lid), fill=(10, 0, 4))
        # des phares s'allument dans le stationnement sous le bâtiment
        for k, (g0, g1) in enumerate(MUR_GAPS):
            if t > MURALE_LIGHTS + 0.3 + k * 0.25:
                cx, cy = (g0 + g1) / 2 + ox, 588 + oy
                col = (255, 240, 150) if t < MURALE_ROT else (255, 30, 30)
                for sg in (-1, 1):
                    d.ellipse((cx + sg * 13 - 5, cy - 4, cx + sg * 13 + 5, cy + 4), fill=col)
    # SORTIE -> ENTRÉE
    if t > MURALE_EXIT or (t > MURALE_EXIT - 0.4 and rng.random() < 0.5):
        d.rectangle((0 + ox, 58 + oy, 98 + ox, 96 + oy), fill=(15, 5, 5))
        d.text((49 + ox, 77 + oy), "ENTRÉE", font=F("anton", 30), fill=(255, 40, 30), anchor="mm")
    # une nouvelle « trace » s'écrit toute seule
    msg = "... et elle me garde."
    n = int(clamp((t - 2.6) / 1.3) * len(msg))
    if n > 0:
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((150 + ox, 440 + oy), msg[:n], font=F("hand", 76), fill=(255, 235, 90, 255),
                                   anchor="lm")
        layer = layer.rotate(3, center=(400, 440), resample=Image.BICUBIC)
        img.paste(layer, (0, 0), layer)
    s = 1 + 0.06 * t / 4.5
    cw, ch = W / s, H / s
    return img.crop((0, int(380 - ch / 2), int(cw), int(380 + ch / 2))).resize((W, H), Image.BILINEAR)


def s_pause2(t, fi, rng):
    # 10 h 45 : exactement la même pause qu'à 9 h 25
    return s_machine(3.6 + t * 0.4, fi, rng)


# ------------------------------------------------------------- CORRIDOR ----
CF, CCX, CCY = 520, 640, 330
XL, XR, YF, YC, YL = -1.6, 1.6, -1.2, 1.4, 0.9


def proj(X, Y, z):
    return (CCX + X * CF / z, CCY - Y * CF / z)


def fog(c, z):
    k = math.exp(-z / 14)
    return tuple(int(v * k + f * (1 - k)) for v, f in zip(c, (6, 14, 10)))


def hsh(*a):
    h = 2166136261
    for v in a:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return h / 0xFFFFFFFF


def banner_img(text, bad):
    b = Image.new("RGB", (1200, 220), (240, 220, 40) if not bad else (150, 0, 0))
    d = ImageDraw.Draw(b)
    d.rectangle((0, 0, 1199, 219), outline=(20, 20, 20), width=10)
    d.text((600, 110), text, font=F("anton", 96 if not bad else 92),
           fill=(30, 30, 120) if not bad else (255, 230, 220), anchor="mm")
    return b


def s_corridor(t, fi, rng):
    for b in CORR_BLACKOUTS:
        if b <= t < b + CORR_BLACK_LEN:
            img = Image.new("RGB", (W, H), (0, 0, 0))
            d = ImageDraw.Draw(img)
            k = CORR_BLACKOUTS.index(b) + 1
            z = CORR_FIG_Z[k]
            ex, ey = proj(0, 0.95, z)
            r = max(2, 0.03 * CF / z)
            for s in (-1, 1):
                d.ellipse((ex + s * 2.2 * r - r, ey - r, ex + s * 2.2 * r + r, ey + r), fill=(255, 255, 255))
            return img
    fz = CORR_FIG_Z[sum(1 for b in CORR_BLACKOUTS if t >= b)]
    o = t * 1.6
    fr = o % 1
    base_n = int(o)
    img = Image.new("RGB", (W, H), (6, 14, 10))
    d = ImageDraw.Draw(img)
    banner_z = 7.0 - o
    fig_drawn = False
    for i in range(46, -1, -1):
        zn = max(0.35, i + 0.5 - fr)
        zf = i + 1.5 - fr
        if zf <= 0.36:
            continue
        n = i + base_n
        if not fig_drawn and zn < fz:
            draw_fig(d, fz, t)
            fig_drawn = True
        # plancher : linoléum en damier
        lanes = 4
        for l in range(lanes):
            x0 = XL + (XR - XL) * l / lanes
            x1 = XL + (XR - XL) * (l + 1) / lanes
            c = (205, 195, 150) if (l + n) % 2 == 0 else (55, 115, 70)
            d.polygon([proj(x0, YF, zn), proj(x1, YF, zn), proj(x1, YF, zf), proj(x0, YF, zf)], fill=fog(c, zn))
        # plafond + néons
        d.polygon([proj(XL, YC, zn), proj(XR, YC, zn), proj(XR, YC, zf), proj(XL, YC, zf)], fill=fog((45, 48, 40), zn))
        if n % 3 == 0:
            on = hsh(n, int(t * 14)) > 0.18 and hsh(n, 7) > 0.15
            c = (235, 255, 225) if on else (40, 50, 40)
            za, zb = max(0.35, zn + 0.15), zn + 0.85
            d.polygon([proj(-0.5, YC, za), proj(0.5, YC, za), proj(0.5, YC, zb), proj(-0.5, YC, zb)], fill=fog(c, zn * 0.6))
        # murs
        for side, X in ((-1, XL), (1, XR)):
            door = side == 1 and n % 6 == 3
            d.polygon([proj(X, YL, zn), proj(X, YC, zn), proj(X, YC, zf), proj(X, YL, zf)], fill=fog((150, 160, 115), zn))
            if door:
                d.polygon([proj(X, YF, zn), proj(X, YL, zn), proj(X, YL, zf), proj(X, YF, zf)], fill=fog((130, 125, 110), zn))
                d.polygon([proj(X, YF, zn + 0.15), proj(X, 0.7, zn + 0.15), proj(X, 0.7, zf - 0.15), proj(X, YF, zf - 0.15)],
                          fill=fog((95, 50, 28), zn))
                d.polygon([proj(X, 0.15, zn + 0.3), proj(X, 0.55, zn + 0.3), proj(X, 0.55, zn + 0.6), proj(X, 0.15, zn + 0.6)],
                          fill=(0, 0, 0))
                sc = CF / (zn + 0.5)
                if sc > 25:
                    px, py = proj(X, 0.82, zn + 0.5)
                    d.text((px, py), "B-104", font=F("vt", 0.22 * sc), fill=fog((255, 255, 255), zn), anchor="mm")
            else:
                c = ((225, 95, 30) if n % 2 else (200, 78, 22)) if side < 0 else ((20, 135, 135) if n % 2 else (12, 108, 112))
                d.polygon([proj(X, YF, zn), proj(X, YL, zn), proj(X, YL, zf), proj(X, YF, zf)], fill=fog(c, zn))
                for k in range(3):
                    zz = zn + 0.2 + k * 0.08
                    d.line([proj(X, 0.55, zz), proj(X, 0.75, zz)], fill=fog((30, 20, 10), zn), width=max(1, int(CF / zz / 120)))
                d.line([proj(X, YF, zn), proj(X, YL, zn)], fill=fog((20, 15, 10), zn), width=max(1, int(CF / zn / 60)))
                hz = zf - 0.12
                d.rectangle((*proj(X, 0.0, hz), proj(X, -0.12, hz)[0] + max(1, CF / hz / 50), proj(X, -0.12, hz)[1]),
                            fill=fog((180, 180, 180), zn))
        # banderole
        if i + 0.5 - fr <= banner_z < i + 1.5 - fr and banner_z > 0.45:
            bad = t > 2.6
            key = ("ban", bad)
            ban = cached(key, lambda: banner_img("PERSÉVÈRE. RESTE. PERSÉVÈRE." if bad else
                                                 "SEMAINE DE LA PERSÉVÉRANCE SCOLAIRE", bad))
            (x0, y0), (x1, y1) = proj(-1.25, 1.25, banner_z), proj(1.25, 0.72, banner_z)
            if x1 - x0 > 4:
                bi = ban.resize((int(x1 - x0), int(y1 - y0)), Image.BILINEAR)
                img.paste(bi, (int(x0), int(y0)))
                d = ImageDraw.Draw(img)
    if not fig_drawn:
        draw_fig(d, fz, t)
    return img


def draw_fig(d, z, t):
    if z < 0.4:
        return
    cx, base = proj(0.15 * math.sin(t * 0.7), YF, z)
    _, top = proj(0, 1.25, z)
    silhouette(d, cx, base, base - top, col=(2, 2, 3), arms=1.3)
    ex, ey = proj(0.15 * math.sin(t * 0.7), 1.1, z)
    r = max(1.5, 0.025 * CF / z)
    for s in (-1, 1):
        d.ellipse((ex + s * 2.2 * r - r, ey - r, ex + s * 2.2 * r + r, ey + r), fill=(255, 250, 230))


# ---------------------------------------------------------------- MATHS ----
BOARD = (170, 82, 1110, 372)
CHALK = [
    (0.0, (210, 112), "Mathématiques — séance 1 de ∞"),
    (1.2, (210, 178), "2 + 2 = 5"),
    (2.0, (560, 178), "x + 1 = x"),
    (2.7, (210, 244), "Combien de temps reste-t-il ?"),
    (4.0, (210, 308), "x = toujours"),
]


def maths_static():
    a = vgrad(W, H, [(0, (120, 140, 95)), (0.7, (150, 165, 115)), (0.72, (80, 62, 45)), (1, (50, 38, 28))])
    img = to_img(a)
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(5)
    for k in range(-8, 9):
        d.line([(640 + k * 30, 518), (640 + k * 220, H)], fill=(60, 45, 32), width=2)
    d.rectangle((BOARD[0] - 16, BOARD[1] - 16, BOARD[2] + 16, BOARD[3] + 16), fill=(110, 72, 38))
    d.rectangle(BOARD, fill=(28, 62, 42))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    do = ImageDraw.Draw(ov)
    for _ in range(40):
        x = rng.uniform(BOARD[0], BOARD[2] - 200)
        y = rng.uniform(BOARD[1], BOARD[3] - 40)
        do.ellipse((x, y, x + rng.uniform(60, 260), y + rng.uniform(10, 40)), fill=(220, 230, 220, int(rng.uniform(8, 25))))
    img.paste(ov, (0, 0), ov)
    d = ImageDraw.Draw(img)
    d.rectangle((BOARD[0] - 16, BOARD[3] + 12, BOARD[2] + 16, BOARD[3] + 26), fill=(95, 60, 30))
    # affiche
    d.rectangle((20, 120, 150, 330), fill=(240, 235, 210))
    d.text((85, 170), "LES", font=F("anton", 26), fill=(200, 30, 30), anchor="mm")
    d.text((85, 205), "MATHS", font=F("anton", 30), fill=(30, 30, 140), anchor="mm")
    d.text((85, 245), "C'EST", font=F("anton", 26), fill=(200, 30, 30), anchor="mm")
    d.text((85, 285), "POUR LA VIE", font=F("anton", 20), fill=(30, 30, 140), anchor="mm")
    return img


def chalk_progress(t):
    """Retourne (lignes révélées, position de la craie)."""
    out, pos = [], None
    for t0, (x, y), txt in CHALK:
        if t < t0:
            break
        n = min(len(txt), int((t - t0) * 16))
        out.append(((x, y), txt[:n]))
        if n < len(txt):
            pos = (x + F("hand", 56).getlength(txt[:n]), y + 18)
    return out, pos


def s_maths(t, fi, rng):
    img = cached("maths_static", maths_static).copy()
    d = ImageDraw.Draw(img)
    lines, pos = chalk_progress(t)
    for (x, y), txt in lines:
        d.text((x, y), txt, font=F("hand", 56), fill=(238, 240, 228), anchor="lm")
    if t > 4.6:
        prng = np.random.default_rng(42)
        n = int((t - 4.6) * 16)
        for k in range(min(n, 60)):
            x = prng.uniform(BOARD[0] + 40, BOARD[2] - 60)
            y = prng.uniform(BOARD[1] + 20, BOARD[3] - 20)
            paste_rot_text(img, (x, y), "RESTE", F("hand", int(prng.uniform(34, 70))), (230, 30, 30), prng.uniform(-25, 25))
        d = ImageDraw.Draw(img)
    # horloge qui recule
    cx, cy, r = 640, 40, 30
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(240, 235, 220), outline=(20, 20, 20), width=4)
    for ang, ln, wd in ((-t * 9, 0.85, 2), (-t * 0.9, 0.55, 4)):
        d.line([(cx, cy), (cx + math.sin(ang) * r * ln, cy - math.cos(ang) * r * ln)], fill=(20, 20, 20), width=wd)
    # l'enseignant, beaucoup trop grand
    tx, base = 1055, 560
    silhouette(d, tx, base, 520, col=(8, 6, 10), arms=0.9, head_dx=-6)
    sh = (tx - 25, 160)
    tip = pos if pos else (950 + 10 * math.sin(t * 3), 200 + 8 * math.cos(t * 2))
    el = ((sh[0] + tip[0]) / 2 + 30, max(sh[1], tip[1]) + 70)
    d.line([sh, el, tip], fill=(8, 6, 10), width=14, joint="curve")
    for k in range(4):
        d.line([tip, (tip[0] - 18 + k * 9, tip[1] + 30)], fill=(8, 6, 10), width=4)
    # les élèves
    turned = t >= MATHS_TURN
    breath = 1 + 0.025 * math.sin(2 * math.pi * 0.45 * t)
    jit = math.sin(t * 40) * 2 if turned else 0
    shirts = [(70, 80, 110), (110, 70, 70), (70, 100, 75), (120, 110, 80), (90, 70, 110)]
    for row, (yb, s, sp) in enumerate(((505, 0.62, 175), (585, 0.85, 235), (705, 1.22, 330))):
        for c in range(5):
            x = 640 + (c - 2) * sp
            d.rectangle((x - 95 * s, yb - 18 * s, x + 95 * s, yb + 30 * s), fill=(150, 110, 60))
            sw, shh = 70 * s, 85 * s * breath
            d.rounded_rectangle((x - sw, yb - shh, x + sw, yb + 10), radius=int(30 * s), fill=shirts[(c + row) % 5])
            hr = 33 * s
            hx, hy = x + jit * s, yb - shh - hr * 0.9
            if not turned:
                d.ellipse((hx - hr, hy - hr * 1.1, hx + hr, hy + hr * 1.1), fill=(35, 22, 15))
            else:
                d.ellipse((hx - hr, hy - hr * 1.15, hx + hr, hy + hr * 1.1), fill=(232, 222, 205))
                for sg in (-1, 1):
                    d.ellipse((hx + sg * hr * 0.4 - hr * 0.2, hy - hr * 0.35, hx + sg * hr * 0.4 + hr * 0.2, hy + hr * 0.15),
                              fill=(0, 0, 0))
                d.ellipse((hx - hr * 0.12, hy + hr * 0.45, hx + hr * 0.12, hy + hr * 0.75), fill=(60, 0, 0))
    if MATHS_TURN - 0.1 <= t < MATHS_TURN:
        return Image.new("RGB", (W, H), (0, 0, 0))
    s = 1 + 0.05 * t / 7
    cw, ch = W / s, H / s
    return img.crop((int(640 - cw / 2), int(330 - ch / 2), int(640 + cw / 2), int(330 + ch / 2))).resize((W, H), Image.BILINEAR)


# ---------------------------------------------------------------- DÎNER ----
MW_POS = [(60 + i * 300, y) for y in (110, 330) for i in range(4)]
WORDS = ["FAIM", "RESTE", "MANGE", "ENCORE", "TOI", "FAIM", "RESTE", "ICI"]


def diner_static():
    img = Image.new("RGB", (W, H), (205, 210, 195))
    d = ImageDraw.Draw(img)
    for x in range(0, W, 40):
        d.line([(x, 0), (x, 520)], fill=(150, 155, 140), width=2)
    for y in range(0, 520, 40):
        d.line([(0, y), (W, y)], fill=(150, 155, 140), width=2)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    do = ImageDraw.Draw(ov)
    rng = np.random.default_rng(8)
    for _ in range(30):
        x, y = rng.uniform(0, W), rng.uniform(0, 500)
        r = rng.uniform(10, 60)
        do.ellipse((x - r, y - r * 0.6, x + r, y + r * 1.5), fill=(110, 80, 30, int(rng.uniform(15, 50))))
    img.paste(ov, (0, 0), ov)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 300, W, 322), fill=(130, 130, 125))
    d.rectangle((0, 505, W, 530), fill=(150, 150, 145))
    d.rectangle((0, 530, W, H), fill=(45, 45, 50))
    return img


def s_diner(t, fi, rng):
    img = cached("diner_static", diner_static).copy()
    d = ImageDraw.Draw(img)
    phase3 = t >= DINER_LONG
    p = heart_pulse(t + 25.5)
    for i, (x, y) in enumerate(MW_POS):
        d.rounded_rectangle((x, y, x + 260, y + 170), radius=10, fill=(205, 205, 200), outline=(90, 90, 90), width=3)
        win = (x + 14, y + 18, x + 186, y + 152)
        inside = (255, 150, 40) if not phase3 else (int(150 + 100 * p), 0, 0)
        d.rectangle(win, fill=inside)
        cx, cy = (win[0] + win[2]) / 2, win[3] - 32
        d.ellipse((cx - 70, cy - 12, cx + 70, cy + 12), fill=(235, 200, 140) if not phase3 else (90, 0, 0))
        ang = t * 2.2 + i
        ix = cx + 40 * math.cos(ang)
        if not phase3:
            for k, c in enumerate(((245, 215, 60), (120, 70, 40), (230, 220, 190))):
                d.rectangle((ix - 26, cy - 34 + k * 10, ix + 26, cy - 24 + k * 10), fill=c)
        else:
            r = 22 + 10 * p
            d.ellipse((ix - r, cy - 20 - r, ix + r, cy - 20 + r * 0.8), fill=(140, 0, 20))
            d.ellipse((ix - r * 0.5, cy - 20 - r * 1.1, ix + r * 0.4, cy - 20), fill=(170, 10, 30))
            for k in range(3):
                dx = win[0] + 20 + k * 55
                ln = clamp((t - DINER_LONG) * 0.8) * (60 + 20 * k)
                d.line([(dx, win[1]), (dx, win[1] + ln)], fill=(60, 0, 0), width=6)
        d.rectangle(win, outline=(40, 40, 40), width=4)
        d.rectangle((x + 192, y + 18, x + 254, y + 52), fill=(10, 20, 10))
        if t < DINER_BEEPS[0]:
            txt = "12:00" if int(t * 3) % 2 == 0 else ""
            col = (60, 255, 90)
        elif not phase3:
            v = 3 - int((t - DINER_BEEPS[0]) / (DINER_BEEPS[1] - DINER_BEEPS[0]))
            txt, col = f"0:0{max(0, v)}", (60, 255, 90)
        else:
            txt, col = WORDS[i], (255, 30, 30)
        d.text((x + 223, y + 36), txt, font=F("vt", 30 if len(txt) < 6 else 24), fill=col, anchor="mm")
        for r_ in range(4):
            for c_ in range(3):
                bx, by = x + 200 + c_ * 16, y + 62 + r_ * 22
                d.rectangle((bx, by, bx + 12, by + 14), fill=(170, 170, 165))
    # affichette
    vip_sign(d, t, 330, 55, w=300)
    paste_rot_text(img, (900, 62), "UN PLAT À LA FOIS S.V.P. :)",
                   F("hand", 44), (20, 20, 20, 255), 2)
    d = ImageDraw.Draw(img)
    if phase3:
        for k in range(5):
            ln = clamp((t - DINER_LONG) * 0.7) * (40 + 15 * k)
            x = 1040 + k * 9
            d.line([(x, 75), (x, 75 + ln)], fill=(150, 0, 0), width=3)
    # la file d'attente : des dos qui se balancent ensemble
    sway = 14 * math.sin(t * 1.4)
    for k, x in enumerate((180, 470, 800, 1110)):
        s = 1.0 + 0.1 * (k % 2)
        hx = x + sway
        d.ellipse((hx - 60 * s, 520 - 70 * s, hx + 60 * s, 520 + 60 * s), fill=(14, 10, 12))
        d.rounded_rectangle((x - 150 * s + sway * 0.5, 600 - 20 * s, x + 150 * s + sway * 0.5, H + 60), radius=60, fill=(14, 10, 12))
        d.arc((hx - 60 * s, 520 - 70 * s, hx + 60 * s, 520 + 60 * s), 200, 340, fill=(90, 40, 30), width=3)
    s = 1 + 0.08 * t / 6
    cw, ch = W / s, H / s
    return img.crop((int(640 - cw / 2), int(300 - ch / 2), int(640 + cw / 2), int(300 + ch / 2))).resize((W, H), Image.BILINEAR)


# --------------------------------------------------------------- EXAMEN ----
PW, PH = 780, 1000
EXAM_WRITE = [  # (t0, x, y, texte, police, couleur)
    (0.3, 110, 205, "je ne partirai pas", "hand", (70, 70, 85)),
    (1.0, 110, 250, "tu ne partiras pas", "hand", (70, 70, 85)),
    (1.7, 110, 295, "nous ne partirons jamais", "hand", (70, 70, 85)),
    (3.4, 390, 700, "Toi", "hand", (40, 40, 60)),
    (4.3, 110, 845, "il n'y en a pas", "hand", (170, 0, 10)),
]
OPTIONS = ["une session", "trois ans", "depuis 1974", "vous avez toujours été ici"]


def wood_static():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = np.sin(xx * 0.012 + 3 * np.sin(yy * 0.004) + 0.8 * np.sin(yy * 0.031 + xx * 0.002))
    v = (v + 1) / 2
    a = np.stack([90 + 70 * v, 50 + 40 * v, 25 + 20 * v], -1)
    return to_img(a)


def s_examen(t, fi, rng):
    paper = Image.new("RGB", (PW, PH), (246, 242, 228))
    d = ImageDraw.Draw(paper)
    for y in range(140, PH, 45):
        d.line([(0, y), (PW, y)], fill=(180, 200, 230), width=1)
    d.line([(80, 0), (80, PH)], fill=(230, 120, 120), width=2)
    d.text((100, 50), "ÉPREUVE FINALE — FRANÇAIS, 5e secondaire", font=F("type", 28), fill=(20, 20, 20), anchor="lm")
    d.text((100, 95), "Nom :", font=F("type", 22), fill=(20, 20, 20), anchor="lm")
    d.text((170, 95), "????", font=F("hand", 40), fill=(70, 70, 85), anchor="lm")
    d.text((PW - 40, 95), "Durée : illimitée", font=F("type", 22), fill=(20, 20, 20), anchor="rm")
    d.text((100, 160), "1. Conjuguez le verbe « partir » au futur simple.", font=F("type", 22), fill=(20, 20, 20), anchor="lm")
    d.text((100, 360), "2. Depuis combien de temps êtes-vous ici ?", font=F("type", 22), fill=(20, 20, 20), anchor="lm")
    for k, opt in enumerate(OPTIONS):
        y = 410 + k * 46
        d.ellipse((115, y - 12, 139, y + 12), outline=(20, 20, 20), width=2)
        if t > 2.0 + k * 0.35:
            d.ellipse((118, y - 9, 136, y + 9), fill=(50, 50, 60))
        d.text((155, y), opt, font=F("type", 22), fill=(20, 20, 20), anchor="lm")
    if t > 3.6:
        prng = np.random.default_rng(9)
        n = int((t - 3.6) * 25)
        pts = [(prng.uniform(110, 600), prng.uniform(395, 560)) for _ in range(min(n, 80) + 2)]
        d.line(pts, fill=(60, 60, 70), width=3)
    d.text((100, 640), "3. Signez ici pour confirmer votre présence :", font=F("type", 22), fill=(20, 20, 20), anchor="lm")
    d.line([(380, 725), (700, 725)], fill=(20, 20, 20), width=2)
    d.text((100, 790), "4. Question bonus : où est la sortie ?", font=F("type", 22), fill=(20, 20, 20), anchor="lm")
    pen = None
    for t0, x, y, txt, fn, col in EXAM_WRITE:
        if t < t0:
            continue
        n = min(len(txt), int((t - t0) * 18))
        d.text((x, y), txt[:n], font=F(fn, 46), fill=col, anchor="lm")
        if n < len(txt):
            pen = (x + F(fn, 46).getlength(txt[:n]), y + 10)
    if t > 4.0:
        n = int((t - 4.0) * 6)
        for k in range(min(n, 8)):
            d.text((470 + k * 42, 700 + (k % 3) * 5), "toi", font=F("hand", 36), fill=(40, 40, 60), anchor="lm")
    # tache d'encre rouge qui s'étend depuis le coin
    if t > 2.5:
        R = (t - 2.5) * 110
        pts = []
        for k in range(48):
            a = math.pi + k / 47 * (math.pi / 2)
            rr = R * (1 + 0.18 * math.sin(k * 1.7 + t) + 0.1 * math.sin(k * 4.3))
            pts.append((PW + math.cos(a) * rr, PH + math.sin(a) * rr))
        pts.append((PW, PH))
        d.polygon(pts, fill=(130, 0, 12))
        for k in range(4):
            x = PW - R * (0.3 + 0.18 * k)
            ln = R * 0.5
            d.line([(x, PH - R * 0.6), (x, PH - R * 0.6 + ln)], fill=(130, 0, 12), width=7)
    if t > EXAM_STAMP:
        k = clamp((t - EXAM_STAMP) / 0.12)
        sc = 2.2 - 1.2 * k
        st = Image.new("RGBA", (560, 150), (0, 0, 0, 0))
        sd = ImageDraw.Draw(st)
        sd.rectangle((5, 5, 555, 145), outline=(210, 0, 20, 230), width=10)
        sd.text((280, 75), "À REPRENDRE", font=F("anton", 92), fill=(210, 0, 20, 230), anchor="mm")
        st = st.resize((int(560 * sc), int(150 * sc))).rotate(14, expand=True, resample=Image.BICUBIC)
        paper.paste(st, (int(390 - st.width / 2), int(560 - st.height / 2)), st)
    # page sur le bureau
    ang = -4 + 1.2 * math.sin(t * 0.6)
    PS = 0.7
    paper = paper.resize((int(PW * PS), int(PH * PS)), Image.LANCZOS)
    pr = paper.convert("RGBA").rotate(ang, expand=True, resample=Image.BICUBIC)
    desk = cached("wood", wood_static).copy()
    sh = Image.new("RGBA", pr.size, (0, 0, 0, 0))
    sh.putalpha(pr.getchannel("A").point(lambda v: v * 0.45))
    ox, oy = 640 - pr.width // 2, 375 - pr.height // 2
    desk.paste(sh, (ox + 25, oy + 25), sh)
    desk.paste(pr, (ox, oy), pr)
    dd = ImageDraw.Draw(desk)
    # crayon (position de l'écriture courante)
    if pen is None:
        pen = (600 + 30 * math.sin(t * 2), 760 + 20 * math.cos(t * 1.5))
    rad = math.radians(-ang)
    px, py = (pen[0] - PW / 2) * PS, (pen[1] - PH / 2) * PS
    sx = 640 + px * math.cos(rad) - py * math.sin(rad) + rng.normal(0, 1)
    sy = 375 + px * math.sin(rad) + py * math.cos(rad) + rng.normal(0, 1)
    dirx, diry = math.cos(math.radians(35)), math.sin(math.radians(35))
    nx, ny = -diry, dirx

    def seg(a, b, w):
        return [(sx + dirx * a + nx * w, sy + diry * a + ny * w), (sx + dirx * b + nx * w, sy + diry * b + ny * w),
                (sx + dirx * b - nx * w, sy + diry * b - ny * w), (sx + dirx * a - nx * w, sy + diry * a - ny * w)]

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).polygon([(x + 30, y + 40) for x, y in seg(0, 360, 14)], fill=(0, 0, 0, 90))
    desk.paste(shadow, (0, 0), shadow)
    dd = ImageDraw.Draw(desk)
    dd.polygon([(sx, sy), seg(40, 40, 13)[0], seg(40, 40, 13)[3]], fill=(230, 190, 130))
    dd.polygon([(sx, sy), seg(12, 12, 4)[0], seg(12, 12, 4)[3]], fill=(40, 40, 40))
    dd.polygon(seg(40, 300, 13), fill=(245, 190, 20))
    dd.polygon(seg(300, 330, 14), fill=(190, 190, 200))
    dd.polygon(seg(330, 365, 13), fill=(240, 130, 150))
    s = 1.0 + 0.22 * t / 6.5
    cw, ch = W / s, H / s
    cx, cy = 640 + 30 * math.sin(t * 0.3), 380
    return desk.crop((int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2))).resize((W, H), Image.BILINEAR)


# -------------------------------------------------------------- MACHINE ----
ITEMS = [
    ("A1", "Croustilles", "1,75 $", (230, 60, 40)),
    ("A2", "Café noir", "1,25 $", (60, 40, 30)),
    ("A3", "Barre tendre", "1,50 $", (220, 180, 40)),
    ("B1", "1 h de sommeil", "9,99 $", (60, 60, 160)),
    ("B2", "Ton diplôme", "ÉPUISÉ", (240, 240, 230)),
    ("B3", "Un week-end", "ÉPUISÉ", (40, 160, 90)),
    ("C1", "Ton nom", "0,25 $", (200, 200, 200)),
    ("C2", "Yeux frais", "2,00 $", None),
    ("C3", "La sortie", "ÉPUISÉ", (40, 200, 60)),
    ("D1", "Encore un lundi", "GRATUIT", (255, 40, 40)),
    ("D2", "Ta mémoire", "0,05 $", (150, 150, 255)),
    ("D3", "Hier", "0,00 $", (120, 120, 120)),
]
GX0, GY0, GX1, GY1 = 410, 120, 770, 620


def s_machine(t, fi, rng):
    def bg():
        a = vgrad(W, H, [(0, (10, 10, 20)), (0.75, (25, 22, 35)), (0.76, (18, 14, 18)), (1, (8, 6, 8))])
        yy, xx = np.mgrid[0:H, 0:W]
        g = np.exp(-(((xx - 640) / 420) ** 2 + ((yy - 700) / 90) ** 2))
        a += g[..., None] * np.array([150, 10, 20])
        return to_img(a)

    img = cached("mach_bg", bg).copy()
    d = ImageDraw.Draw(img)
    vip_sign(d, t, 160, 150)
    d.rounded_rectangle((380, 30, 905, 705), radius=18, fill=(125, 12, 22), outline=(60, 0, 8), width=6)
    head = "CONSOLATIONS" if (int(t * 5) % 7 == 0 or t > 3.8) else "COLLATIONS"
    d.rectangle((400, 45, 885, 105), fill=(20, 0, 5))
    d.text((642, 75), head, font=F("anton", 50), fill=(255, 235, 200), anchor="mm")
    light = 0.75 + 0.25 * (hsh(int(t * 15)) > 0.2)
    gl = vgrad(GX1 - GX0, GY1 - GY0, [(0, (220, 235, 255)), (1, (120, 140, 170))]) * light
    img.paste(to_img(gl), (GX0, GY0))
    d = ImageDraw.Draw(img)
    cw_, rh = (GX1 - GX0) / 3, (GY1 - GY0) / 4
    drop = None
    for idx, (code, name, price, col) in enumerate(ITEMS):
        r, c = divmod(idx, 3)
        x0, y0 = GX0 + c * cw_, GY0 + r * rh
        cx = x0 + cw_ / 2
        # spirale
        phase = 0
        if code == "D1" and MACHINE_DROP[0] <= t:
            phase = clamp((t - MACHINE_DROP[0]) / 0.6) * 2 * math.pi
        pts = [(x0 + 10 + k * 3, y0 + rh - 30 + 6 * math.sin(k * 0.6 + phase)) for k in range(int((cw_ - 20) / 3))]
        d.line(pts, fill=(180, 180, 190), width=3)
        iy = y0 + 12
        if code == "D1" and t > MACHINE_DROP[0] + 0.5:
            ft = t - MACHINE_DROP[0] - 0.5
            iy += 0.5 * 2600 * ft * ft
            if iy > 640:
                iy = None
            drop = True
        if iy is not None:
            if col is None:
                for k in range(2):
                    jx = x0 + 22 + k * 52
                    d.rectangle((jx, iy + 10, jx + 42, iy + 70), fill=(200, 230, 210), outline=(120, 140, 130), width=2)
                    e = eye_img(16, (6 * math.sin(t * 2 + k), 4), blink(t, [1.4, 3.3]) if k else 0)
                    img.paste(e, (int(jx + 5), int(iy + 24)), e)
                d = ImageDraw.Draw(img)
            else:
                d.rectangle((x0 + 14, iy, x0 + cw_ - 14, iy + rh - 50), fill=col, outline=(20, 20, 20), width=2)
                tc = (0, 0, 0) if sum(col) > 400 else (255, 255, 255)
                words = name.split(" ")
                lines = [" ".join(words[:len(words) // 2 + 1]), " ".join(words[len(words) // 2 + 1:])] if len(name) > 11 else [name]
                for k, ln in enumerate(lines):
                    d.text((cx, iy + 22 + k * 18), ln, font=F("sans", 14), fill=tc, anchor="mm")
        d.rectangle((x0 + 6, y0 + rh - 22, x0 + cw_ - 6, y0 + rh - 4), fill=(10, 10, 12))
        pc = (255, 60, 60) if price in ("ÉPUISÉ",) else (120, 255, 140)
        d.text((x0 + 10, y0 + rh - 13), code, font=F("vt", 20), fill=(255, 255, 255), anchor="lm")
        d.text((x0 + cw_ - 10, y0 + rh - 13), price, font=F("vt", 20), fill=pc, anchor="rm")
    # reflets sur la vitre
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    do = ImageDraw.Draw(ov)
    for k, x in enumerate((450, 560, 700)):
        do.polygon([(x, GY0), (x + 40 + k * 10, GY0), (x - 80, GY1), (x - 120 - k * 10, GY1)], fill=(255, 255, 255, 28))
    img.paste(ov, (0, 0), ov)
    d = ImageDraw.Draw(img)
    d.rectangle((GX0, GY0, GX1, GY1), outline=(40, 0, 5), width=6)
    # afficheur DEL défilant
    msg = "  MERCI DE VOTRE PATIENCE ••• IL VOUS RESTE 0 MINUTE ••• VOTRE COURS COMMENCE DANS 0 SECONDE •••"
    strip = cached("led", lambda: _led_strip(msg))
    off = int(t * 140) % (strip.width - 110)
    img.paste(strip.crop((off, 0, off + 110, 40)), (785, 140))
    d.rectangle((783, 138, 897, 182), outline=(60, 60, 60), width=2)
    for r in range(4):
        for c in range(3):
            bx, by = 795 + c * 32, 210 + r * 34
            d.rounded_rectangle((bx, by, bx + 26, by + 26), radius=4, fill=(200, 200, 205))
            d.text((bx + 13, by + 13), "123456789*0#"[r * 3 + c], font=F("vt", 22), fill=(20, 20, 20), anchor="mm")
    d.rectangle((815, 370, 865, 380), fill=(10, 10, 10))
    d.rectangle((800, 400, 880, 430), fill=(40, 40, 40))
    d.rectangle((808, 410, 872, 418), fill=(60, 255, 120) if int(t * 2) % 2 else (20, 80, 40))
    d.rectangle((420, 640, 760, 690), fill=(15, 10, 12))
    if drop and t > MACHINE_DROP[1]:
        d.rectangle((470, 650, 640, 688), fill=(255, 40, 40))
        d.text((555, 669), "LUNDI", font=F("anton", 30), fill=(255, 255, 255), anchor="mm")
    d.rectangle((420, 640, 760, 655), fill=(70, 10, 15))
    s = 1 + 0.12 * t / 5
    cw, ch = W / s, H / s
    return img.crop((int(640 - cw / 2), int(380 - ch / 2), int(640 + cw / 2), int(380 + ch / 2))).resize((W, H), Image.BILINEAR)


def vip_sign(d, t, cx, cy, w=280):
    """Enseigne de la salle V.I.P. — Vous Ici, Pour toujours."""
    bad = hsh(int(t * 6), 3) < 0.18 or (t > 3.0 and hsh(int(t * 4), 9) < 0.5)
    d.rectangle((cx - w / 2, cy - 38, cx + w / 2, cy + 38), fill=(20, 20, 30) if not bad else (90, 0, 10),
                outline=(230, 190, 60), width=4)
    if not bad:
        d.text((cx, cy), "SALLE V.I.P.", font=F("anton", 46), fill=(240, 200, 70), anchor="mm")
    else:
        d.text((cx, cy - 14), "VOUS · ICI ·", font=F("anton", 26), fill=(255, 230, 220), anchor="mm")
        d.text((cx, cy + 15), "POUR TOUJOURS", font=F("anton", 26), fill=(255, 230, 220), anchor="mm")


def _led_strip(msg):
    f = F("vt", 40)
    w = int(f.getlength(msg)) + 120
    s = Image.new("RGB", (w, 40), (10, 0, 0))
    ImageDraw.Draw(s).text((110, 20), msg, font=f, fill=(255, 40, 30), anchor="lm")
    return s


# --------------------------------------------------------------- CLOCHE ----
def blocks(base):
    a = np.zeros((H, W, 3), np.float32) + np.array(base, np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    m = (yy % 60 < 4) | (((xx + (yy // 60 % 2) * 60) % 120) < 4)
    a[m] *= 0.6
    a *= 0.9 + 0.2 * np.random.default_rng(4).random((H, W, 1))
    return to_img(a)


def s_cloche(t, fi, rng):
    if t < 1.9:
        img = cached("blk1", lambda: blocks((185, 180, 160))).copy()
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        do = ImageDraw.Draw(ov)
        for k in range(10):
            r = (t * 700 + k * 90) % 900
            al = int(200 * (1 - r / 900))
            do.ellipse((640 - r, 400 - r, 640 + r, 400 + r), outline=(255, 220, 0, al) if k % 2 else (255, 0, 30, al), width=12)
        img.paste(ov, (0, 0), ov)
        d = ImageDraw.Draw(img)
        sx, sy = rng.normal(0, 9, 2)
        cx, cy = 640 + sx, 400 + sy
        d.rectangle((cx - 30, cy - 160, cx + 30, cy - 110), fill=(60, 60, 60))
        d.ellipse((cx - 130, cy - 130, cx + 130, cy + 130), fill=(190, 20, 25), outline=(80, 0, 0), width=6)
        d.ellipse((cx - 90, cy - 100, cx - 20, cy - 50), fill=(255, 140, 140))
        d.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), fill=(50, 50, 50))
        hx = cx + 120 + rng.normal(0, 14)
        d.line([(cx + 40, cy + 110), (hx, cy + 150)], fill=(40, 40, 40), width=10)
        d.ellipse((hx - 18, cy + 132, hx + 18, cy + 168), fill=(60, 60, 60))
        shaky_text(d, (640, 120), "DRRRRRRRING", F("anton", 130), (255, 230, 0), rng, amp=10,
                   stroke=6, stroke_fill=(200, 0, 0))
        return img
    tt = t - 1.9
    img = cached("blk2", lambda: blocks((45, 75, 60))).copy()
    d = ImageDraw.Draw(img)
    d.rectangle((0, 640, W, H), fill=(25, 30, 28))
    flip = t >= CLOCHE_EXIT_FLIP or (t > CLOCHE_EXIT_FLIP - 0.7 and rng.random() < 0.5)
    sign_col = (255, 30, 30) if flip else (40, 255, 90)
    # porte double
    d.rectangle((390, 195, 890, 645), fill=(30, 30, 35))
    for x0 in (400, 645):
        d.rectangle((x0, 205, x0 + 235, 640), fill=(105, 118, 130), outline=(60, 65, 70), width=4)
        # fenêtres : le stationnement, encore et toujours
        mini = s_arrivee((t * 1.3) % 7, fi, rng).resize((150, 180), Image.BILINEAR)
        img.paste(mini, (x0 + 42, 250))
        d = ImageDraw.Draw(img)
        d.rectangle((x0 + 42, 250, x0 + 192, 430), outline=(40, 40, 45), width=6)
        d.rectangle((x0 + 20, 470, x0 + 215, 492), fill=(170, 175, 180), outline=(60, 60, 60), width=3)
    # chaînes et cadenas
    for k in range(18):
        x = 430 + k * 24
        y = 455 + 28 * math.sin(k / 17 * math.pi)
        d.ellipse((x - 14, y - 9, x + 14, y + 9), outline=(150, 150, 140), width=5)
    d.rounded_rectangle((615, 490, 665, 545), radius=6, fill=(180, 150, 40))
    d.arc((620, 460, 660, 510), 180, 360, fill=(160, 160, 160), width=8)
    # enseigne
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    do = ImageDraw.Draw(ov)
    do.ellipse((440, 60, 840, 220), fill=sign_col + (60,))
    ov = ov.filter(ImageFilter.GaussianBlur(25))
    img.paste(ov, (0, 0), ov)
    d = ImageDraw.Draw(img)
    d.rectangle((510, 100, 770, 175), fill=(10, 10, 10), outline=(70, 70, 70), width=4)
    d.text((640, 138), "ENTRÉE" if flip else "SORTIE", font=F("anton", 56), fill=sign_col, anchor="mm")
    d.ellipse((520, 650, 760, 690), fill=mix((25, 30, 28), sign_col, 0.25))
    # la foule arrive des deux côtés
    n = int(clamp(tt / 3.0) * 14)
    prng = np.random.default_rng(21)
    eyes = t > CLOCHE_EXIT_FLIP + 0.4
    for k in range(14):
        side = -1 if k % 2 == 0 else 1
        hgt = prng.uniform(330, 520)
        x_end = 640 + side * prng.uniform(260, 600)
        x_start = 640 + side * 900
        kk = clamp((tt - k * 0.2) / 2.2)
        if k >= n and kk <= 0:
            continue
        x = lerp(x_start, x_end, 1 - (1 - kk) ** 2)
        base = 730 + prng.uniform(0, 40)
        silhouette(d, x, base, hgt, col=(4, 6, 6), arms=1.1)
        if eyes:
            hr = hgt * 0.07
            hy = base - hgt + hr * 1.1
            for s in (-1, 1):
                d.ellipse((x + s * hr * 0.45 - 3, hy - 3, x + s * hr * 0.45 + 3, hy + 3), fill=(255, 255, 255))
    return img


# --------------------------------------------------------------- REWIND ----
SCENE_FUNCS = {}
INSERTS = []
_tt = 0.15
while _tt < 5.6:
    INSERTS.append(_tt)
    _tt += 0.45 - 0.33 * (_tt / 5.6)


def s_rewind(t, fi, rng):
    for k, it in enumerate(INSERTS):
        if it <= t < it + 2.0 / FPS:
            prng = np.random.default_rng(k + 100)
            name = ["arrivee", "murale", "corridor", "maths", "diner", "examen", "machine", "cloche"][k % 8]
            src = SCENE_FUNCS[name](prng.uniform(2.5, 5.8), fi, prng)
            if k % 3 == 0:
                src = ImageOps.invert(src)
            elif k % 3 == 1:
                a = np.asarray(src, np.float32)
                a = a.mean(-1, keepdims=True) * np.array([1.4, 0.2, 0.3])
                src = to_img(a)
            return src

    def rays_setup():
        yy, xx = np.mgrid[0:H, 0:W]
        return np.arctan2(yy - 360, xx - 640).astype(np.float32), np.hypot(yy - 360, xx - 640).astype(np.float32)

    ang, rad = cached("rays", rays_setup)
    rot = -t * (1.5 + t * 1.2)
    stripes = (np.sin(ang * 12 + rot + rad * 0.004 * t) > 0)[..., None]
    a = np.where(stripes, np.array([175, 0, 25], np.float32), np.array([12, 0, 4], np.float32))
    img = to_img(a)
    d = ImageDraw.Draw(img)
    R = 235 + 12 * math.sin(t * 9)
    cx, cy = 640 + rng.normal(0, 2 + t), 360 + rng.normal(0, 2 + t)
    d.ellipse((cx - R - 18, cy - R - 18, cx + R + 18, cy + R + 18), fill=(10, 10, 10))
    d.ellipse((cx - R, cy - R, cx + R, cy + R), fill=(242, 232, 205))
    for h in range(1, 13):
        a_ = h / 12 * 2 * math.pi
        lab = "13" if h == 12 else str(h)
        d.text((cx + math.sin(a_) * R * 0.8, cy - math.cos(a_) * R * 0.8), lab, font=F("anton", 46),
               fill=(20, 20, 20) if h != 12 else (200, 0, 0), anchor="mm")
    for k in range(60):
        a_ = k / 60 * 2 * math.pi
        r0 = R * (0.93 if k % 5 else 0.88)
        d.line([(cx + math.sin(a_) * r0, cy - math.cos(a_) * r0), (cx + math.sin(a_) * R * 0.97, cy - math.cos(a_) * R * 0.97)],
               fill=(30, 30, 30), width=3 if k % 5 else 6)
    spin = -(t ** 2.2) * 6
    for ang_, ln, wd, col in ((spin / 12, 0.5, 14, (15, 15, 15)), (spin, 0.75, 9, (15, 15, 15)), (spin * 7, 0.85, 3, (220, 0, 0))):
        d.line([(cx, cy), (cx + math.sin(ang_) * R * ln, cy - math.cos(ang_) * R * ln)], fill=col, width=wd)
    d.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), fill=(200, 0, 0))
    if t > 3.2:
        shaky_text(d, (640, 72), "DEMAIN, C'EST AUJOURD'HUI.", F("anton", 82), (255, 255, 255), rng,
                   amp=3 + 3 * (t - 3.2), split=6, stroke=4, stroke_fill=(0, 0, 0))
    return img


# ------------------------------------------------------------------ FIN ----
def s_fin(t, fi, rng):
    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    if t < 4.2:
        fade = 1 - clamp((t - 3.9) / 0.3)

        def c(v):
            return tuple(int(x * fade) for x in v)

        # le logo : son cercle, ouvert d'habitude, se referme... et rougit
        if t > 0.2:
            logo = Image.new("RGB", (W, H), (0, 0, 0))
            ImageDraw.Draw(logo).ellipse((640 - 168, 215 - 168, 640 + 168, 215 + 168), fill=(200, 200, 195))
            draw_logo(logo, 640, 215, 330, close=clamp((t - 1.6) / 1.4), lit=clamp((t - 2.4) / 1.0), t=t)
            k = clamp((t - 0.2) / 0.5) * fade
            img = Image.blend(img, logo, k)
            d = ImageDraw.Draw(img)
        if t > 1.0:
            s = "Inscriptions ouvertes en tout temps."
            n = int(clamp((t - 1.0) / 0.9) * len(s))
            d.text((640, 455), s[:n], font=F("type", 36), fill=c((210, 210, 210)), anchor="mm")
        if t > 2.3:
            shaky_text(d, (640, 530), "Pour toujours.", F("creep", 84), c((230, 0, 20)), rng, amp=1.5)
        if t > 1.0:
            d.text((640, 680), "Œuvre de fiction. Toute ressemblance avec un lundi réel serait purement fortuite.",
                   font=F("sans", 17), fill=c((110, 110, 110)), anchor="mm")
        return img
    if t < 4.95:
        k = clamp((t - 4.25) / 0.2) * (1 - blink(t, [4.62], 0.25))
        for s in (-1, 1):
            e = eye_img(46, (0, 4), 1 - k, iris=(120, 0, 0))
            img.paste(e, (640 + s * 85 - 46, 330 - 46), e)
    return img


SCENE_FUNCS.update({
    "titre": s_titre, "arrivee": s_arrivee, "murale": s_murale, "pause2": s_pause2, "corridor": s_corridor, "maths": s_maths,
    "diner": s_diner, "examen": s_examen, "machine": s_machine, "cloche": s_cloche,
    "rewind": s_rewind, "fin": s_fin,
})

HEADINGS = {
    "arrivee": "08:24 · ARRIVÉE", "murale": "08:25 · JE LAISSE MA TRACE", "corridor": "08:41 · CORRIDOR, 2e ÉTAGE",
    "maths": "09:05 · MATHÉMATIQUES", "machine": "09:25 · PAUSE — SALLE V.I.P.", "pause2": "10:45 · PAUSE — SALLE V.I.P.",
    "diner": "12:00 · DÎNER — SALLE V.I.P.", "examen": "13:15 · EXAMEN", "cloche": "14:55 · PAUSE",
    "rewind": "??:?? · ????",
}
CAPTIONS = {
    "arrivee": [(0.3, 3.6, "Le stationnement est plein."), (3.6, 7, "Il l'est toujours.")],
    "murale": [(0.3, 2.3, "Des centaines de noms. Des années de signatures."), (2.3, 4.5, "Elle ne partira jamais. Vous non plus.")],
    "corridor": [(0.3, 3.3, "Le corridor du 2e étage est un peu plus long aujourd'hui."), (3.3, 6.5, "Ne comptez pas les portes.")],
    "maths": [(0.3, 2.9, "Personne ne lève la main."), (3.1, 7, "Personne ne cligne des yeux.")],
    "machine": [(0.3, 2.6, "Dix minutes de pause."), (2.6, 5, "La machine a tout ce qu'il vous faut. Presque.")],
    "pause2": [(0.0, 2.0, "Encore la même pause.")],
    "diner": [(0.3, 3.0, "Le dîner finit à 12 h 45."), (3.0, 6, "Pour les autres.")],
    "examen": [(0.3, 3.2, "Prenez tout votre temps."), (3.2, 6.5, "Il prendra le vôtre.")],
    "cloche": [(0.3, 1.9, "13 h 45. La cloche."), (1.9, 3.4, "14 h 55. Déjà ?"), (3.4, 6, "Personne ne sort.")],
}
CAM = {"arrivee": 1, "murale": 4, "corridor": 3, "maths": 7, "machine": 9, "pause2": 9, "diner": 12, "examen": 13,
       "cloche": 1, "rewind": 0}


def osd(img, t, name, local, rng):
    d = ImageDraw.Draw(img)
    if int(t * 2) % 2 == 0:
        d.ellipse((42, 34, 66, 58), fill=(255, 25, 25))
    d.text((76, 46), "REC", font=F("vt", 44), fill=(255, 255, 255), anchor="lm", stroke_width=2, stroke_fill=(0, 0, 0))
    cam = CAM[name] if name != "rewind" else int(rng.integers(0, 99))
    d.text((W - 42, 46), f"CAM {cam:02d}", font=F("vt", 44), fill=(255, 255, 255), anchor="rm", stroke_width=2,
           stroke_fill=(0, 0, 0))
    day = "JOUR 04813" if (name == "rewind" and local > 5.5) else "JOUR 04812"
    d.text((42, H - 36), f"{day}  {osd_clock(name, local)}", font=F("vt", 40), fill=(255, 255, 255), anchor="ls",
           stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((W - 42, H - 36), "LECTURE >" if name != "rewind" else "<< RETOUR", font=F("vt", 40),
           fill=(255, 255, 255), anchor="rs", stroke_width=2, stroke_fill=(0, 0, 0))
    h0 = 1.9 if name == "cloche" else 0.0
    if h0 <= local < h0 + 2.2:
        d.text((W / 2, 52), HEADINGS[name], font=F("anton", 44), fill=(255, 245, 120), anchor="mm",
               stroke_width=4, stroke_fill=(0, 0, 0))


def _vo_chunks():
    """Découpe chaque réplique en phrases, minutées selon la durée réelle de la voix."""
    import re
    import wave
    out = []
    for key, start, who, text in VO_LINES:
        try:
            w = wave.open(os.path.join(HERE, "build", f"vo_{key}.wav"))
            dur = w.getnframes() / w.getframerate() / VO_SLOW.get(key, 1.0)
        except FileNotFoundError:
            dur = len(text) * 0.07
        parts = [p.strip() for p in re.split(r"(?<=[.?!])\s+", text) if p.strip()]
        total = sum(len(p) + 6 for p in parts)
        t = start
        for p in parts:
            d = dur * (len(p) + 6) / total
            out.append((t, t + d + 0.25, who, p))
            t += d
    return out


def _box_text(img, xy, txt, font, fill, pad=(18, 10), alpha=185, anchor="ms"):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox(xy, txt, font=font, anchor=anchor)
    box = Image.new("RGBA", (int(x1 - x0) + 2 * pad[0], int(y1 - y0) + 2 * pad[1]), (0, 0, 0, alpha))
    img.paste(box, (int(x0) - pad[0], int(y0) - pad[1]), box)
    ImageDraw.Draw(img).text(xy, txt, font=font, fill=fill, anchor=anchor)


def subtitles(img, t, name, local):
    """Dessinés APRÈS les effets VHS : toujours nets et lisibles."""
    # ce qui est dit (bas de l'écran)
    for a, b, who, txt in cached("vo_chunks", _vo_chunks):
        if a <= t < b:
            f = F("sans", 34)
            words, lines, cur = txt.split(), [], ""
            for w_ in words:
                test = (cur + " " + w_).strip()
                if f.getlength(test) > 1050:
                    lines.append(cur)
                    cur = w_
                else:
                    cur = test
            lines.append(cur)
            y = H - 82 - 46 * (len(lines) - 1)
            _box_text(img, (W / 2, y - 44), who, F("sans", 20), (255, 210, 60), pad=(10, 4), alpha=200)
            for ln in lines:
                _box_text(img, (W / 2, y), ln, f, (255, 255, 255))
                y += 46
            break
    # ce qui se passe (haut de l'écran, sous le titre de la scène)
    for a, b, txt in CAPTIONS.get(name, []):
        if a <= local < b:
            _box_text(img, (W / 2, 128), txt, F("vt", 46), (255, 245, 170), alpha=150)


# ----------------------------------------------------------- POST-EFFETS ---
def _post_setup():
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    vig = (1 - 0.32 * np.clip(r - 0.45, 0, None) ** 1.6)[..., None].astype(np.float32)
    scan = np.ones((H, 1, 1), np.float32)
    scan[1::3] = 0.92
    return vig * scan


def post(img, t, fi, g):
    rng = np.random.default_rng(fi * 9973 + 17)
    small = img.resize((320, 180), Image.BILINEAR)
    s = np.asarray(small, np.float32)
    s = np.clip(s - 140, 0, None) * 1.7
    bloom = np.asarray(to_img(s).filter(ImageFilter.GaussianBlur(6)).resize((W, H), Image.BILINEAR), np.float32)
    a = np.asarray(img, np.float32) + bloom * 0.45
    # saturation « bruyante »
    m = a.mean(-1, keepdims=True)
    a = m + (a - m) * 1.2
    # déchirures horizontales
    for _ in range(int(rng.poisson(0.3 + 3.5 * max(0.0, g - 0.3)))):
        y = int(rng.integers(0, H))
        h = int(rng.integers(2, int(6 + 40 * g)))
        a[y:y + h] = np.roll(a[y:y + h], int(rng.normal(0, 4 + 60 * g)), axis=1)
    # barre de tracking VHS qui descend
    ty = int((t * 160) % (H + 120)) - 60
    if 0 <= ty < H - 30:
        a[ty:ty + 14] = a[ty:ty + 14] * 0.8 + rng.random((14, W, 1)).astype(np.float32) * 50
    # aberration chromatique
    k = int(1 + 4 * g + rng.random() * 2 * g)
    a[..., 0] = np.roll(a[..., 0], k, axis=1)
    a[..., 2] = np.roll(a[..., 2], -k, axis=1)
    # grain
    a += rng.normal(0, 3 + 7 * g, (H, W, 1)).astype(np.float32)
    a *= cached("vigscan", _post_setup)
    a *= 1 + rng.normal(0, 0.012 + 0.02 * g)
    if g > 1.3 and rng.random() < 0.25:
        a = 255 - a
    return np.clip(a, 0, 255).astype(np.uint8)


def render_frame(fi):
    t = fi / FPS
    name, local, _ = scene_at(t)
    rng = np.random.default_rng(fi * 31 + 5)
    img = SCENE_FUNCS[name](local, fi, rng)
    if img.size != (W, H):
        img = img.resize((W, H))
    img = img.convert("RGB")
    if name not in ("titre", "fin"):
        osd(img, t, name, local, rng)
    out = Image.fromarray(post(img, t, fi, glitch_level(t)))
    subtitles(out, t, name, local)
    return out.tobytes()


def main():
    args = sys.argv[1:]
    if args and args[0] == "--preview":
        os.makedirs(os.path.join(HERE, "build", "preview"), exist_ok=True)
        for s in args[1:]:
            fi = int(float(s) * FPS)
            buf = render_frame(fi)
            Image.frombytes("RGB", (W, H), buf).save(os.path.join(HERE, "build", "preview", f"f_{float(s):05.2f}.png"))
            print("ok", s)
        return
    out = os.path.join(HERE, "build", "video_seule.mp4")
    n = int(DUR * FPS)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                           "-crf", "18", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    with Pool(os.cpu_count()) as pool:
        for i, buf in enumerate(pool.imap(render_frame, range(n), chunksize=4)):
            ff.stdin.write(buf)
            if i % 120 == 0:
                print(f"{i}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("vidéo :", out)


if __name__ == "__main__":
    main()
