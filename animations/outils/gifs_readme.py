"""Rend les quatre boucles du README et les convertit en GIF (assets/readme/), avec une planche de contrôle par scène.

À lancer avec le Python qui porte Manim (0.19), depuis la racine du dépôt :

    py -3.13 animations/outils/gifs_readme.py                  # les quatre, qualité finale
    py -3.13 animations/outils/gifs_readme.py --essai R2       # une seule, basse qualité, planche seulement

Les GIF visent moins de 4 Mo : 900 px de large, 15 images par seconde, palette ffmpeg. Si l'un dépasse, on baisse
dans l'ordre le nombre de couleurs, puis la largeur.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
SCENES = RACINE / "animations" / "scenes" / "readme.py"
RENDUS = RACINE / "animations" / "rendus"
GIFS = RACINE / "assets" / "readme"
BOUCLES = {"R1": ("R1_MelangeTirage", "melange_tirage"), "R2": ("R2_MesurerFlou", "mesurer_flou"),
           "R3": ("R3_UnPas", "un_pas"), "R4": ("R4_SeizePas", "seize_pas")}
LIMITE = 4_000_000


def lancer(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"échec : {' '.join(map(str, cmd[:4]))} …")


def duree(video: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def rendre(classe: str, essai: bool) -> Path:
    qualite = ["-ql"] if essai else ["-qh"]
    lancer([sys.executable, "-m", "manim", *qualite, "-r", "1280,560", "--fps", "15" if essai else "30",
            "--media_dir", str(RENDUS), "--disable_caching", str(SCENES), classe])
    videos = sorted(RENDUS.rglob(f"{classe}.mp4"), key=lambda p: p.stat().st_mtime)
    return videos[-1]


def planche(video: Path, nom: str) -> Path:
    sortie = RENDUS / f"planche_{nom}.png"
    d = duree(video)
    lancer(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"fps=12/{d},scale=427:-1,tile=4x3",
            "-frames:v", "1", str(sortie)])
    return sortie


def gif(video: Path, nom: str) -> Path:
    GIFS.mkdir(parents=True, exist_ok=True)
    sortie = GIFS / f"{nom}.gif"
    for couleurs, largeur in ((96, 900), (64, 900), (64, 800), (48, 760)):
        filtre = (f"fps=15,scale={largeur}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={couleurs}:stats_mode=diff[p];"
                  "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
        lancer(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", filtre, "-loop", "0", str(sortie)])
        if sortie.stat().st_size <= LIMITE:
            break
    print(f"  {sortie.relative_to(RACINE)} : {sortie.stat().st_size / 1e6:.2f} Mo, {couleurs} couleurs, {largeur} px, "
          f"{duree(video):.1f} s")
    return sortie


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--essai", nargs="*", help="rendu rapide de ces boucles (R1 … R4), planches seulement")
    args = ap.parse_args()
    RENDUS.mkdir(parents=True, exist_ok=True)
    cles = args.essai if args.essai else list(BOUCLES)
    for cle in cles:
        classe, nom = BOUCLES[cle]
        print(f"== {cle} {classe}")
        video = rendre(classe, essai=args.essai is not None)
        print(f"  planche : {planche(video, nom).relative_to(RACINE)}")
        if args.essai is None:
            gif(video, nom)


if __name__ == "__main__":
    main()
