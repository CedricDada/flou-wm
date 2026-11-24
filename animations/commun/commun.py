"""Style et aides partagés par toutes les scènes.

Les scènes ne lisent que `animations/donnees/` (écrit par `outils/exporter.py`) : images réelles des modèles,
coordonnées de leurs prédictions, chiffres du rapport. Rien n'est recalculé ici, sauf les illustrations signalées
comme telles à l'écran.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import *  # noqa: F401,F403

DOSSIER = Path(__file__).resolve().parents[1]
DONNEES = DOSSIER / "donnees"

Text.set_default(font="Segoe UI")
config.frame_height = 7.0
config.frame_width = 16.0

FOND = "#141414"
TEXTE = "#E8E8E8"
GRIS = "#8A8A8A"
GRIS_FONCE = "#4A4A4A"
REG = "#FF8A65"        # régression
DIF = "#4FC3F7"        # diffusion
JET = "#B39DDB"        # jetons
NETTE = "#81C784"
FLOUE = "#FFB74D"
AUTRE = "#9E9E9E"


def charger() -> tuple[dict, dict]:
    if not (DONNEES / "donnees.npz").exists():
        raise SystemExit("animations/donnees/ absent : lancer d'abord `uv run python animations/outils/exporter.py`")
    z = np.load(DONNEES / "donnees.npz")
    return {k: z[k] for k in z.files}, json.loads((DONNEES / "chiffres.json").read_text(encoding="utf-8"))


D, CHIFFRES = charger()


def fr(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}".replace(".", ",") + " %"


def image(arr: np.ndarray, hauteur: float = 3.0, cadre: str | None = GRIS_FONCE) -> Group:
    """Une image du jeu agrandie sans lissage (pixels nets), avec un cadre fin."""
    rgb = np.asarray(arr, dtype=np.uint8)
    im = ImageMobject(rgb).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.height = hauteur
    if cadre is None:
        return Group(im)
    return Group(im, SurroundingRectangle(im, buff=0, color=cadre, stroke_width=1.5))


def etiquette(texte: str, couleur: str = GRIS, taille: int = 24) -> Text:
    return Text(texte, font_size=taille, color=couleur)


class Boucle(Scene):
    """Boucle courte pour le README : fond sombre, peu de texte, revient à l'écran vide pour boucler sans saut."""

    def setup(self):
        super().setup()
        self.camera.background_color = FOND

    def finir(self, tenue: float = 1.8):
        self.wait(tenue)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        self.wait(0.2)
