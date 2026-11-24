"""Exporte les données réelles dont les animations ont besoin, depuis la copie des résultats (`resultats/`).

À lancer avec l'environnement du projet (il a besoin du paquet `flou` pour rendre les deux futurs A et B) :

    uv run python animations/outils/exporter.py

Écrit `animations/donnees/donnees.npz` (images) et `animations/donnees/chiffres.json` (nombres), puis vérifie que les
nombres exportés redonnent ceux du rapport. Les scènes ne lisent que ces deux fichiers : elles tournent avec le Python
qui porte Manim, sans dépendre du projet.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "src"))
from flou.data.dataset import BifurcationSet  # noqa: E402
from flou.train.runs import test_file  # noqa: E402

RES = RACINE / "resultats"
SORTIE = RACINE / "animations" / "donnees"
REG, DIF, JET = "core-regression-p0.30-s0", "core-diffusion-p0.30-s0", "core-tokens-K256-p0.30-s0"
CLASSES = ("nette-A", "nette-B", "floue", "autre")


def tableaux(run: str) -> np.lib.npyio.NpzFile:
    return np.load(RES / "runs" / run / "eval_arrays.npz")


def bande(run: str, fichier: str, ligne: int) -> np.ndarray:
    """Les 16 images d'une ligne d'une bande `filmstrip_*.png` (8 lignes x 16 colonnes, bandeau de titre en haut)."""
    img = np.asarray(Image.open(RES / "runs" / run / fichier).convert("RGB"))
    h, w = img.shape[:2]
    cw, ch = w // 16 - 6, h // 8 - 17 - 6
    y0 = round(ligne * h / 8) + 17 + 3
    return np.stack([img[y0:y0 + ch, round(c * w / 16) + 3: round(c * w / 16) + 3 + cw] for c in range(16)])


def main() -> None:
    SORTIE.mkdir(parents=True, exist_ok=True)
    chemin_test = test_file(str(RACINE / "data"), 0.3)
    bset = BifurcationSet.load(chemin_test)
    img_a, img_b, delta = bset.counterfactuals()
    ctx = bset.contexts()[:, -1]                                # dernière image du contexte

    zr, zd, zt = tableaux(REG), tableaux(DIF), tableaux(JET)
    n_ab = zr["main__ab_img"].shape[0]
    donnees = {
        "a": img_a[:n_ab], "b": img_b[:n_ab], "ctx": ctx[:n_ab],
        "reg": zr["main__ab_img"], "dif3": zd["n=3__ab_img"], "dif1": zd["n=1__ab_img"], "jet": zt["tau=1__ab_img"],
        "lab_reg": zr["main__label"][:n_ab], "lab_dif3": zd["n=3__label"][:n_ab], "lab_jet": zt["tau=1__label"][:n_ab],
        "alpha_reg": zr["main__alpha"], "alpha_dif3": zd["n=3__alpha"], "alpha_jet": zt["tau=1__alpha"],
        "alpha_dif1": zd["n=1__alpha"],
        "rho_reg": zr["main__rho"], "rho_dif3": zd["n=3__rho"], "rho_jet": zt["tau=1__rho"], "rho_dif1": zd["n=1__rho"],
        # bandes de 16 pas : 3e état des bandes (lignes 4 et 5), où la diffusion suit l'autre branche que la réalité
        "bande_reel": bande(REG, "filmstrip_net.png", 4), "bande_reg": bande(REG, "filmstrip_net.png", 5),
        "bande_dif": bande(DIF, "filmstrip_n3.png", 5),
    }
    # les 40 états imagés sont bien les 40 premiers états du jeu de test, dans l'ordre
    for run, z, cle in ((REG, zr, "main"), (DIF, zd, "n=3"), (JET, zt, "tau=1")):
        assert (z[f"{cle}__state"][:n_ab] == np.arange(n_ab)).all(), run
    np.savez_compressed(SORTIE / "donnees.npz", **donnees)

    def ev(run, variante):
        e = json.loads((RES / "runs" / run / "eval.json").read_text(encoding="utf-8"))
        return [v["summary"] for v in e["variants"] if (v["variant"] or "main") == variante][0]

    sr, sd, st = ev(REG, "main"), ev(DIF, "n=3"), ev(JET, "tau=1")
    risque = {r["modèle"]: {k: float(v) for k, v in r.items() if k[:4] in ("doux", "seui")}
              for r in csv.DictReader(open(RES / "analyse" / "table_risque.csv", encoding="utf-8"))}
    sigma = {}
    for g in range(3):
        rows = json.loads((RES / "runs" / f"core-diffusion-p0.30-s{g}" / "sigma_sweep.json").read_text(encoding="utf-8"))["rows"]
        sigma[f"s{g}"] = [{k: r[k] for k in ("sigma", "floue", "autre", "nettes", "rho_median")} for r in rows]
    chiffres = {
        "p": 0.3,
        "rho_max": json.loads((RES / "analyse" / "instrument.json").read_text(encoding="utf-8"))["rho_max"],
        "n_etats": sr["n"],
        "reg_floue": sr["counts"]["floue"] / sr["n"], "reg_mse": sr["mse"]["ratio_expected"],
        "dif3_nettes": (sd["counts"]["nette-A"] + sd["counts"]["nette-B"]) / sd["n"],
        "dif3_phat": sd["p_hat"]["value"], "dif3_nA": sd["counts"]["nette-A"], "dif3_nB": sd["counts"]["nette-B"],
        "jet_autre": st["counts"]["autre"] / st["n"],
        "risque": risque,
        "sigma": sigma,
        # débruiteur exact, un pas depuis du bruit pur : référence fixée avant le balayage (rapport, annexe E)
        "sigma_reference": {"sigma": [1, 2, 5, 10, 20, 30, 40, 60, 80],
                            "floue": [0.071, 0.144, 0.338, 0.564, 0.759, 0.856, 0.923, 0.984, 0.998]},
        # exemples d'entraînement au-dessus de chaque niveau de bruit, sur 2 227 200 (loi log-normale d'EDM)
        "exemples_au_dessus": {"5": 21406, "10": 3913, "20": 525, "40": 51, "80": 4},
        "delta_test_moyen": float(np.mean(delta)),
    }
    (SORTIE / "chiffres.json").write_text(json.dumps(chiffres, ensure_ascii=False, indent=1), encoding="utf-8")

    # contrôle : les nombres exportés redonnent ceux du rapport (graine 0, p = 0,3)
    assert abs(chiffres["reg_floue"] - 0.982) < 1e-9 and abs(chiffres["dif3_phat"] - 0.3271) < 1e-3
    assert abs(risque["A. Régression"]["doux_k1"] - 0.0307) < 1e-3 and abs(risque["B. Diffusion (EDM) n=3"]["doux_k1"] - 0.1586) < 1e-3
    print(f"écrit : {SORTIE / 'donnees.npz'} ({sum(v.nbytes for v in donnees.values()) / 1e6:.1f} Mo non compressés)")
    print(f"écrit : {SORTIE / 'chiffres.json'} ; contrôles avec le rapport : ok")


if __name__ == "__main__":
    main()
