#!/usr/bin/env python
"""Figures et tableaux du rapport, produits directement depuis les résultats de la campagne.

Entrées : la copie légère des résultats de la campagne (`resultats/`, versionnée : mesures par run, journaux d'entraînement, tableaux par état), dont les journaux de l'essai écarté à 100 000 transitions (`resultats/essai_100000_transitions/`). Sorties : `rapport/figures/*.pdf|png`
et `rapport/tableaux/*.tex`, plus `rapport/figures/chiffres.json`, qui rassemble les valeurs citées dans le texte.

    uv run python rapport/make_figures.py
    uv run python rapport/make_figures.py --etat flou-etat      # l'état complet téléchargé de Kaggle, s'il est là
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from PIL import Image  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
SORTIE_FIG = RACINE / "rapport" / "figures"
SORTIE_TAB = RACINE / "rapport" / "tableaux"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
COULEUR = {"regression": "#c0392b", "diffusion": "#2471a3", "tokens": "#7d3c98", "oracle": "#1e8449",
           "reference": "#555555"}
VIRGULE = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))


# ------------------------------------------------------------------------------------------------ lecture
def _analyse(etat: Path) -> Path:
    """Sorties de l'analyse : `analyse/` dans la copie versionnée, `reports/` dans l'état complet de Kaggle."""
    return etat / "analyse" if (etat / "analyse").is_dir() else etat / "reports"

def charger(etat: Path) -> list[dict]:
    """Une ligne par (run, variante) : toutes les mesures utiles, à plat."""
    lignes = []
    for d in sorted((etat / "runs").iterdir()):
        f = d / "eval.json"
        if not f.exists():
            continue
        ev = json.loads(f.read_text(encoding="utf-8"))
        spec = ev["spec"]
        for v in ev["variants"]:
            s = v["summary"]
            n = s["n"]
            c, r = s["counts"], s["rail"]["counts"]
            ro = v.get("rollout") or {}
            track = ro.get("track_ok") or [np.nan]
            lignes.append({
                "run": d.name, "famille": spec["family"], "groupe": spec["group"], "p": spec["p"], "graine": spec["seed"],
                "K": spec["K"], "grille": spec["grid"], "variante": v["variant"] or "main",
                "floue": c["floue"] / n, "autre": c["autre"] / n, "nettes": (c["nette-A"] + c["nette-B"]) / n,
                "rail_floue": r["floue"] / n, "rail_autre": r["autre"] / n,
                "rail_nettes": (r["nette-A"] + r["nette-B"]) / n,
                "p_hat": s["p_hat"]["value"], "p_lo": s["p_hat"]["ci_lo"], "p_hi": s["p_hat"]["ci_hi"],
                "n_nettes": c["nette-A"] + c["nette-B"], "alpha": s["alpha_mean"],
                "mse_esp": s["mse"]["ratio_expected"], "mse_tire": s["mse"]["ratio_realized"],
                "rappel": s["recall"]["value"], "suivi1": track[0], "suivi16": track[-1],
            })
    return lignes


def choisir(lignes, **critères):
    return [r for r in lignes if all(r[k] == v for k, v in critères.items())]


def moy(rr, cle):
    x = np.array([r[cle] for r in rr], dtype=float)
    x = x[~np.isnan(x)]
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan")) if len(x) else (np.nan, np.nan)


def pct(x, d=1):
    return "--" if np.isnan(x) else f"{100 * x:.{d}f}".replace(".", ",")


def num(x, d=2):
    return "--" if x is None or np.isnan(x) else f"{x:.{d}f}".replace(".", ",")


def enregistrer(fig, nom):
    fig.savefig(SORTIE_FIG / f"{nom}.pdf")
    fig.savefig(SORTIE_FIG / f"{nom}.png", dpi=200)
    plt.close(fig)
    print(f"  figure : {nom}")


# ------------------------------------------------------------------------------------------------ images
def planche(etat: Path, run: str, variante: str) -> Image.Image:
    return Image.open(etat / "runs" / run / f"samples_{variante}.png").convert("RGB")


def case(img: Image.Image, ligne: int, colonne: int, n_l: int = 6, n_c: int = 6, bandeau: int = 15) -> Image.Image:
    """Une case d'une planche `samples_*.png` (6 lignes x 6 colonnes, bandeau de titre en haut de chaque case)."""
    w, h = img.size
    cw, ch = w // n_c - 6, h // n_l - bandeau - 6          # taille fixe : les cases se superposent pixel à pixel
    x0 = round(colonne * w / n_c) + 3
    y0 = round(ligne * h / n_l) + bandeau + 3
    return img.crop((x0, y0, x0 + cw, y0 + ch))


def fig_jeu(etat: Path):
    """Figure 1 : un état, ses deux futurs possibles et leur mélange pondéré."""
    pl = planche(etat, "core-regression-p0.30-s0", "main")
    ligne = 2
    ctx, a, b = (np.asarray(case(pl, ligne, c), dtype=float) for c in range(3))
    melange = 0.3 * a + 0.7 * b
    cases = [ctx, a, b, melange]
    titres = ["image $t$\n(dernière du contexte)", "futur A : ennemi à droite\n(probabilité $p = 0{,}3$)",
              "futur B : ennemi à gauche\n(probabilité $0{,}7$)", "mélange\n$0{,}3\\,A + 0{,}7\\,B$"]
    fig, ax = plt.subplots(1, 4, figsize=(6.6, 2.2))
    for x, im, t in zip(ax, cases, titres):
        x.imshow(np.clip(im, 0, 255).astype(np.uint8), interpolation="nearest")
        x.set_title(t, fontsize=7)
        x.axis("off")
    enregistrer(fig, "fig_jeu")


def fig_echantillons(etat: Path):
    """Figure : les mêmes états de test, vus par chaque famille."""
    sources = [("core-regression-p0.30-s0", "main", [3], "Régression"),
               ("core-diffusion-p0.30-s0", "n1", [3], "Diffusion\n$n = 1$"),
               ("core-diffusion-p0.30-s0", "n3", [3, 4], "Diffusion\n$n = 3$"),
               ("core-tokens-K256-p0.30-s0", "tau1", [3, 4], "Jetons\n$K = 256$")]
    lignes = [1, 2, 4]
    ref = planche(etat, "core-regression-p0.30-s0", "main")
    colonnes = [("Contexte $t$", ref, 0), ("A", ref, 1), ("B", ref, 2)]
    for run, var, cols, titre in sources:
        pl = planche(etat, run, var)
        for j, c in enumerate(cols):
            colonnes.append((titre + (f"\n(tirage {j + 1})" if len(cols) > 1 else ""), pl, c))
    fig, ax = plt.subplots(len(lignes), len(colonnes), figsize=(6.6, 2.75))
    for i, l in enumerate(lignes):
        for j, (titre, pl, c) in enumerate(colonnes):
            ax[i, j].imshow(case(pl, l, c), interpolation="nearest")
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(titre, fontsize=6.5)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    enregistrer(fig, "fig_echantillons")


def fig_rollouts(etat: Path):
    """Prédictions enchaînées sur 16 pas, même état : réalité, régression, diffusion n = 3."""
    etat_idx = 2                                   # 3e état des bandes : la diffusion y choisit l'autre branche
    sources = [("réalité", "core-regression-p0.30-s0", "filmstrip_net.png", 2 * etat_idx),
               ("régression", "core-regression-p0.30-s0", "filmstrip_net.png", 2 * etat_idx + 1),
               ("diffusion $n=3$", "core-diffusion-p0.30-s0", "filmstrip_n3.png", 2 * etat_idx + 1)]
    pas = [1, 2, 3, 6, 7, 8, 12, 13, 16]
    fig, ax = plt.subplots(len(sources), len(pas), figsize=(6.6, 2.45))
    for i, (nom, run, fichier, ligne) in enumerate(sources):
        img = Image.open(etat / "runs" / run / fichier).convert("RGB")
        for j, t in enumerate(pas):
            ax[i, j].imshow(case(img, ligne, t - 1, n_l=8, n_c=16, bandeau=17), interpolation="nearest")
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            for s in ax[i, j].spines.values():
                s.set_visible(False)
            if i == 0:
                ax[i, j].set_title(f"$t+{t}$", fontsize=7)
        ax[i, 0].set_ylabel(nom, fontsize=7)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    enregistrer(fig, "fig_rollouts")


def tableau_sensibilite(etat: Path, chiffres):
    """Part d'images « autre » selon le seuil ρmax (×0,5, ×1, ×2), recalculée sur les sorties des modèles."""
    rho_max = json.loads((_analyse(etat) / "instrument.json").read_text(encoding="utf-8"))["rho_max"]
    familles = [("Régression", ["core-regression-p0.30-s%d"], "main"),
                ("Diffusion, $n=1$", ["core-diffusion-p0.30-s%d"], "n=1"),
                ("Diffusion, $n=3$", ["core-diffusion-p0.30-s%d"], "n=3"),
                ("Diffusion, $n=20$", ["core-diffusion-p0.30-s%d"], "n=20"),
                ("Jetons, $K=256$", ["core-tokens-K256-p0.30-s%d"], "tau=1")]
    corps = []
    for nom, motifs, cle in familles:
        parts = {f: [] for f in (0.5, 1.0, 2.0)}
        floues = {f: [] for f in (0.5, 1.0, 2.0)}
        for g in range(3):
            z = np.load(etat / "runs" / (motifs[0] % g) / "eval_arrays.npz")
            rho, alpha = z[f"{cle}__rho"], z[f"{cle}__alpha"]
            for f in parts:
                autre = (rho > f * rho_max) | np.isnan(rho)
                parts[f].append(float(autre.mean()))
                floues[f].append(float(((alpha > 0.2) & (alpha < 0.8) & ~autre).mean()))
        m = {f: (np.mean(parts[f]), np.mean(floues[f])) for f in parts}
        chiffres.setdefault("sensibilite", {})[nom] = {str(f): m[f] for f in m}
        corps.append(f"{nom} & " + " & ".join(f"{pct(m[f][0])} & {pct(m[f][1])}" for f in (0.5, 1.0, 2.0)) + " \\\\")
    tex = ("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n"
           "& \\multicolumn{2}{c}{$\\rho_{\\max}\\times0{,}5$} & \\multicolumn{2}{c}{$\\rho_{\\max}$ (retenu)} "
           "& \\multicolumn{2}{c}{$\\rho_{\\max}\\times2$} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
           "Modèle & autre & floue & autre & floue & autre & floue \\\\\n\\midrule\n"
           + "\n".join(corps) + "\n\\bottomrule\n\\end{tabular}\n")
    (SORTIE_TAB / "sensibilite.tex").write_text(tex, encoding="utf-8")
    print("  tableau : sensibilite")


# ------------------------------------------------------------------------------------------------ courbes
def lire_log(path: Path) -> tuple[dict, dict]:
    train, val = {}, {}
    for l in path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("stage", "main") != "main" or "step" not in r:
            continue
        if "loss" in r:
            train[r["step"]] = r["loss"]
        if "val_loss" in r:
            val[r["step"]] = r["val_loss"]
    return dict(sorted(train.items())), dict(sorted(val.items()))


def fig_surapprentissage(etat: Path, chiffres: dict):
    """Annexe : régression entraînée sur 100 000 puis 400 000 transitions."""
    essai = etat / "essai_100000_transitions/log.jsonl"
    if not essai.exists():                     # recalcul complet : l'essai écarté n'est pas relancé, ses journaux sont versionnés
        essai = RACINE / "resultats/essai_100000_transitions/log.jsonl"
    cas = [("100 000 transitions (22 passages)", essai),
           ("400 000 transitions (5,6 passages)", etat / "runs/core-regression-p0.30-s0/log.jsonl")]
    plancher = 6.6e-4
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.3), sharey=True)
    for x, (titre, chemin) in zip(ax, cas):
        tr, va = lire_log(chemin)
        s = np.array(list(tr)); y = np.array(list(tr.values()))
        k = 15
        lisse = np.convolve(y, np.ones(k) / k, mode="valid")
        x.plot(s[k - 1:], lisse, color="#888888", lw=1, label="entraînement (moyenne glissante)")
        x.plot(list(va), list(va.values()), "o-", color=COULEUR["regression"], ms=3, lw=1.3, label="validation")
        x.axhline(plancher, ls="--", color="black", lw=0.9, label="plancher de Bayes")
        x.set_yscale("log")
        x.set_title(titre)
        x.set_xlabel("pas d'entraînement")
        x.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v / 1000)} k" if v else "0"))
        vmin = min(va.values())
        chiffres.setdefault("surapprentissage", {})[titre] = {
            "val_min": vmin, "val_fin": list(va.values())[-1], "pas_min": min(va, key=va.get),
            "train_fin_10": float(np.mean(y[-10:])), "hausse_val": list(va.values())[-1] / vmin - 1}
    ax[0].set_ylabel("perte MSE (échelle log)")
    ax[1].legend(loc="lower left", frameon=False)
    ax[0].set_ylim(1e-4, 3e-3)
    enregistrer(fig, "fig_surapprentissage")


def fig_classes(lignes, chiffres):
    """Répartition floue / nette / autre par modèle (p = 0,3), image entière et rail."""
    modeles = [("Régression", dict(famille="regression", groupe="core")),
               ("Diff. $n=1$", dict(famille="diffusion", groupe="core", variante="n=1")),
               ("Diff. $n=3$", dict(famille="diffusion", groupe="core", variante="n=3")),
               ("Diff. $n=20$", dict(famille="diffusion", groupe="core", variante="n=20")),
               ("Jetons $K=64$", dict(famille="tokens", groupe="k_axis", K=64)),
               ("Jetons $K=256$", dict(famille="tokens", groupe="core")),
               ("Jetons $K=1024$", dict(famille="tokens", groupe="k_axis", K=1024)),
               ("Oracle", dict(famille="oracle"))]
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.5), sharey=True)
    for x, prefixe, titre in ((ax[0], "", "image entière (mesure principale)"),
                              (ax[1], "rail_", "restreinte au rail de l'ennemi (exploratoire)")):
        bas = np.zeros(len(modeles))
        for cle, coul, lab in (("nettes", "#4c9a6a", "nette"), ("floue", COULEUR["regression"], "floue"),
                               ("autre", "#b9b9b9", "autre")):
            h = np.array([moy(choisir(lignes, **crit), prefixe + cle)[0] for _, crit in modeles])
            x.bar(range(len(modeles)), h, bottom=bas, color=coul, label=lab, width=0.72)
            bas += h
        x.set_xticks(range(len(modeles)), [m for m, _ in modeles], rotation=40, ha="right")
        x.set_title(titre)
        x.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(100 * v))} %"))
    ax[0].set_ylabel("part des 2 000 états")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False)
    enregistrer(fig, "fig_classes")


def fig_calibration(lignes, chiffres):
    """p̂ selon le nombre de pas n (3 graines) et selon p (axe p)."""
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
    ns = [3, 5, 20]
    jouet = {3: 0.19, 5: 0.23, 20: 0.27}
    for g, mk in zip(range(3), "osD"):
        rr = [choisir(lignes, famille="diffusion", groupe="core", graine=g, variante=f"n={n}")[0] for n in ns]
        y = np.array([r["p_hat"] for r in rr]); lo = np.array([r["p_lo"] for r in rr]); hi = np.array([r["p_hi"] for r in rr])
        xs = np.arange(3) + (g - 1) * 0.09
        ax[0].errorbar(xs, y, yerr=[y - lo, hi - y], fmt=mk + "-", color=COULEUR["diffusion"],
                       alpha=0.5 + 0.25 * g, ms=4, lw=1, capsize=2, label=f"graine {g}")
    ax[0].plot(range(3), [jouet[n] for n in ns], "x--", color=COULEUR["reference"], label="débruiteur exact (jouet 1D)")
    ax[0].axhline(0.3, color="black", lw=0.8)
    ax[0].axhspan(0.25, 0.35, color="0.88", zorder=0, label="$p \\pm 0{,}05$")
    ax[0].set_xticks(range(3), [str(n) for n in ns]); ax[0].set_xlim(-0.4, 2.4)
    ax[0].set_xlabel("pas d'échantillonnage $n$"); ax[0].set_ylabel("$\\hat p$")
    ax[0].set_title("diffusion, $p = 0{,}3$"); ax[0].yaxis.set_major_formatter(VIRGULE)
    ax[0].legend(frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)

    ps = [0.1, 0.3, 0.5]
    for fam, var, lab, mk in (("diffusion", "n=3", "diffusion $n=3$", "o"), ("diffusion", "n=20", "diffusion $n=20$", "s"),
                              ("tokens", "tau=1", "jetons $K=256$", "^")):
        y, lo, hi = [], [], []
        for p in ps:
            r = [x for x in choisir(lignes, famille=fam, variante=var, graine=0) if abs(x["p"] - p) < 1e-9
                 and x["groupe"] in ("core", "p_axis") and (x["K"] in (None, 256)) and x["grille"] == 8][0]
            y.append(r["p_hat"]); lo.append(r["p_lo"]); hi.append(r["p_hi"])
        y, lo, hi = map(np.array, (y, lo, hi))
        coul = COULEUR["tokens"] if fam == "tokens" else COULEUR["diffusion"]
        ax[1].errorbar(ps, y, yerr=[y - lo, hi - y], fmt=mk + "-", color=coul, ms=4, lw=1, capsize=2, label=lab,
                       alpha=1 if var != "n=20" else 0.55)
    ax[1].plot([0, 0.6], [0, 0.6], color="black", lw=0.8, label="$\\hat p = p$")
    ax[1].set_xlim(0.03, 0.57); ax[1].set_ylim(0, 0.6)
    ax[1].set_xlabel("probabilité réelle $p$"); ax[1].set_title("graine 0, trois valeurs de $p$")
    ax[1].xaxis.set_major_formatter(VIRGULE); ax[1].yaxis.set_major_formatter(VIRGULE)
    ax[1].legend(frameon=False, fontsize=6.5)
    enregistrer(fig, "fig_calibration")


def fig_sigma(etat: Path, chiffres):
    """Balayage de σ : un pas d'Euler depuis σ, contre le débruiteur exact (référence fixée avant la mesure)."""
    ref_sigma = [1, 2, 5, 10, 20, 30, 40, 60, 80]
    ref_floue = [0.071, 0.144, 0.338, 0.564, 0.759, 0.856, 0.923, 0.984, 0.998]
    ref_p5 = [0.000, 0.000, 0.003, 0.034, 0.109, 0.158, 0.187, 0.221, 0.239]
    ref_p95 = [1.000, 1.000, 0.985, 0.842, 0.600, 0.496, 0.444, 0.394, 0.370]
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.2))
    ax[0].plot(ref_sigma, ref_floue, "k--", lw=1, label="débruiteur exact")
    ax[2].fill_between(ref_sigma, ref_p5, ref_p95, color="0.8", label="débruiteur exact")
    for g in range(3):
        sw = json.loads((etat / f"runs/core-diffusion-p0.30-s{g}/sigma_sweep.json").read_text(encoding="utf-8"))["rows"]
        s = [r["sigma"] for r in sw]
        kw = dict(color=COULEUR["diffusion"], alpha=0.45 + 0.25 * g, ms=3, lw=1)
        ax[0].plot(s, [r["floue"] for r in sw], "o-", label=f"réseau, graine {g}", **kw)
        ax[1].plot(s, [r["autre"] for r in sw], "o-", **kw)
        ax[2].plot(s, [r["alpha_p5"] for r in sw], "-", **kw)
        ax[2].plot(s, [r["alpha_p95"] for r in sw], "-", **kw)
        chiffres.setdefault("sigma", {})[f"s{g}"] = {str(r["sigma"]): {k: r[k] for k in ("floue", "autre", "rho_median",
                                                                                         "alpha_p5", "alpha_p95")} for r in sw}
    for x in ax:
        x.set_xscale("log"); x.set_xticks([1, 2, 5, 10, 20, 40, 80], ["1", "2", "5", "10", "20", "40", "80"])
        x.set_xlabel("$\\sigma$ de départ (un seul pas)")
        x.axvspan(4, 12, color="#e8f1f8", zorder=0)
    ax[0].set_title("part d'images floues"); ax[1].set_title("part d'images « autre »")
    ax[2].set_title("$\\hat\\alpha$, 5$^e$ et 95$^e$ centiles")
    for x in ax[:2]:
        x.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(100 * v))} %"))
    ax[2].yaxis.set_major_formatter(VIRGULE)
    ax[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout(w_pad=0.6)
    enregistrer(fig, "fig_sigma")


def fig_k(lignes, chiffres):
    """Axe K : rappel au plafond, mais « autre », MSE et suivi à 16 pas bougent."""
    Ks = [64, 256, 1024]
    def runs_k(K):
        return [r for r in lignes if r["famille"] == "tokens" and r["K"] == K and r["grille"] == 8 and abs(r["p"] - 0.3) < 1e-9
                and r["groupe"] in ("core", "k_axis")]
    grille = choisir(lignes, groupe="grid")
    mesures = [("rappel", "rappel du petit objet", True), ("autre", "part « autre »", True),
               ("mse_esp", "MSE / plancher de Bayes", False), ("suivi16", "suivi à 16 pas", True)]
    fig, ax = plt.subplots(1, 4, figsize=(6.6, 1.9))
    for x, (cle, titre, en_pct) in zip(ax, mesures):
        for i, K in enumerate(Ks):
            v = [r[cle] for r in runs_k(K)]
            x.plot([i] * len(v), v, "o", color=COULEUR["tokens"], alpha=0.5, ms=3.5)
            x.plot([i - 0.2, i + 0.2], [np.mean(v)] * 2, color="black", lw=1.2)
        if cle == "autre":
            v = [np.mean([r["rail_autre"] for r in runs_k(K)]) for K in Ks]
            x.plot(range(3), v, "s--", color="0.45", ms=3, lw=0.9, label="au rail")
            x.legend(frameon=False, fontsize=6.5)
        g = grille[0][cle]
        x.plot([1.25], [g], "*", color="#d4a017", ms=7, label="grille $16\\times16$" if cle == "rappel" else None)
        x.set_xticks(range(3), [str(K) for K in Ks]); x.set_xlabel("$K$")
        x.set_title(titre, fontsize=8)
        x.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f} %") if en_pct else VIRGULE)
        if cle == "rappel":
            x.set_ylim(0.9, 1.0)
            x.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout(w_pad=0.5)
    enregistrer(fig, "fig_k")
    for K in Ks:
        rr = runs_k(K)
        chiffres.setdefault("K", {})[str(K)] = {c: moy(rr, c) for c in ("rappel", "autre", "rail_autre", "mse_esp", "suivi16")}
    chiffres["K"]["grille16"] = {c: grille[0][c] for c in ("rappel", "autre", "rail_autre", "mse_esp", "suivi16")}


def fig_p(lignes, chiffres):
    """Axe p : la régression reste au plancher mais le classifieur ne voit plus son flou à p = 0,1."""
    ps = [0.1, 0.3, 0.5]
    def reg(p):
        return [r for r in lignes if r["famille"] == "regression" and abs(r["p"] - p) < 1e-9 and r["graine"] == 0
                and r["groupe"] in ("core", "p_axis")][0]
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.1))
    ax[0].plot(ps, [reg(p)["floue"] for p in ps], "o-", color=COULEUR["regression"], label="part classée « floue »")
    ax[0].plot(ps, [reg(p)["nettes"] for p in ps], "s--", color="#4c9a6a", label="part classée « nette »")
    ax[0].axvspan(0, 0.2, color="0.9", zorder=0)
    ax[0].set_xlim(0.05, 0.55); ax[0].set_xlabel("$p$"); ax[0].set_title("régression")
    ax[0].xaxis.set_major_formatter(VIRGULE)
    ax[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f} %"))
    ax[0].legend(frameon=False, fontsize=6.5, loc="center right")
    ax[1].plot(ps, [reg(p)["alpha"] for p in ps], "o-", color=COULEUR["regression"], label="$\\hat\\alpha$ moyen")
    ax[1].plot([0, 0.6], [0, 0.6], color="black", lw=0.8, label="$\\hat\\alpha = p$ (mélange idéal)")
    ax[1].axhspan(0, 0.2, color="0.9", zorder=0); ax[1].axhspan(0.8, 1, color="0.9", zorder=0)
    ax[1].text(0.34, 0.06, "classé « nette-B »", fontsize=6.5, color="0.35")
    ax[1].set_xlim(0.05, 0.55); ax[1].set_ylim(0, 0.6); ax[1].set_xlabel("$p$"); ax[1].set_title("régression : position $\\hat\\alpha$")
    ax[1].xaxis.set_major_formatter(VIRGULE); ax[1].yaxis.set_major_formatter(VIRGULE)
    ax[1].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(w_pad=1.2)
    enregistrer(fig, "fig_p")
    chiffres["axe_p_regression"] = {str(p): {c: reg(p)[c] for c in ("floue", "nettes", "alpha", "mse_esp")} for p in ps}


def fig_risque(etat: Path, chiffres):
    """Niveau 1 : erreur d'estimation du risque de contact selon l'horizon, deux lecteurs."""
    import csv
    rows = {r["modèle"]: r for r in csv.DictReader(open(_analyse(etat) / "table_risque.csv", encoding="utf-8"))}
    ks = [1, 3, 5]
    series = [("A. Régression", "régression", COULEUR["regression"]),
              ("B. Diffusion (EDM) n=3", "diffusion $n=3$", COULEUR["diffusion"]),
              ("C. Jetons K=256", "jetons $K=256$", COULEUR["tokens"])]
    ideal = {"doux": [0.001, 0.001, 0.002], "seuil": [0.300, 0.345, 0.447]}
    calibre = [0.132, 0.157, 0.206]
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.3), sharey=True)
    for x, lecteur, titre in ((ax[0], "doux", "lecteur doux (affine en l'image)"), (ax[1], "seuil", "lecteur à seuil")):
        for cle, lab, coul in series:
            x.plot(ks, [float(rows[cle][f"{lecteur}_k{k}"]) for k in ks], "o-", color=coul, label=lab, ms=4)
        x.plot(ks, ideal[lecteur], "x:", color=COULEUR["regression"], alpha=0.6, label="image moyenne exacte")
        x.plot(ks, calibre, "x:", color=COULEUR["diffusion"], alpha=0.6, label="générateur parfait, 8 tirages")
        x.set_xticks(ks); x.set_xlabel("horizon $k$ (pas)"); x.set_title(titre)
        x.yaxis.set_major_formatter(VIRGULE)
    ax[0].set_ylabel("erreur moyenne $|\\hat C - C^\\star|$")
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(1.05, -0.24), ncol=3)
    enregistrer(fig, "fig_risque")
    chiffres["risque"] = {cle: {k: float(v) for k, v in rows[cle].items() if k[:4] in ("doux", "seui")} for cle, _, _ in series}


# ------------------------------------------------------------------------------------------------ tableaux
def tableau_principal(lignes, chiffres):
    modeles = [("Régression", dict(famille="regression", groupe="core")),
               ("Diffusion, $n=1$", dict(famille="diffusion", groupe="core", variante="n=1")),
               ("Diffusion, $n=3$", dict(famille="diffusion", groupe="core", variante="n=3")),
               ("Diffusion, $n=5$", dict(famille="diffusion", groupe="core", variante="n=5")),
               ("Diffusion, $n=20$", dict(famille="diffusion", groupe="core", variante="n=20")),
               ("Jetons, $K=64$", dict(famille="tokens", groupe="k_axis", K=64)),
               ("Jetons, $K=256$", dict(famille="tokens", groupe="core")),
               ("Jetons, $K=1\\,024$", dict(famille="tokens", groupe="k_axis", K=1024)),
               ("Jetons, $K=256$, grille $16\\times16$ $^\\dagger$", dict(groupe="grid")),
               ("Régresseur oracle $^\\dagger$", dict(famille="oracle"))]
    corps = []
    for nom, crit in modeles:
        rr = choisir(lignes, **crit)
        m = {c: moy(rr, c)[0] for c in ("floue", "nettes", "autre", "rail_autre", "mse_esp", "p_hat", "rappel", "suivi16")}
        if min(r["n_nettes"] for r in rr) < 100:        # p̂ sur quelques dizaines d'images : sans valeur
            m["p_hat"] = float("nan")
        chiffres.setdefault("principal", {})[nom] = m
        corps.append(f"{nom} & {pct(m['floue'])} & {pct(m['nettes'])} & {pct(m['autre'])} & {pct(m['rail_autre'])} & "
                     f"{num(m['mse_esp'])} & {num(m['p_hat'], 3)} & {pct(m['rappel'])} & {pct(m['suivi16'], 0)} \\\\")
    tex = ("\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n"
           "& \\multicolumn{3}{c}{image entière (\\%)} & rail & MSE / & & rappel & suivi \\\\\n"
           "\\cmidrule(lr){2-4}\n"
           "Modèle & floue & nette & autre & autre (\\%) & plancher & $\\hat p$ & (\\%) & 16 pas (\\%) \\\\\n\\midrule\n"
           + "\n".join(corps) + "\n\\bottomrule\n\\end{tabular}\n")
    (SORTIE_TAB / "principal.tex").write_text(tex, encoding="utf-8")
    print("  tableau : principal")


def tableau_par_graine(lignes):
    corps = []
    for fam, crit, nom in (("regression", dict(famille="regression", groupe="core"), "Régression"),
                           ("diffusion", dict(famille="diffusion", groupe="core", variante="n=3"), "Diffusion, $n=3$"),
                           ("tokens", dict(famille="tokens", groupe="core"), "Jetons, $K=256$")):
        for r in sorted(choisir(lignes, **crit), key=lambda r: r["graine"]):
            ph = "--" if r["n_nettes"] < 100 else f"{num(r['p_hat'], 3)} [{num(r['p_lo'], 3)} ; {num(r['p_hi'], 3)}]"
            corps.append(f"{nom} & {r['graine']} & {pct(r['floue'])} & {pct(r['nettes'])} & {pct(r['autre'])} & "
                         f"{r['n_nettes']} & {ph} & {num(r['mse_esp'], 3)} \\\\")
        corps.append("\\addlinespace")
    tex = ("\\begin{tabular}{@{}lcrrrrlr@{}}\n\\toprule\n"
           "Modèle & graine & floue (\\%) & nette (\\%) & autre (\\%) & images nettes & $\\hat p$ [IC 95 \\%] & MSE / plancher \\\\\n"
           "\\midrule\n" + "\n".join(corps[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    (SORTIE_TAB / "par_graine.tex").write_text(tex, encoding="utf-8")
    print("  tableau : par_graine")


def tableau_calcul(etat: Path, chiffres):
    import csv
    rows = list(csv.DictReader(open(_analyse(etat) / "calcul.csv", encoding="utf-8")))
    total = sum(float(r["heures_gpu"]) for r in rows)
    groupes = {}
    for r in rows:
        groupes.setdefault(r["groupe"], []).append(float(r["heures_gpu"]))
    chiffres["calcul"] = {"total_h": total, "runs": len(rows), "par_groupe": {g: [len(v), sum(v)] for g, v in groupes.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etat", type=Path, default=RACINE / "resultats")
    args = ap.parse_args()
    SORTIE_FIG.mkdir(parents=True, exist_ok=True)
    SORTIE_TAB.mkdir(parents=True, exist_ok=True)
    lignes = charger(args.etat)
    print(f"{len(lignes)} variantes lues dans {args.etat}")
    chiffres: dict = {}
    fig_jeu(args.etat)
    fig_echantillons(args.etat)
    fig_rollouts(args.etat)
    fig_surapprentissage(args.etat, chiffres)
    fig_classes(lignes, chiffres)
    fig_calibration(lignes, chiffres)
    fig_sigma(args.etat, chiffres)
    fig_k(lignes, chiffres)
    fig_p(lignes, chiffres)
    fig_risque(args.etat, chiffres)
    tableau_principal(lignes, chiffres)
    tableau_par_graine(lignes)
    tableau_calcul(args.etat, chiffres)
    tableau_sensibilite(args.etat, chiffres)
    (SORTIE_FIG / "chiffres.json").write_text(json.dumps(chiffres, ensure_ascii=False, indent=1, default=float),
                                             encoding="utf-8")
    print("chiffres : rapport/figures/chiffres.json")


if __name__ == "__main__":
    main()
