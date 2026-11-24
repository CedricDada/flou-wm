"""Quatre boucles courtes pour le README : peu de texte, fond sombre, retour à l'écran vide pour boucler.

    py -3.13 animations/outils/gifs_readme.py         # rend les quatre scènes et écrit les GIF dans assets/readme/

Données réelles : images produites par les modèles (graine 0, p = 0,3) et chiffres du rapport, exportés par
`outils/exporter.py`. Les seules illustrations calculées ici (le curseur de la boucle 2, le grain qu'on y ajoute)
sont signalées à l'écran.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "commun"))
from commun import *  # noqa: E402,F401,F403

N_ETATS = 16                                   # tirages consécutifs : les 16 premiers états du jeu de test


# ------------------------------------------------------------------------------------------------- 1
class R1_MelangeTirage(Boucle):
    """Le même état : la régression dessine le mélange, la diffusion tire A ou B, dans la bonne proportion."""

    def construct(self):
        p = CHIFFRES["p"]
        x_reg, x_dif, y_img = -3.3, 3.3, 0.35
        t_reg = etiquette("régression", REG, 28).move_to([x_reg, 2.75, 0])
        t_dif = etiquette("diffusion, 3 pas", DIF, 28).move_to([x_dif, 2.75, 0])
        t_fut = etiquette("futurs possibles", GRIS, 20).move_to([-7.0 + 0.95, 2.75, 0])
        t_ab = VGroup(etiquette("A", GRIS, 20), etiquette("B", GRIS, 20))

        def vignettes(i):
            a = image(D["a"][i], 1.55).move_to([-6.05, 1.25, 0])
            b = image(D["b"][i], 1.55).move_to([-6.05, -0.75, 0])
            return a, b

        def compteurs(k, n_a, n_b):
            n_floues = int((D["lab_reg"][:k] == 2).sum())
            gauche = etiquette(f"floues : {n_floues} / {k}", REG, 26).move_to([x_reg, -2.15, 0])
            droite = etiquette(f"A : {n_a}     B : {n_b}", DIF, 26).move_to([x_dif, -2.15, 0])
            ph = etiquette(f"part de A : {fr(n_a / max(1, n_a + n_b))}   (p = {fr(p, 1)})", GRIS, 22).move_to([x_dif, -2.75, 0])
            return VGroup(gauche, droite, ph)

        a, b = vignettes(0)
        t_ab[0].next_to(a, LEFT, buff=0.15)
        t_ab[1].next_to(b, LEFT, buff=0.15)
        reg = image(D["reg"][0], 3.4).move_to([x_reg, y_img, 0])
        dif = image(D["dif3"][0], 3.4).move_to([x_dif, y_img, 0])
        n_a = int(D["lab_dif3"][0] == 0)
        cpt = compteurs(1, n_a, 1 - n_a)
        self.play(FadeIn(t_reg), FadeIn(t_dif), FadeIn(t_fut), FadeIn(t_ab), FadeIn(a), FadeIn(b),
                  FadeIn(reg), FadeIn(dif), FadeIn(cpt), run_time=0.8)
        self.wait(1.4)
        n_b = 1 - n_a
        for i in range(1, N_ETATS):
            n_a += int(D["lab_dif3"][i] == 0)
            n_b += int(D["lab_dif3"][i] == 1)
            a2, b2 = vignettes(i)
            reg2 = image(D["reg"][i], 3.4).move_to(reg)
            dif2 = image(D["dif3"][i], 3.4).move_to(dif)
            cpt2 = compteurs(i + 1, n_a, n_b)
            self.remove(a, b, reg, dif, cpt)
            self.add(a2, b2, reg2, dif2, cpt2)
            a, b, reg, dif, cpt = a2, b2, reg2, dif2, cpt2
            self.wait(0.5 if i < 4 else 0.3)
        n_txt = f"{CHIFFRES['n_etats']:,}".replace(",", " ")
        fin = VGroup(
            etiquette(f"sur les {n_txt} états de test : {pct(CHIFFRES['reg_floue'])} floues", REG, 24),
            etiquette(f"sur les {n_txt} états de test : part de A = {fr(CHIFFRES['dif3_phat'])}", DIF, 24),
        )
        fin[0].move_to([x_reg, -2.45, 0])
        fin[1].move_to([x_dif, -2.45, 0])
        self.play(FadeOut(cpt), FadeIn(fin), run_time=0.7)
        self.finir(2.6)


# ------------------------------------------------------------------------------------------------- 2
class R2_MesurerFlou(Boucle):
    """Situer une prédiction entre B et A : la position α̂ dit nette ou floue, l'écart ρ dit « autre »."""

    def construct(self):
        i = 2                                                   # un état du jeu de test (le rail est bien visible)
        a = D["a"][i].astype(float)
        b = D["b"][i].astype(float)
        rng = np.random.default_rng(0)
        grain = rng.normal(0, 255, a.shape)                     # bruit d'exemple, ajouté seulement en fin de boucle
        delta = np.linalg.norm((a - b) / 255.0)
        alpha = ValueTracker(0.0)
        sig = ValueTracker(0.0)

        def pixels():
            x = alpha.get_value() * a + (1 - alpha.get_value()) * b + sig.get_value() * grain
            return np.clip(x, 0, 255).astype(np.uint8)

        im = image(pixels(), 3.2).move_to([-4.2, 0.55, 0])
        im[0].add_updater(lambda m: m.pixel_array.__setitem__(
            (slice(None), slice(None), slice(0, 3)), pixels()))
        t_im = etiquette("image prédite", GRIS, 22).next_to(im, UP, buff=0.15)

        # segment B -> A, avec ses trois zones
        x0, x1, y0 = 0.2, 7.0, -0.9
        def X(v):
            return x0 + v * (x1 - x0)
        zones = VGroup(
            Line([X(0), y0, 0], [X(0.2), y0, 0], color=NETTE, stroke_width=10),
            Line([X(0.2), y0, 0], [X(0.8), y0, 0], color=FLOUE, stroke_width=10),
            Line([X(0.8), y0, 0], [X(1), y0, 0], color=NETTE, stroke_width=10),
        )
        noms = VGroup(etiquette("nette-B", NETTE, 20).move_to([X(0.1), y0 - 0.45, 0]),
                      etiquette("floue", FLOUE, 20).move_to([X(0.5), y0 - 0.45, 0]),
                      etiquette("nette-A", NETTE, 20).move_to([X(0.9), y0 - 0.45, 0]))
        bouts = VGroup(etiquette("B", TEXTE, 26).next_to([X(0), y0, 0], LEFT, buff=0.25),
                       etiquette("A", TEXTE, 26).next_to([X(1), y0, 0], RIGHT, buff=0.25))
        # écart au segment : 1 unité verticale = rho de 0,5
        echelle = 2.0
        rho_max = CHIFFRES["rho_max"]
        seuil = DashedLine([X(0) - 0.1, y0 + echelle * rho_max, 0], [X(1) + 0.1, y0 + echelle * rho_max, 0],
                           color=AUTRE, stroke_width=1.6, dash_length=0.08)
        t_seuil = etiquette("au-delà : « autre »", AUTRE, 18).next_to(seuil, UP, buff=0.08).align_to(seuil, RIGHT)

        def rho():
            return sig.get_value() * np.sqrt(a.size) / delta

        point = Dot(radius=0.11, color=TEXTE)
        point.add_updater(lambda m: m.move_to([X(alpha.get_value()), y0 + echelle * rho(), 0]))
        # valeurs à virgule : 101 textes préparés une fois, un seul visible à la fois
        lab_pos = etiquette("position entre B et A :", GRIS, 22)
        valeurs = VGroup(*[Text(fr(k / 100), font_size=24, color=TEXTE) for k in range(101)])
        lab_alpha = VGroup(lab_pos, valeurs[0].copy()).arrange(RIGHT, buff=0.15).move_to([X(0.5), 2.3, 0])
        for v in valeurs:
            v.move_to(lab_alpha[1], aligned_edge=LEFT)
        lab_alpha.remove(lab_alpha[1])
        alpha_txt = valeurs
        alpha_txt.add_updater(lambda g: [m.set_opacity(1.0 if k == int(round(100 * alpha.get_value())) else 0.0)
                                        for k, m in enumerate(g)])
        lab_alpha.add(alpha_txt)

        self.play(FadeIn(im), FadeIn(t_im), Create(zones), FadeIn(noms), FadeIn(bouts), FadeIn(point),
                  FadeIn(lab_alpha), run_time=0.9)
        self.play(alpha.animate.set_value(1.0), run_time=3.0, rate_func=rate_functions.ease_in_out_sine)
        self.play(alpha.animate.set_value(CHIFFRES["p"]), run_time=1.6, rate_func=rate_functions.ease_in_out_sine)
        t_moy = etiquette("le mélange 0,3 A + 0,7 B : ce que vise la régression", REG, 20).next_to(lab_alpha, DOWN, buff=0.2)
        self.play(FadeIn(t_moy), run_time=0.4)
        self.wait(1.0)
        t_grain = etiquette("grain ajouté pour l'exemple", AUTRE, 18).next_to(im, DOWN, buff=0.15)
        self.play(FadeOut(t_moy), FadeIn(seuil), FadeIn(t_seuil), FadeIn(t_grain), run_time=0.5)
        self.play(sig.animate.set_value(0.035), run_time=2.0, rate_func=rate_functions.ease_in_out_sine)
        self.wait(0.8)
        self.play(sig.animate.set_value(0.0), alpha.animate.set_value(0.0), FadeOut(t_grain), run_time=1.4)
        for m in (im[0], point, alpha_txt):
            m.clear_updaters()
        self.finir(0.5)


# ------------------------------------------------------------------------------------------------- 3
class R3_UnPas(Boucle):
    """Un seul pas d'échantillonnage depuis le bruit σ : le réseau suit le débruiteur exact entre 5 et 10, pas au-delà."""

    def construct(self):
        ref = CHIFFRES["sigma_reference"]
        ax = Axes(x_range=[0, np.log10(80), 1], y_range=[0, 1, 0.25], x_length=9.0, y_length=4.4,
                  axis_config={"color": GRIS_FONCE, "stroke_width": 2, "include_ticks": False}).move_to([-2.2, -0.2, 0])
        xs = [1, 2, 5, 10, 20, 40, 80]
        ticks = VGroup(*[etiquette(str(v), GRIS, 18).next_to(ax.c2p(np.log10(v), 0), DOWN, buff=0.15) for v in xs])
        yt = VGroup(*[etiquette(f"{v} %", GRIS, 18).next_to(ax.c2p(0, v / 100), LEFT, buff=0.15) for v in (0, 50, 100)])
        tx = etiquette("bruit de départ σ (un seul pas)", GRIS, 20).next_to(ticks, DOWN, buff=0.2)
        ty = etiquette("images floues", GRIS, 20).next_to(ax, UP, buff=0.2).align_to(ax, LEFT)
        c_ref = DashedVMobject(ax.plot_line_graph([np.log10(s) for s in ref["sigma"]], ref["floue"], add_vertex_dots=False,
                                                  line_color=TEXTE, stroke_width=2.2)["line_graph"], num_dashes=60)
        l_ref = etiquette("débruiteur exact", TEXTE, 18).next_to(ax.c2p(np.log10(30), 0.8), DR, buff=0.1)

        curseur = ValueTracker(0.0)
        courbes = VGroup()
        for g in range(3):
            rows = CHIFFRES["sigma"][f"s{g}"]
            xv = [np.log10(r["sigma"]) for r in rows]
            yv = [r["floue"] for r in rows]
            courbes.add(ax.plot_line_graph(xv, yv, add_vertex_dots=False, line_color=DIF, stroke_width=3,
                                           stroke_opacity=0.55 + 0.2 * g)["line_graph"])
        l_res = etiquette("réseau entraîné (3 graines)", DIF, 18).next_to(ax.c2p(np.log10(5), 0.6), UL, buff=0.05)
        trait = always_redraw(lambda: DashedLine(ax.c2p(curseur.get_value(), 0), ax.c2p(curseur.get_value(), 1),
                                                 color=GRIS, stroke_width=1.2, dash_length=0.06))

        # exemples d'entraînement au-dessus du niveau σ (calcul sur la loi log-normale d'EDM)
        ex = CHIFFRES["exemples_au_dessus"]
        titre_ex = etiquette("exemples vus à l'entraînement", GRIS, 20).move_to([5.4, 1.6, 0])
        titre_ex2 = etiquette("au-dessus de ce bruit", GRIS, 20).next_to(titre_ex, DOWN, buff=0.1)
        valeurs = {s: etiquette(f"{n:,}".replace(",", " "), TEXTE, 44).move_to([5.4, 0.2, 0]) for s, n in
                   ((5, ex["5"]), (10, ex["10"]), (20, ex["20"]), (40, ex["40"]), (80, ex["80"]))}
        sur = etiquette("sur 2 227 200", GRIS, 18).move_to([5.4, -0.55, 0])

        self.play(Create(ax), FadeIn(ticks), FadeIn(yt), FadeIn(tx), FadeIn(ty), run_time=0.8)
        self.play(Create(c_ref), FadeIn(l_ref), run_time=1.2)
        self.add(trait)
        self.play(FadeIn(titre_ex), FadeIn(titre_ex2), FadeIn(sur), run_time=0.4)
        courant = None
        for s in (5, 10, 20, 40, 80):
            anims = [curseur.animate.set_value(np.log10(s))]
            if courant is None:
                anims += [Create(c) for c in courbes]
            self.play(*anims, run_time=1.6 if courant is None else 0.9, rate_func=linear)
            if courant is None:
                self.play(FadeIn(l_res), run_time=0.3)
            nouveau = valeurs[s]
            self.play(FadeOut(courant) if courant else Wait(0.01), FadeIn(nouveau), run_time=0.3)
            courant = nouveau
            self.wait(0.35)
        bande = Rectangle(width=ax.c2p(1, 0)[0] - ax.c2p(np.log10(5), 0)[0], height=4.4, stroke_width=0,
                          fill_color=DIF, fill_opacity=0.10)
        bande.move_to([(ax.c2p(np.log10(5), 0)[0] + ax.c2p(1, 0)[0]) / 2, ax.c2p(0, 0.5)[1], 0])
        trait.clear_updaters()
        l_bande = etiquette("bruit souvent vu à l'entraînement", DIF, 18).move_to([bande.get_center()[0], ax.c2p(0, 0.9)[1], 0])
        self.play(FadeOut(trait), FadeIn(bande), FadeIn(l_bande), run_time=0.6)
        self.finir(2.2)


# ------------------------------------------------------------------------------------------------- 4
class R4_SeizePas(Boucle):
    """Seize pas imaginés : la régression floute à chaque bifurcation, la diffusion suit une branche."""

    def construct(self):
        xs = (-4.6, 0.0, 4.6)
        noms = (("réalité", TEXTE), ("régression", REG), ("diffusion, 3 pas", DIF))
        bandes = (D["bande_reel"], D["bande_reg"], D["bande_dif"])
        titres = VGroup(*[etiquette(n, c, 26).move_to([x, 2.85, 0]) for x, (n, c) in zip(xs, noms)])

        def images(k):
            return Group(*[image(bd[k], 3.9).move_to([x, 0.45, 0]) for x, bd in zip(xs, bandes)])

        def pas(k):
            return etiquette(f"pas t + {k + 1}", GRIS, 24).move_to([0, -2.1, 0])

        bif = etiquette("bifurcation : l'ennemi choisit sa direction", FLOUE, 22).move_to([0, -2.7, 0])
        imgs, cpt = images(0), pas(0)
        self.play(FadeIn(titres), FadeIn(imgs), FadeIn(cpt), FadeIn(bif), run_time=0.7)
        self.wait(1.4)
        for k in range(1, 16):
            imgs2, cpt2 = images(k), pas(k)
            self.remove(imgs, cpt)
            self.add(imgs2, cpt2)
            if k in (6, 12):
                self.add(bif)
            elif bif in self.mobjects:
                self.remove(bif)
            imgs, cpt = imgs2, cpt2
            self.wait(1.1 if k in (6, 12) else 0.42)
        r = CHIFFRES["risque"]
        reg1, dif1 = r["A. Régression"]["doux_k1"], r["B. Diffusion (EDM) n=3"]["doux_k1"]
        reg5, dif5 = r["A. Régression"]["doux_k5"], r["B. Diffusion (EDM) n=3"]["doux_k5"]
        bilan = VGroup(
            etiquette("erreur sur le risque de contact", GRIS, 22),
            VGroup(etiquette("à 1 pas :", GRIS, 22), etiquette(f"régression {fr(reg1)}", REG, 22),
                   etiquette(f"diffusion {fr(dif1)}", DIF, 22)).arrange(RIGHT, buff=0.35),
            VGroup(etiquette("à 5 pas :", GRIS, 22), etiquette(f"régression {fr(reg5)}", REG, 22),
                   etiquette(f"diffusion {fr(dif5)}", DIF, 22)).arrange(RIGHT, buff=0.35),
        ).arrange(DOWN, buff=0.12).move_to([0, -2.45, 0])
        self.remove(cpt)
        if bif in self.mobjects:
            self.remove(bif)
        self.play(FadeIn(bilan), run_time=0.6)
        self.finir(3.0)
