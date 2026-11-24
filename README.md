# Où naît le flou ?

Un modèle du monde prédit l'image suivante d'un jeu à partir des images passées et de l'action jouée. Quand le futur est
incertain, un modèle entraîné à l'erreur quadratique dessine la moyenne des futurs possibles, donc une image floue, alors
qu'un modèle génératif est censé en tirer un seul. Sur un vrai jeu, on le constate sans pouvoir le mesurer, faute de
connaître la loi des futurs. Ce projet construit un jeu où on la connaît : un seul événement aléatoire, de probabilité `p`
réglable, dont les deux issues se calculent exactement. Trois familles de prédicteurs y sont comparées à budget égal :
une régression (U-Net), un modèle de diffusion (EDM) et un modèle à jetons discrets (VQ-VAE et Transformer), soit
25 entraînements et 46,6 heures de GPU T4. Le rapport complet est dans [`rapport/main.pdf`](rapport/main.pdf).

![La régression dessine le mélange, la diffusion tire une issue](assets/readme/melange_tirage.gif)

Le même état de test vu par deux modèles. L'ennemi rouge va partir à droite (A, probabilité 0,3) ou à gauche (B). La
régression dessine les deux positions à la fois, pondérées 0,3 et 0,7 ; la diffusion choisit l'une ou l'autre. Ce sont leurs
vraies sorties sur les 16 premiers états du jeu de test. Sur les 2 000 états, la régression est floue dans 98 % des cas et la
diffusion choisit A dans 33 % des images nettes.

## Mesurer le flou

![Situer une prédiction entre les deux futurs](assets/readme/mesurer_flou.gif)

Les deux images possibles A et B se calculent pour chaque état de test. Une prédiction est projetée sur le segment qui les
relie : sa position dit si elle est nette (près de A ou de B) ou floue (entre les deux), sa distance au segment dit si elle
n'est ni l'une ni l'autre (classe « autre » : objet déplacé, bruit). L'erreur quadratique de chaque modèle se compare aussi
à un minimum exact, `p(1−p)‖A−B‖²`, qu'atteint une régression idéale ; un générateur idéal fait le double.

## Résultats

| Question | Réponse | Chiffres |
|---|---|---|
| La régression produit-elle le mélange des futurs, et les modèles génératifs tranchent-ils ? | oui | régression floue dans 97,6 % des cas, à 1,01 fois l'erreur minimale ; diffusion à 3 pas nette dans 90,7 % |
| Un seul pas de diffusion produit-il la moyenne, comme l'observe DIAMOND sur *Boxing* ? | non | 0 % d'images floues, 100 % hors des deux futurs, même sur le contrôle sans hasard |
| Les issues sont-elles tirées avec la bonne fréquence ? | en moyenne seulement | part de A : 0,313 en moyenne pour p = 0,3, de 0,255 à 0,358 selon la graine ; 0,03 à 0,07 pour p = 0,1 |
| Un petit dictionnaire de jetons perd-il le petit objet ? | non tranché | rappel de 98 % pour tous les modèles : la mesure sature ; les autres erreurs baissent quand le dictionnaire grandit |
| Le flou change-t-il l'estimation d'un risque ? | oui, dans les deux sens | lecteur affine à un pas : erreur 0,03 pour la régression, 0,16 pour la diffusion ; lecteur à seuil : 0,30 contre 0,17 |

![Un seul pas d'échantillonnage depuis différents niveaux de bruit](assets/readme/un_pas.gif)

Pourquoi un seul pas de diffusion échoue. Partant d'un bruit de niveau σ, le réseau suit le débruiteur exact (en pointillés)
entre σ = 5 et 10, où l'entraînement l'a beaucoup exposé, puis décroche. L'échantillonneur à un pas part de σ = 80, un
niveau que le réseau n'a vu que 4 fois sur 2,2 millions d'exemples. Les très petits σ échouent pour une autre raison : le
point de départ y est trop loin des images bruitées vues à l'entraînement.

![Seize pas imaginés par chaque modèle](assets/readme/seize_pas.gif)

Chaque modèle enchaîne seize prédictions en recevant ses propres images. La régression floute l'ennemi à chaque bifurcation
(un pas sur six) ; la diffusion choisit une issue et la suit, ici l'autre que celle qui s'est réalisée. Pour estimer le risque
de toucher l'ennemi au pas suivant, l'image floue est la meilleure (erreur 0,03) ; cinq pas plus loin, nourrie de ses propres
mélanges, la régression fait moins bien que la diffusion (0,32 contre 0,23).

Limites principales : un seul jeu très simple, trois graines par comparaison (une seule pour les axes secondaires), et une
mesure du petit objet qui ne départage pas les modèles. Les autres sont dans le rapport.

## Le jeu

Une arène de 64 × 64 pixels vue de dessus. L'agent (bleu) ramasse un petit objet (jaune, +1) en évitant un ennemi (rouge,
−1) qui va et vient sur un rail horizontal ; chaque fois qu'il repasse au centre, soit un pas sur six, il part à droite avec
la probabilité `p`. C'est le seul hasard du jeu. Le carré gris clair est un obstacle mobile entièrement déterministe : il
sert de témoin, puisqu'aucun modèle n'a de raison de le prédire flou, et il peut se poser sur l'agent sans pénalité.

```bash
uv run scripts/play.py                    # jouer : flèches ou ZQSD, espace pour attendre, x pour quitter
uv run scripts/play.py --bot expert       # regarder un bot
```

## Installation

Il faut [uv](https://docs.astral.sh/uv/getting-started/installation/) et Python 3.10 ou plus récent.

```bash
uv sync                    # jeu, données, mesures, analyse et figures (NumPy, SciPy, Matplotlib)
uv sync --extra train      # ajoute PyTorch, pour entraîner et évaluer les modèles
uv run pytest              # 273 tests, quelques minutes
```

Sous Windows, préfixer les commandes par `PYTHONUTF8=1` (PowerShell : `$env:PYTHONUTF8=1`) : sans cela, les scripts qui
affichent des accents ou des lettres grecques s'arrêtent sur une erreur d'encodage.

## Reproduire les chiffres du rapport (quelques minutes, sans GPU)

Le dossier [`resultats/`](resultats/) contient, pour chacun des 25 entraînements, ses mesures d'évaluation, son journal
d'entraînement et les coordonnées de ses 2 000 prédictions de test, mais pas les poids des réseaux (16 Mo au lieu de
1,2 Go). Tout le rapport s'en déduit :

```bash
uv run scripts/analyze.py --runs resultats/runs --instrument resultats/analyse/instrument.json \
    --decision-instrument resultats/analyse/decision_instrument.json --out reports
uv run python rapport/make_figures.py      # figures et tableaux du rapport, depuis resultats/
cd rapport && latexmk -pdf main.tex
```

La première commande recalcule les réponses aux questions (`reports/verdicts.json`, `reports/report.md`) ; la deuxième
redonne à l'identique les figures et tableaux de `rapport/`.

## Tout recalculer (GPU)

```bash
make data                                              # données d'entraînement, validation et test (p = 0 ; 0,1 ; 0,3 ; 0,5)
uv run scripts/smoke_test.py --time                    # tout le code PyTorch sur de minuscules données, et temps par pas
uv run scripts/validate_instrument.py                  # seuil du classifieur et contrôle de l'instrument
uv run scripts/run_plan.py --steps 34800               # les 25 entraînements et leurs évaluations (environ 47 h sur un T4)
uv run scripts/sigma_sweep.py --all                    # diagnostic : un pas de diffusion depuis plusieurs niveaux de bruit
uv run scripts/filmstrip.py --run-id core-regression-p0.30-s0            # seize pas imaginés (figure du rapport)
uv run scripts/filmstrip.py --run-id core-diffusion-p0.30-s0 --n-steps 3
uv run scripts/validate_decision.py --games 20 --candidates 32   # contrôles de l'estimation du risque (sans GPU)
uv run scripts/evaluate_decision.py --all              # estimation du risque par les modèles (environ 1 h 20 sur un T4)
uv run scripts/analyze.py                              # réponses aux questions, dans reports/
uv run python rapport/make_figures.py --etat .         # figures du rapport depuis runs/ et reports/
```

`run_plan.py` reprend au dernier point de sauvegarde après une interruption et saute les runs terminés ; `--max-hours 10`
arrête proprement une session au bout de 10 heures. Les données se régénèrent à l'identique à partir de leurs graines.
Durées mesurées sur T4 : 2,3 à 2,7 h pour une régression ou une diffusion, environ 1 h pour un modèle à jetons. Un autre
GPU ou une autre version de PyTorch peut faire varier les derniers chiffres d'un run.

## Organisation

```
src/flou/          jeu, bots, données, modèles, entraînement, mesures et analyse
scripts/           les commandes ci-dessus
tests/             273 tests
resultats/         copie légère des résultats de la campagne
rapport/           le rapport (main.tex, main.pdf) et le script qui en refait les figures
animations/        les animations de ce README (Manim), voir animations/README.md
assets/readme/     les GIF de ce README
```

Dans le code et dans `verdicts.json`, les questions portent des noms internes hérités du protocole : `H1` pour la première,
`H2a` et `H2b` pour la deuxième et la troisième, `H3a` pour la quatrième, `H4a` et `H4b` pour la cinquième ; `H0a` à `H0f`
sont les contrôles.

## Licences

Aucune licence n'est choisie pour ce dépôt. Le gabarit LaTeX `sleek` (`rapport/packages/`) est sous licence MIT.
