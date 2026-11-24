# animations/

Les animations du projet, faites avec [Manim](https://www.manim.community/) (Community 0.19). Pour l'instant : les quatre
boucles du README. La présentation et le cours viendront ensuite dans `presentation/` et `cours/`.

```bash
uv run python animations/outils/exporter.py            # 1. données réelles depuis resultats/ -> animations/donnees/
py -3.13 animations/outils/gifs_readme.py              # 2. rend les boucles et écrit assets/readme/*.gif
py -3.13 animations/outils/gifs_readme.py --essai R2   #    une seule boucle, basse qualité, planche de contrôle
```

L'étape 1 utilise l'environnement du projet (elle rend les deux futurs A et B avec le code du jeu). L'étape 2 utilise le
Python qui porte Manim ; il lui faut aussi ffmpeg. Les planches de contrôle et les vidéos vont dans `rendus/` (ignoré par git).

| Dossier | Contenu |
|---|---|
| `commun/` | style (fond `#141414`, Segoe UI, une couleur par famille) et aides partagées |
| `scenes/readme.py` | les quatre boucles |
| `outils/` | export des données, rendu des GIF |

Origine des données : les images sont les vraies sorties des modèles (graine 0, p = 0,3), et les chiffres ceux du rapport ;
`exporter.py` vérifie que ses nombres redonnent ceux du rapport. Deux éléments sont des illustrations, signalées à l'écran :
le curseur qui fait glisser une image entre B et A, et le grain qu'on y ajoute (boucle « mesurer le flou »). La courbe du
débruiteur exact (boucle « un seul pas ») est un calcul, décrit dans l'annexe E du rapport.
