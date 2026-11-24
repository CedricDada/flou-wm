# Rapport d'analyse

25 runs évalués, 43 variantes d'échantillonnage.

## Verdicts

| Hypothèse | Énoncé | Statut | Règle / raison |
|---|---|---|---|
| H0a | Instrument : sans entraînement, α̂ exact et images pures bien classées | **réussi** |  |
| H0b | Régression au plancher de Bayes (rapport MSE/plancher dans [1,00 ; 1,10], sans fuite) | **réussi** | sous-apprentissage si > 1,10 ; fuite d'information si le rapport tiré est nettement < 1 |
| H0c | Régresseur oracle (reçoit la branche) : moins de 5 % de flou | **réussi** |  |
| H0d | Régression à p = 0 : moins de 5 % de flou | **réussi** |  |
| H0d-diffusion(n=1) | Diffusion à p = 0, n = 1 : moins de 5 % de flou | **réussi** |  |
| H1 | Régression floue, diffusion (n = 3) nette | **confirmée** | confirmée si flou_reg > 50 % et nettes_diff(n=3) > 90 % ; réfutée si flou_reg < 30 % ou flou_diff(n=3) > 10 % |
| H2a | n = 1 plus floue que n = 3 d'au moins 10 points (comparaison appariée) | **réfutée** | confirmée si écart >= 10 points et > 4,42 s_d (Bonferroni, k = 3) ; réfutée si écart < 10 points |
| H2b | Calibration à n = 3 : \|p̂ − p\| < 0,05 à chaque graine | **non concluante** | confirmée si \|p̂-p\| < 0,05 pour chaque graine avec >= 1000 images nettes ; réfutée si >= 0,05 pour chaque graine |
| H3a | Le rappel du petit objet croît avec K (K = 64 contre K = 1024) | **non concluante** | confirmée si rappel(64) < rappel(1024) de plus de 3,23 écarts-types poolés (Bonferroni, k = 3) |
| H3b | Rappel ou MSE-pixel comme prédicteur de la qualité à l'horizon 16 (exploratoire) | **exploratoire (sans seuil)** |  |
| H0e | Lecteur de contact : sur de vraies images, il retrouve l'événement du jeu (accord >= 99 %) | **réussi** |  |
| H0f | Planificateur avec le vrai jeu comme modèle : <= 0,5 contact par partie et >= 3 points de plus que le bot aléatoire | **réussi** |  |
| H4a | Lecteur à seuil, 1 pas : la régression estime moins bien le risque que la diffusion (n = 3) | **confirmée** | confirmée si écart >= 0.10 et > 2,27 écarts-types poolés ; réfutée si écart < 0.10 |
| H4b | Lecteur doux, 1 pas : la régression estime le risque au moins aussi bien que la diffusion (n = 3) | **confirmée** | confirmée si E(régression) <= E(diffusion) + 0.02 ; réfutée si > + 0.05 ; entre les deux : non concluante |
| H4c | Erreur d'estimation du coût de contact selon l'horizon k (exploratoire) | **exploratoire (sans seuil)** |  |
| H5a | Planification : le modèle flou (régression) subit plus de contacts que le modèle net (diffusion n = 3) | **non évaluable** | moins de 2 graines de régression ou de diffusion (n = 3) évaluées au niveau 2 |
| H5b | Score du planificateur contre qualité d'image (exploratoire) | **non évaluable** | moins de 4 runs du cœur avec à la fois eval.json et planning.json |

## Types d'erreur par modèle (taxonomie du dossier, §1.9)

| modèle | graines | flou | autre | effondrement_de_mode | perte_petit_objet | chute_du_suivi_1_16 | excès_blocs | hors_palette | mse_sur_plancher | heures_gpu |
|---|---|---|---|---|---|---|---|---|---|---|
| A. Régression | 3 | 0.976 | 0.014 | 0.667 | 0.012 | -0.001 | 0.000 | 0.017 | 1.013 | 2.537 |
| B. Diffusion (EDM) n=1 | 3 | 0.000 | 1.000 | 0.000 | 0.014 | 0.233 | 0.001 | 0.014 | 2.117 | 2.556 |
| B. Diffusion (EDM) n=20 | 3 | 0.000 | 0.035 | 0.000 | 0.012 | 0.135 | 0.000 | 0.000 | 2.058 | 2.556 |
| B. Diffusion (EDM) n=3 | 3 | 0.045 | 0.048 | 0.000 | 0.012 | 0.189 | 0.000 | 0.001 | 2.018 | 2.556 |
| B. Diffusion (EDM) n=5 | 3 | 0.003 | 0.043 | 0.000 | 0.012 | 0.156 | 0.000 | 0.000 | 1.974 | 2.556 |
| C. Jetons K=1024 | 3 | 0.000 | 0.325 | 0.000 | 0.016 | 0.732 | 0.000 | 0.001 | 2.840 | 1.039 |
| C. Jetons K=256 | 3 | 0.000 | 0.532 | 0.000 | 0.018 | 0.856 | 0.000 | 0.001 | 3.461 | 0.958 |
| C. Jetons K=256 grille 16 | 1 | 0.000 | 0.392 | 0.000 | 0.018 | 0.748 | 0.000 | 0.001 | 3.152 | 3.165 |
| C. Jetons K=64 | 3 | 0.000 | 0.633 | 0.000 | 0.019 | 0.907 | 0.000 | 0.001 | 3.508 | 1.012 |
| Régresseur oracle | 1 | 0.000 | 0.013 | 0.000 | 0.012 | nan | -0.000 | 0.000 | 1.980 | 2.333 |

## Calibration de la diffusion selon le nombre de pas n

| n | graines | p_hat_moyen | erreur_absolue_moyenne | flou | nettes |
|---|---|---|---|---|---|
| 1 | 3 | nan | nan | 0.000 | 0.000 |
| 3 | 3 | 0.313 | 0.043 | 0.045 | 0.907 |
| 5 | 3 | 0.265 | 0.035 | 0.003 | 0.954 |
| 20 | 3 | 0.302 | 0.035 | 0.000 | 0.965 |

## Calcul consommé

Total : 46.6 h de GPU sur 25 runs.

| run | groupe | paramètres | heures_gpu |
|---|---|---|---|
| control-diffusion-p0.00-s0 | control | 6005651 | 2.503 |
| control-oracle-p0.30-s0 | control | 5873635 | 2.333 |
| control-regression-p0.00-s0 | control | 5872771 | 2.374 |
| core-diffusion-p0.30-s0 | core | 6005651 | 2.643 |
| core-diffusion-p0.30-s1 | core | 6005651 | 2.655 |
| core-diffusion-p0.30-s2 | core | 6005651 | 2.369 |
| core-regression-p0.30-s0 | core | 5872771 | 2.674 |
| core-regression-p0.30-s1 | core | 5872771 | 2.586 |
| core-regression-p0.30-s2 | core | 5872771 | 2.351 |
| core-tokens-K256-p0.30-s0 | core | 6043715 | 1.001 |
| core-tokens-K256-p0.30-s1 | core | 6043715 | 1.036 |
| core-tokens-K256-p0.30-s2 | core | 6043715 | 0.838 |
| grid-tokens-K256-g16-p0.30-s0 | grid | 6128835 | 3.165 |
| kaxis-tokens-K1024-p0.30-s0 | k_axis | 6585155 | 1.039 |
| kaxis-tokens-K1024-p0.30-s1 | k_axis | 6585155 | 1.039 |
| kaxis-tokens-K1024-p0.30-s2 | k_axis | 6585155 | 1.039 |
| kaxis-tokens-K64-p0.30-s0 | k_axis | 5908355 | 1.012 |
| kaxis-tokens-K64-p0.30-s1 | k_axis | 5908355 | 1.011 |
| kaxis-tokens-K64-p0.30-s2 | k_axis | 5908355 | 1.012 |
| paxis-diffusion-p0.10-s0 | p_axis | 6005651 | 2.525 |
| paxis-diffusion-p0.50-s0 | p_axis | 6005651 | 2.568 |
| paxis-regression-p0.10-s0 | p_axis | 5872771 | 2.482 |
| paxis-regression-p0.50-s0 | p_axis | 5872771 | 2.488 |
| paxis-tokens-K256-p0.10-s0 | p_axis | 6043715 | 0.936 |
| paxis-tokens-K256-p0.50-s0 | p_axis | 6043715 | 0.971 |

## Extension « décision » (H4, H5)

Hypothèses, seuils et articles qui les guident : `docs/EXTENSION_DECISION.md`.

Erreur attendue si les modèles étaient parfaits (prédicteurs factices, vrai jeu) :

| prédicteur factice | doux_k1 | seuil_k1 | doux_k3 | seuil_k3 | doux_k5 | seuil_k5 | théorie_k1_seuil |
|---|---|---|---|---|---|---|---|
| régression idéale (image moyenne) | 0.001 | 0.300 | 0.001 | 0.345 | 0.002 | 0.447 | 0.300 |
| échantillonneur net calibré, m = 8 | 0.132 | 0.132 | 0.157 | 0.157 | 0.206 | 0.206 | 0.133 |
| échantillonneur net, biais d'Euler n = 3 (q = 0.188), m = 8 | 0.151 | 0.151 | 0.175 | 0.175 | 0.224 | 0.224 | 0.152 |

Niveau 1 — erreur d'estimation du coût de contact, états à risque ambigu :

| modèle | graines | tirages_m | doux_k1 | seuil_k1 | doux_k3 | seuil_k3 | doux_k5 | seuil_k5 |
|---|---|---|---|---|---|---|---|---|
| A. Régression | 3 | 1 | 0.031 | 0.302 | 0.166 | 0.347 | 0.321 | 0.450 |
| B. Diffusion (EDM) n=1 | 3 | 8 | 0.127 | 0.180 | 0.171 | 0.204 | 0.226 | 0.248 |
| B. Diffusion (EDM) n=3 | 3 | 8 | 0.159 | 0.167 | 0.183 | 0.189 | 0.232 | 0.237 |
| C. Jetons K=256 | 3 | 8 | 0.221 | 0.224 | 0.252 | 0.259 | 0.337 | 0.343 |

Niveau 2 — scores dans le vrai jeu (mêmes parties pour tous) :

| joueur | graines | parties | score | contacts_par_partie | ramassages_par_partie |
|---|---|---|---|---|---|
| bot aléatoire | 1 | 20 | -4.10 | 4.15 | 0.05 |
| bot expert | 1 | 20 | 1.75 | 0.00 | 1.75 |
| planificateur, vrai jeu (H0f) | 1 | 20 | 0.80 | 0.30 | 1.10 |

## Figures

- fig_flou.png
- fig_flou_selon_p.png
- fig_calibration.png
- fig_rappel.png
- fig_rollout.png
- fig_mse_plancher.png

## Limites

- 3 graines par cellule du cœur : un écart sous le seuil est « non concluant », jamais « pas d'effet » ; les points bruts sont donnés dans verdicts.json.
- Les seuils sont des règles de décision à très peu de graines, pas des preuves. Le cœur compte k = 3 comparaisons simultanées fixées avant les runs (H1, H2a, H3a ; docs/DECISIONS.md, C4) : 3,23 s non apparié (H3a), 4,42 s_d apparié (H2a) ; l'extension (H4a, H5a) reste à 2,27 s.
- ρmax est réglé sur un bruit synthétique : la part « autre » dépend de ce choix (sensibilité × 0,5 / × 1 / × 2 dans instrument.json).
- Les axes p et la grille 16 × 16 n'ont qu'une graine : exploratoire.
- Aucune relecture humaine n'est incluse : la validité du classifieur s'estime avec review.csv (scripts/human_review.py).
- Extension « décision » : la récompense est lue dans les pixels (supposée connue), le planificateur est un tirage simple de suites d'actions ; un autre planificateur ou une récompense apprise pourraient changer les conclusions.
- H5b porte sur 6 à 9 points : Lambert et al. (arXiv:2002.04523) trouvent des corrélations de -0,06 à 0,59 avec plus de 1 000 modèles ; une corrélation de rang sur si peu de points est indicative, rien de plus.
