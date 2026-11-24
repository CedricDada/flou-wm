# resultats/

Copie légère des résultats de la campagne (25 entraînements, p = 0 ; 0,1 ; 0,3 ; 0,5), sans les poids des réseaux.
C'est la source de tous les chiffres et figures du rapport.

- `runs/<run>/` : la configuration (`spec.json`, `train.json`), le coût (`compute.json`), le journal d'entraînement
  (`log.jsonl`), l'évaluation sur les 2 000 états de test (`eval.json`), les coordonnées de chaque prédiction de test
  (`eval_arrays.npz`), et selon le run l'estimation du risque (`decision.json`) et le balayage du bruit (`sigma_sweep.json`).
  Quelques planches d'images servent aux figures du rapport.
- `analyse/` : les sorties de l'analyse (réponses aux questions, tableaux, seuil du classifieur).
- `essai_100000_transitions/` : le premier entraînement, écarté pour mémorisation (rapport, annexe A).

Le nom d'un run donne sa famille, sa probabilité p et sa graine, par exemple `core-diffusion-p0.30-s1`.
