"""
Étape 5 — Entraînement et évaluation des 4 modèles de dégradation.
Reprend notebooks/07_degradations.ipynb (évaluation) et 08_score.ipynb (modèles de production).

  a) modèles d'ÉVALUATION : appris sur <= 2021, évalués sur le test 2022-2024
     -> reports/metrics_test.csv  (AUC, AP, lift, comparaison à la meilleure règle simple)
  b) modèles de PRODUCTION : appris sur toutes les années avec cible (<= 2024)
     -> models/prod_<événement>.joblib, utilisés pour scorer les bilans récents

La configuration (features, réglages, règles) est figée dans src/config.py :
relancer cette étape redonne exactement les mêmes chiffres.

Usage : python -m src.train
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config
from src.events import LIBELLES, add_split, evenements
from src.features import feature_names


def make_clf(nom: str) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**config.BASE_PARAMS, **config.PARAMS[nom])


def top_k(y: np.ndarray, p: np.ndarray, q: float = 0.10) -> float:
    """Taux d'événements parmi les q % d'entreprises jugées les plus risquées."""
    return float(y[p >= np.quantile(p, 1 - q)].mean())


def fit(table: pd.DataFrame, splits: list[str]) -> dict[str, HistGradientBoostingClassifier]:
    feats = feature_names(table)
    modeles = {}
    for nom, (evt, ok) in evenements(table).items():
        tr = table[ok & table["split"].isin(splits)]
        utiles = [f for f in feats if tr[f].notna().any()]    # une colonne 100 % vide fait planter HGB
        clf = make_clf(nom)
        clf.fit(tr[utiles], evt[tr.index].astype(int))       # le modèle retient ses colonnes (feature_names_in_)
        modeles[nom] = clf
    return modeles


def evaluate(table: pd.DataFrame, modeles: dict, split: str = "test") -> pd.DataFrame:
    lignes = []
    for nom, (evt, ok) in evenements(table).items():
        ev = table[ok & (table["split"] == split)]
        y = evt[ev.index].astype(int).to_numpy()
        m = modeles[nom]
        p = m.predict_proba(ev[list(m.feature_names_in_)])[:, 1]
        f, sens = config.REGLE_VAL[nom]
        ref = table[ok & table["split"].isin(["train", "val"])][f].median()
        regle = sens * ev[f].fillna(ref)
        taux = top_k(y, p)
        lignes.append({
            "événement": LIBELLES[nom], "n": len(y), "fréquence": y.mean(),
            "AUC modèle": roc_auc_score(y, p), "AUC meilleure règle": roc_auc_score(y, regle),
            "AP modèle": average_precision_score(y, p),
            "taux top 10 %": taux, "lift top 10 %": taux / y.mean(),
        })
    return pd.DataFrame(lignes).set_index("événement").round(3)


def run(table: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    table = add_split(table)
    eval_models = fit(table, ["train", "val"])            # <= 2021
    metrics = evaluate(table, eval_models, "test")        # 2022-2024
    prod_models = fit(table, ["train", "val", "test"])    # <= 2024
    if verbose:
        print(metrics[["fréquence", "AUC modèle", "AUC meilleure règle", "lift top 10 %"]].to_string())
    return metrics, eval_models, prod_models


def main() -> None:
    metrics, eval_models, prod_models = run(pd.read_parquet(config.MODEL_TABLE))
    config.MODELS.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    for nom in eval_models:
        joblib.dump(eval_models[nom], config.MODELS / f"eval_{nom}.joblib")
        joblib.dump(prod_models[nom], config.MODELS / f"prod_{nom}.joblib")
    metrics.to_csv(config.REPORTS / "metrics_test.csv")
    print(f"  -> {config.REPORTS / 'metrics_test.csv'} | {config.MODELS}")


if __name__ == "__main__":
    main()
