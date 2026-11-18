"""
Étape 6 — Score de risque 0-100 pour chaque bilan (notebooks/08_score.ipynb).

  1. les 4 probabilités de dégradation (modèles de production)
  2. chaque probabilité -> rang dans l'année (0-100)
  3. moyenne pondérée des rangs (poids : config.POIDS), re-classée dans l'année
     -> score 95 = « plus risquée que 95 % des entreprises de la même année »
  4. niveau : Faible < 50 <= Modéré < 75 <= Élevé < 90 <= Très élevé
  Capitaux propres déjà négatifs : rang 100 sur la dimension « CP négatifs » (événement déjà survenu).

Vérification honnête : le score des années de test calculé avec les modèles d'ÉVALUATION (<= 2021)
-> taux réel de chaque dégradation par niveau (reports/score_par_niveau_test.csv).

Usage : python -m src.score [--src "E:/DSI/projet analytics edge"]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.events import add_split, evenements


def scorer(table: pd.DataFrame, modeles: dict) -> pd.DataFrame:
    out = table[["cik", "end"]].copy()
    out["annee"] = table["end"].dt.year
    for nom, m in modeles.items():
        out[f"p_{nom}"] = m.predict_proba(table[list(m.feature_names_in_)])[:, 1]
        out[f"rang_{nom}"] = out.groupby("annee")[f"p_{nom}"].rank(pct=True) * 100
    # Capitaux propres DÉJÀ négatifs : l'événement a déjà eu lieu (le modèle n'a appris que sur E > 0)
    # -> rang maximal sur cette dimension, au lieu d'une probabilité hors domaine
    out["CP_deja_negatifs"] = table["E_A"] < 0
    out.loc[out["CP_deja_negatifs"], "rang_CP_negatifs"] = 100.0
    brut = sum(w * out[f"rang_{nom}"] for nom, w in config.POIDS.items())
    out["score"] = brut.groupby(out["annee"]).rank(pct=True) * 100
    bornes, labels = config.NIVEAUX
    out["niveau"] = pd.cut(out["score"], bornes, right=False, labels=labels)
    return out


def verification(table: pd.DataFrame, eval_models: dict) -> pd.DataFrame:
    """Taux réel de chaque dégradation par niveau, sur le test, avec les modèles <= 2021."""
    table = add_split(table)
    te = table[table["split"] == "test"]
    s = scorer(te, eval_models)
    for nom, (evt, ok) in evenements(table).items():
        s[nom] = evt[te.index].astype(float).where(ok[te.index])
    verif = s.groupby("niveau", observed=True)[list(config.POIDS)].mean()
    verif.insert(0, "entreprises", s.groupby("niveau", observed=True).size())
    return verif.round(3)


def company_names(src: Path) -> dict[str, str]:
    """CIK -> nom, lu dans les 600 premiers caractères de chaque JSON (rapide)."""
    noms = {}
    if not src.is_dir():
        print(f"  [info] {src} introuvable : noms d'entreprises non disponibles")
        return noms
    for f in src.glob("CIK*.json"):
        with open(f, "rb") as fh:
            tete = fh.read(600).decode("utf-8", errors="ignore")
        m = re.search(r'"entityName"\s*:\s*"([^"]*)"', tete)
        if m:
            noms[f.stem[3:]] = m.group(1)
    return noms


def run(table: pd.DataFrame, prod_models: dict, eval_models: dict, src: Path,
        verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    verif = verification(table, eval_models)
    scores = scorer(table, prod_models)
    scores["nom"] = scores["cik"].map(company_names(src)).astype("object").fillna("(nom inconnu)")
    scores = scores.join(table[config.RATIOS_AFFICHES])
    if verbose:
        print("  taux réel par niveau (test 2022-2024, modèles <= 2021) :")
        print(verif.to_string())
    return scores, verif


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(config.SRC_JSON))
    a = p.parse_args()
    load = lambda kind: {n: joblib.load(config.MODELS / f"{kind}_{n}.joblib") for n in config.POIDS}
    scores, verif = run(pd.read_parquet(config.MODEL_TABLE), load("prod"), load("eval"), Path(a.src))
    scores.to_parquet(config.SCORES, index=False)
    verif.to_csv(config.REPORTS / "score_par_niveau_test.csv")
    print(f"  -> {config.SCORES}")


if __name__ == "__main__":
    main()
