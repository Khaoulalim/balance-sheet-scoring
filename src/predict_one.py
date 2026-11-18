"""
Inférence sur UNE entreprise : un fichier JSON SEC (companyfacts) importé dans le dashboard.

Même chaîne que la pipeline, appliquée à un seul fichier :
  parse (bilan + flux) -> build_wide -> build_flows -> features -> 4 probabilités (modèles de production)
puis positionnement par rapport à TOUTES les entreprises de la même année (data/processed/scores.parquet) :
rang de chaque probabilité, score 0-100, niveau, rang de risque de chaque ratio.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.build_flows import build_flows
from src.build_wide import build_wide
from src.concepts import BALANCE_SHEET_CONCEPTS, FLOW_CONCEPTS
from src.features import build_model_table
from src.ingest import _to_frame, parse_company

FORMS = {"10-K", "10-K/A"}
FAITS = {
    "Assets": "Total actif", "Liabilities": "Total dettes", "StockholdersEquity": "Capitaux propres",
    "AssetsCurrent": "Actif courant", "LiabilitiesCurrent": "Dettes courantes",
    "CashAndCashEquivalentsAtCarryingValue": "Trésorerie",
}


# Poste -> étiquette de repli quand l'entreprise utilise un autre concept XBRL pour la même notion
REPLI = {"CashAndCashEquivalentsAtCarryingValue": "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"}
COLONNES_FAITS = list(FAITS) + list(REPLI.values())


def faits_bilan(ligne) -> dict:
    """Chiffres affichés d'un bilan (dict ou Series), avec repli sur l'étiquette alternative."""
    out = {}
    for col, lib in FAITS.items():
        v = ligne.get(col, np.nan)
        if pd.isna(v) and col in REPLI:
            v = ligne.get(REPLI[col], np.nan)
        out[lib] = v
    return out


class AnalyseImpossible(ValueError):
    """Le fichier ne permet pas d'analyse (pas de 10-K exploitable, actifs < 1 M$...)."""


def lire_json(raw: bytes) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    data = json.loads(raw)
    nom = data.get("entityName") or "(nom inconnu)"
    cik = str(data.get("cik") or "").zfill(10)
    if not cik.strip("0"):
        cik = "9999999999"                      # fichier sans CIK : identifiant local
    bilan = _to_frame(parse_company(raw, set(BALANCE_SHEET_CONCEPTS), FORMS, cik))
    flux = _to_frame(parse_company(raw, set(FLOW_CONCEPTS), FORMS, cik))
    if bilan.empty:
        raise AnalyseImpossible("Aucun poste de bilan en USD issu d'un rapport annuel 10-K dans ce fichier.")
    return bilan, flux, nom


def rang_dans(reference: pd.Series, valeur: float) -> float:
    """Part (0-100) des valeurs de référence strictement inférieures à la valeur."""
    ref = reference.dropna()
    if pd.isna(valeur) or ref.empty:
        return float("nan")
    return float((ref < valeur).mean() * 100)


def analyser(raw: bytes, modeles: dict, reference: pd.DataFrame) -> dict:
    """reference = scores.parquet (toutes les entreprises), pour situer l'entreprise dans son année."""
    bilan, flux, nom = lire_json(raw)
    wide = build_wide(bilan, verbose=False)
    if wide.empty:
        raise AnalyseImpossible("Aucun bilan annuel exploitable (dates incohérentes ou total actif absent).")
    flows = build_flows(flux, verbose=False) if not flux.empty else pd.DataFrame(columns=["cik", "end"])
    table = build_model_table(wide, flows, verbose=False)
    if table.empty:
        raise AnalyseImpossible(f"Total actif inférieur à {config.MIN_ASSETS:,.0f} $ : hors du champ du modèle.")

    ligne = table.sort_values("end").iloc[[-1]]
    fin = ligne["end"].iloc[0]
    annee = fin.year if fin.year in set(reference["annee"]) else int(reference["annee"].max())
    ref = reference[reference["annee"] == annee]

    res = {"nom": nom, "cik": ligne["cik"].iloc[0], "end": fin, "annee_reference": annee,
           "n_reference": len(ref), "historique": len(table)}
    dernier_bilan = wide.sort_values("end").iloc[-1]
    res["faits"] = faits_bilan(dernier_bilan)

    res["CP_deja_negatifs"] = bool(ligne["E_A"].iloc[0] < 0)
    brut = 0.0
    for nom_evt, m in modeles.items():
        p = float(m.predict_proba(ligne[list(m.feature_names_in_)])[:, 1][0])
        res[f"p_{nom_evt}"] = p
        res[f"rang_{nom_evt}"] = rang_dans(ref[f"p_{nom_evt}"], p)
        if nom_evt == "CP_negatifs" and res["CP_deja_negatifs"]:
            res[f"rang_{nom_evt}"] = 100.0           # événement déjà survenu (même règle que src/score.py)
        brut += config.POIDS[nom_evt] * res[f"rang_{nom_evt}"]

    brut_ref = sum(w * ref[f"rang_{n}"] for n, w in config.POIDS.items())
    res["score"] = rang_dans(brut_ref, brut)
    bornes, labels = config.NIVEAUX
    res["niveau"] = str(pd.cut([res["score"]], bornes, right=False, labels=labels)[0])
    for r in config.RATIOS_AFFICHES:
        res[r] = float(ligne[r].iloc[0]) if r in ligne else np.nan
    return res
