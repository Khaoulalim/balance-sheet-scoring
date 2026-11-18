"""
Étape 4 — Features et cibles : bilan + flux -> table de modélisation.
Reprend notebooks/04_features.ipynb.

Règle d'or : une FEATURE n'utilise que l'année t (et t-1) ; une CIBLE porte sur t -> t+1.
Le test tests/test_fuite.py vérifie automatiquement cette règle.

Usage : python -m src.features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

NOMS = {
    "Assets": "A", "Liabilities": "L", "StockholdersEquity": "E",
    "CashAndCashEquivalentsAtCarryingValue": "C",
    "AssetsCurrent": "AC", "LiabilitiesCurrent": "LC",
    "RetainedEarningsAccumulatedDeficit": "RE",
    "PropertyPlantAndEquipmentNet": "PPE",
    "Goodwill": "GW", "IntangibleAssetsNetExcludingGoodwill": "INT",
}
NOMS_FLUX = {
    "CA": "CA", "NetIncomeLoss": "NI", "OperatingIncomeLoss": "OI",
    "NetCashProvidedByUsedInOperatingActivities": "CFO",
    "NetCashProvidedByUsedInInvestingActivities": "CFI",
    "NetCashProvidedByUsedInFinancingActivities": "CFF",
    "PaymentsToAcquirePropertyPlantAndEquipment": "CAPEX",
    "InterestExpense": "INTEXP",
    "PaymentsOfDividends": "DIV",
    "PaymentsForRepurchaseOfCommonStock": "RACHATS",
}
POSITIFS = ["A", "L", "AC", "LC"]   # toujours > 0 -> variation en log
SIGNES = ["C", "E"]                 # peuvent être 0 ou < 0 -> variation / actifs


def log_var(nouveau: pd.Series, ancien: pd.Series) -> pd.Series:
    """log(nouveau) - log(ancien), NaN si une valeur est <= 0."""
    return np.log(nouveau.where(nouveau > 0)) - np.log(ancien.where(ancien > 0))


def ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, NaN si le dénominateur est <= 0 (évite les inf)."""
    return num / den.where(den > 0)


def features_et_cibles(w: pd.DataFrame) -> pd.DataFrame:
    tous = {**NOMS, **NOMS_FLUX}
    w = w.copy()
    for col in tous:                                   # un poste jamais déclaré -> colonne vide
        if col not in w:
            w[col] = np.nan
    b = w[["cik", "end"] + list(tous)].rename(columns=tous)
    b = b.sort_values(["cik", "end"]).reset_index(drop=True)
    cols = list(tous.values())

    g = b.groupby("cik")
    suiv = g[["end"] + cols].shift(-1)                 # t+1 -> UNIQUEMENT pour les cibles
    prec = g[["end"] + cols].shift(1)                  # t-1 -> pour les features
    ok_suiv = (suiv["end"] - b["end"]).dt.days.between(*config.ANNUAL_GAP)
    ok_prec = (b["end"] - prec["end"]).dt.days.between(*config.ANNUAL_GAP)

    out = b[["cik", "end"]].copy()

    # --- features du bilan (année t)
    out["taille"] = np.log10(b["A"])
    out["L_A"] = ratio(b["L"], b["A"])
    out["E_A"] = ratio(b["E"], b["A"])
    out["C_A"] = ratio(b["C"], b["A"])
    out["RE_A"] = ratio(b["RE"], b["A"])
    out["AC_LC"] = ratio(b["AC"], b["LC"])
    out["AC_A"] = ratio(b["AC"], b["A"])
    out["LC_A"] = ratio(b["LC"], b["A"])
    out["PPE_A"] = ratio(b["PPE"], b["A"])
    out["incorp_A"] = ratio(b["GW"].fillna(0) + b["INT"].fillna(0), b["A"])
    out["E_negatif"] = (b["E"] < 0).astype(float).where(b["E"].notna())

    # --- features des flux (exercice qui se termine à la date du bilan t)
    out["ROA"] = ratio(b["NI"], b["A"])
    out["perte"] = (b["NI"] < 0).astype(float).where(b["NI"].notna())
    out["marge_op"] = ratio(b["OI"], b["CA"])
    out["rotation"] = ratio(b["CA"], b["A"])
    out["CFO_A"] = ratio(b["CFO"], b["A"])
    out["CFI_A"] = ratio(b["CFI"], b["A"])
    out["CFF_A"] = ratio(b["CFF"], b["A"])
    out["capex_A"] = ratio(b["CAPEX"], b["A"])
    out["distrib_A"] = ratio((b["DIV"].fillna(0) + b["RACHATS"].fillna(0)).where(b["CFF"].notna()), b["A"])
    out["cout_dette"] = ratio(b["INTEXP"], b["L"])
    out["couv_interets"] = ratio(b["OI"], b["INTEXP"])
    out["accruals_A"] = ratio(b["NI"] - b["CFO"], b["A"])
    out["d_CA"] = log_var(b["CA"], prec["CA"]).where(ok_prec)

    # --- variations passées (t-1 -> t)
    for k in POSITIFS:
        out[f"d_{k}"] = log_var(b[k], prec[k]).where(ok_prec)
    for k in SIGNES:
        out[f"d_{k}"] = ((b[k] - prec[k]) / prec["A"].where(prec["A"] > 0)).where(ok_prec)

    # --- cibles (t -> t+1)
    for k in POSITIFS:
        out[f"y_{k}"] = log_var(suiv[k], b[k]).where(ok_suiv)
    for k in SIGNES:
        out[f"y_{k}"] = ((suiv[k] - b[k]) / b["A"]).where(ok_suiv)
    return out


def feature_names(table: pd.DataFrame) -> list[str]:
    return [c for c in table if c not in ("cik", "end", "split") and not c.startswith("y_")]


def build_model_table(wide: pd.DataFrame, flows: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    w = wide[(wide["Assets"] >= config.MIN_ASSETS) & (wide["cik"] != config.INVALID_CIK)]
    w = w.merge(flows, on=["cik", "end"], how="left")
    table = features_et_cibles(w)
    if verbose:
        print(f"  {len(table):,} bilans | {table['cik'].nunique():,} entreprises"
              f" | {len(feature_names(table))} features | paires avec cible y_A : {table['y_A'].notna().sum():,}")
    return table


def main() -> None:
    table = build_model_table(pd.read_parquet(config.BALANCE_WIDE), pd.read_parquet(config.FLOWS_WIDE))
    config.MODEL_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(config.MODEL_TABLE, index=False)
    print(f"  -> {config.MODEL_TABLE}")


if __name__ == "__main__":
    main()
