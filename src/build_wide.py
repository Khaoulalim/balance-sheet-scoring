"""
Étape 2 — Bilan : table longue -> une ligne par (entreprise, clôture annuelle), une colonne par poste.
Reprend notebooks/02_build_wide.ipynb.

Règles :
  1. 0 <= filed - end <= 550 j            (bilan de l'année + comparatif N-1 ; supprime dates futures/aberrantes)
  2. première valeur publiée par (cik, concept, end)   (point-in-time : pas de look-ahead bias)
  3. négatifs impossibles (actifs, dettes, trésorerie...) -> NaN, APRÈS la règle 2
  4. pivot wide, puis seulement les dates où Assets est déclaré
  5. seulement les clôtures annuelles (date max du rapport ou comparatif ~1 an avant) :
     supprime les bilans trimestriels publiés dans certains 10-K
  6. Liabilities manquant = (Liabilities + Equity ou Assets) - Equity   (identité comptable)

Usage : python -m src.build_wide
"""
from __future__ import annotations

import pandas as pd

from src import config

NON_NEGATIVE = [
    "Assets", "AssetsCurrent", "AssetsNoncurrent",
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "AccountsReceivableNetCurrent", "InventoryNet", "PrepaidExpenseCurrent",
    "PropertyPlantAndEquipmentNet", "Goodwill", "IntangibleAssetsNetExcludingGoodwill",
    "Liabilities", "LiabilitiesCurrent", "LiabilitiesNoncurrent",
    "AccountsPayableCurrent", "AccruedLiabilitiesCurrent",
    "DebtCurrent", "LongTermDebt", "LongTermDebtCurrent",
    "CommonStockValue", "LiabilitiesAndStockholdersEquity",
]


def first_reported(df: pd.DataFrame) -> pd.DataFrame:
    """Une valeur par (cik, concept, end) : celle du rapport déposé en premier."""
    df = df.sort_values(["cik", "concept", "end", "filed", "accn"])
    return df.drop_duplicates(["cik", "concept", "end"], keep="first")


def annual_dates(d: pd.DataFrame) -> pd.DataFrame:
    """(cik, end) qui sont des clôtures annuelles : la date la plus récente d'un rapport,
    ou son comparatif ~1 an avant. Calculé AVANT dédoublonnage (il faut voir tout le rapport)."""
    a = d[d["concept"] == "Assets"].copy()
    fin = a.groupby("accn")["end"].transform("max")
    ecart = (fin - a["end"]).dt.days
    ok = (ecart == 0) | ecart.between(*config.ANNUAL_GAP)
    return a.loc[ok, ["cik", "end"]].drop_duplicates()


def build_wide(long_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    gap = (long_df["filed"] - long_df["end"]).dt.days
    d = long_df[gap.between(0, config.MAX_GAP_DAYS)]
    annuelles = annual_dates(d)

    d = first_reported(d).copy()
    impossible = d["concept"].isin(NON_NEGATIVE) & (d["val"] < 0)
    d.loc[impossible, "val"] = float("nan")

    wide = d.pivot(index=["cik", "end"], columns="concept", values="val")
    wide.columns.name = None
    wide = wide.reset_index()
    n_pivot = len(wide)
    wide = wide[wide["Assets"].notna()]
    n_assets = len(wide)
    wide = wide.merge(annuelles, on=["cik", "end"], how="inner")

    for col in ["Liabilities", "LiabilitiesAndStockholdersEquity", "StockholdersEquity"]:
        if col not in wide:
            wide[col] = float("nan")
    manquant = wide["Liabilities"].isna()
    total = wide["LiabilitiesAndStockholdersEquity"].fillna(wide["Assets"])
    wide.loc[manquant, "Liabilities"] = total[manquant] - wide.loc[manquant, "StockholdersEquity"]
    wide.loc[wide["Liabilities"] < 0, "Liabilities"] = float("nan")
    wide["liabilities_imputed"] = manquant & wide["Liabilities"].notna()

    wide = wide.sort_values(["cik", "end"]).reset_index(drop=True)
    wide["days_since_prev"] = wide.groupby("cik")["end"].diff().dt.days

    if verbose:
        print(f"  lignes long : {len(long_df):,} -> filtre dates : {len(d):,} (après dédoublonnage)")
        print(f"  négatifs impossibles -> NaN : {int(impossible.sum()):,}")
        print(f"  bilans : pivot {n_pivot:,} -> avec Assets {n_assets:,} -> annuels {len(wide):,}"
              f" | {wide['cik'].nunique():,} entreprises | dettes reconstruites {wide['liabilities_imputed'].mean():.1%}")
    return wide


def main() -> None:
    wide = build_wide(pd.read_parquet(config.FACTS_LONG))
    config.BALANCE_WIDE.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(config.BALANCE_WIDE, index=False)
    print(f"  -> {config.BALANCE_WIDE}")


if __name__ == "__main__":
    main()
