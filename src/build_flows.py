"""
Étape 3 — Flux (compte de résultat + flux de trésorerie) : table longue -> une ligne par (entreprise, exercice).
Reprend notebooks/06_flux.ipynb.

Règles :
  1. durée end - start entre 350 et 380 j   (flux annuels uniquement ; exclut trimestres et 9 mois)
  2. 0 <= filed - end <= 550 j              (exercice + comparatif N-1)
  3. première valeur publiée par (cik, concept, end)
  4. pivot wide
  5. chiffre d'affaires CA = première étiquette disponible parmi 3 (Revenues, ASC 606, ancienne)

Usage : python -m src.build_flows
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.build_wide import first_reported

CA_ETIQUETTES = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]


def build_flows(long_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    duree = (long_df["end"] - long_df["start"]).dt.days
    ecart = (long_df["filed"] - long_df["end"]).dt.days
    d = long_df[duree.between(*config.FLOW_DURATION) & ecart.between(0, config.MAX_GAP_DAYS)]
    d = first_reported(d)

    fw = d.pivot(index=["cik", "end"], columns="concept", values="val")
    fw.columns.name = None
    fw = fw.reset_index()

    fw["CA"] = float("nan")
    for c in CA_ETIQUETTES:
        if c in fw:
            fw["CA"] = fw["CA"].fillna(fw[c])

    if verbose:
        print(f"  lignes long : {len(long_df):,} -> annuelles dédoublonnées {len(d):,}"
              f" -> {len(fw):,} exercices | CA renseigné {fw['CA'].notna().mean():.1%}")
    return fw


def main() -> None:
    fw = build_flows(pd.read_parquet(config.FLOWS_LONG))
    fw.to_parquet(config.FLOWS_WIDE, index=False)
    print(f"  -> {config.FLOWS_WIDE}")


if __name__ == "__main__":
    main()
