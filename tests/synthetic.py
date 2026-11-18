"""
Fausses données SEC, au même format que src/ingest.py, pour tester la pipeline sans les vrais fichiers
(et dans GitHub Actions, où les 1,3 Go de JSON ne sont pas disponibles).

Chaque entreprise dépose un 10-K par an : bilan de l'année + comparatif N-1, flux de l'année + N-1.
On y glisse les pièges trouvés dans l'EDA : valeur corrigée l'année suivante, bilan trimestriel
dans un 10-K, date future, négatif impossible, flux trimestriel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_long(n: int = 80, first: int = 2008, last: int = 2025, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    bilan, flux = [], []
    for i in range(n):
        cik = f"{i + 1:010d}"
        A, e, c = rng.uniform(5e7, 5e9), rng.uniform(-0.1, 0.7), rng.uniform(0.02, 0.4)
        hist = {}
        for y in range(first, last + 1):
            g = rng.normal(0.03, 0.25)
            A *= np.exp(g)
            e = float(np.clip(e + rng.normal(0, 0.12), -0.8, 0.95))
            c = float(np.clip(c * np.exp(rng.normal(0, 0.5)), 0.001, 0.9))
            E = A * e
            hist[y] = dict(
                Assets=A, StockholdersEquity=E, Liabilities=A - E,
                LiabilitiesAndStockholdersEquity=A,
                CashAndCashEquivalentsAtCarryingValue=A * c,
                AssetsCurrent=A * 0.45, LiabilitiesCurrent=A * rng.uniform(0.1, 0.5),
                RetainedEarningsAccumulatedDeficit=E * 0.5 - A * 0.1,
                PropertyPlantAndEquipmentNet=A * 0.3,
                NetIncomeLoss=A * (e * 0.1 + rng.normal(0, 0.05)), Revenues=A * 0.8,
                OperatingIncomeLoss=A * (e * 0.12 + rng.normal(0, 0.04)),
                NetCashProvidedByUsedInOperatingActivities=A * rng.normal(0.05, 0.05),
                NetCashProvidedByUsedInFinancingActivities=A * rng.normal(0, 0.05),
                InterestExpense=A * 0.01,
            )
        flux_cols = ["NetIncomeLoss", "Revenues", "OperatingIncomeLoss",
                     "NetCashProvidedByUsedInOperatingActivities",
                     "NetCashProvidedByUsedInFinancingActivities", "InterestExpense"]
        for y in range(first + 1, last + 1):                     # un 10-K par an
            filed = pd.Timestamp(f"{y + 1}-03-01")
            accn = f"{cik}-{y}"
            for yy in (y, y - 1):                                 # année + comparatif
                end = pd.Timestamp(f"{yy}-12-31")
                for k, v in hist[yy].items():
                    if k in flux_cols:
                        flux.append(dict(cik=cik, concept=k, val=v, start=pd.Timestamp(f"{yy}-01-01"),
                                         end=end, filed=filed, form="10-K", fy=y, fp="FY", accn=accn))
                    else:
                        corr = 1.001 if (yy == y - 1 and k == "Assets" and i % 7 == 0) else 1.0  # correction
                        bilan.append(dict(cik=cik, concept=k, val=v * corr, start=pd.NaT, end=end,
                                          filed=filed, form="10-K", fy=y, fp="FY", accn=accn))
            if i % 9 == 0:                                        # bilan trimestriel caché dans le 10-K
                bilan.append(dict(cik=cik, concept="Assets", val=hist[y]["Assets"] * 0.97, start=pd.NaT,
                                  end=pd.Timestamp(f"{y}-06-30"), filed=filed, form="10-K", fy=y, fp="FY", accn=accn))
            if i % 11 == 0:                                       # flux trimestriel (Q4 seul)
                flux.append(dict(cik=cik, concept="NetIncomeLoss", val=hist[y]["NetIncomeLoss"] / 4,
                                 start=pd.Timestamp(f"{y}-10-01"), end=pd.Timestamp(f"{y}-12-31"),
                                 filed=filed, form="10-K", fy=y, fp="FY", accn=accn))
        bilan.append(dict(cik=cik, concept="LongTermDebt", val=1e6, start=pd.NaT,      # date future
                          end=pd.Timestamp(f"{last + 2}-12-31"), filed=pd.Timestamp(f"{last}-03-01"),
                          form="10-K", fy=last, fp="FY", accn=f"{cik}-x"))
    b = pd.DataFrame(bilan)
    neg = b.index[(b["concept"] == "LiabilitiesCurrent")][:3]
    b.loc[neg, "val"] = -b.loc[neg, "val"]                         # négatifs impossibles
    return b, pd.DataFrame(flux)
