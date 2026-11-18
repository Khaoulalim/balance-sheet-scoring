"""
Calculs du dashboard, séparés de l'interface pour pouvoir les tester (tests/test_dashboard.py).
"""
from __future__ import annotations

import pandas as pd

NIVEAUX = ["Faible", "Modéré", "Élevé", "Très élevé"]

EVENEMENTS = {   # nom court -> libellé affiché
    "CP_negatifs":  "Capitaux propres deviennent négatifs",
    "perte_CP":     "Perte de capitaux propres > 10 % des actifs",
    "actifs_chute": "Actifs en baisse de plus de 20 %",
    "treso_moitie": "Trésorerie divisée par 2 ou plus",
}

# ratio -> (libellé, sens du risque : -1 = une valeur FAIBLE est risquée, +1 = une valeur ÉLEVÉE est risquée)
RATIOS = {
    "E_A":           ("Capitaux propres / actifs", -1),
    "L_A":           ("Dettes / actifs", 1),
    "RE_A":          ("Bénéfices cumulés / actifs", -1),
    "ROA":           ("Rentabilité des actifs (ROA)", -1),
    "marge_op":      ("Marge d'exploitation", -1),
    "CFO_A":         ("Cash généré par l'activité / actifs", -1),
    "C_A":           ("Trésorerie / actifs", -1),
    "AC_LC":         ("Liquidité générale", -1),
    "couv_interets": ("Couverture des intérêts", -1),
    "d_CA":          ("Croissance du chiffre d'affaires", -1),
    "taille":        ("Taille (log10 des actifs)", -1),
}


def prepare(scores: pd.DataFrame) -> pd.DataFrame:
    """Niveau ordonné + rang de risque de chaque ratio dans son année (100 = le plus risqué)."""
    df = scores.copy()
    df["niveau"] = pd.Categorical(df["niveau"].astype("object"), categories=NIVEAUX, ordered=True)
    for r, (_, sens) in RATIOS.items():
        if r in df:
            df[f"risque_{r}"] = (sens * df[r]).groupby(df["annee"]).rank(pct=True) * 100
    return df


def derniers_bilans(df: pd.DataFrame) -> pd.DataFrame:
    """Le bilan le plus récent de chaque entreprise."""
    return df.sort_values("end").groupby("cik", as_index=False).tail(1)


def annee_par_defaut(df: pd.DataFrame, minimum: int = 1000) -> int:
    """La dernière année assez complète (l'année en cours n'a que les entreprises qui clôturent tôt)."""
    n = df.groupby("annee").size()
    completes = n[n >= minimum]
    return int(completes.index.max() if len(completes) else n.index.max())


def signaux(row: pd.Series, k: int = 3, seuil: float = 80) -> list[str]:
    """Les k ratios qui placent l'entreprise parmi les plus risquées de son année."""
    items = []
    for r, (libelle, _) in RATIOS.items():
        rang = row.get(f"risque_{r}")
        if pd.notna(rang) and rang >= seuil:
            items.append((rang, f"{libelle} : plus défavorable que {rang:.0f} % des entreprises"))
    return [texte for _, texte in sorted(items, reverse=True)[:k]]


def points_forts(row: pd.Series, k: int = 3, seuil: float = 20) -> list[str]:
    """Les k ratios qui placent l'entreprise parmi les plus solides de son année."""
    items = []
    for r, (libelle, _) in RATIOS.items():
        rang = row.get(f"risque_{r}")
        if pd.notna(rang) and rang <= seuil:
            items.append((rang, f"{libelle} : meilleur que {100 - rang:.0f} % des entreprises"))
    return [texte for _, texte in sorted(items)[:k]]


def libelle_entreprise(df: pd.DataFrame) -> dict[str, str]:
    """cik -> « Nom (CIK) », pour la liste de recherche."""
    d = derniers_bilans(df)
    return dict(zip(d["cik"], d["nom"] + " (" + d["cik"].str.lstrip("0") + ")"))
