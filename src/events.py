"""
Les 4 événements de dégradation prédits (notebooks/07_degradations.ipynb) et le découpage temporel.
Chaque événement porte sur l'année t -> t+1 ; il n'est défini que sur les lignes où il est mesurable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

LIBELLES = {
    "CP_negatifs":  "Capitaux propres deviennent négatifs",
    "perte_CP":     "Capitaux propres : perte > 10 % des actifs",
    "actifs_chute": "Actifs en baisse de plus de 20 %",
    "treso_moitie": "Trésorerie divisée par 2 ou plus",
}


def evenements(t: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """{nom court: (événement True/False, lignes où il est mesurable)}"""
    return {
        "perte_CP":     (t["y_E"] < -0.10,               t["y_E"].notna()),
        "treso_moitie": (t["y_C"] < -0.5 * t["C_A"],     t["y_C"].notna() & (t["C_A"] > 0)),
        "actifs_chute": (t["y_A"] < np.log(0.8),         t["y_A"].notna()),
        "CP_negatifs":  (t["E_A"] + t["y_E"] < 0,        t["y_E"].notna() & (t["E_A"] > 0)),
    }


def add_split(t: pd.DataFrame) -> pd.DataFrame:
    """train <= 2019, val 2020-2021, test 2022-2024, prediction ensuite (année du bilan t)."""
    t = t.copy()
    annee = t["end"].dt.year
    t["split"] = np.select(
        [annee <= config.TRAIN_END, annee <= config.VAL_END, annee <= config.TEST_END],
        ["train", "val", "test"], default="prediction")
    return t
