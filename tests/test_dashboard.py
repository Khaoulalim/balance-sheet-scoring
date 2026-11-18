"""Tests des calculs du dashboard (app/logic.py), sans lancer Streamlit."""
import pandas as pd

from app.logic import annee_par_defaut, derniers_bilans, libelle_entreprise, points_forts, prepare, signaux


def _scores():
    return pd.DataFrame({
        "cik": ["0000000001", "0000000002", "0000000003", "0000000001"],
        "nom": ["Alpha", "Beta", "Gamma", "Alpha"],
        "end": pd.to_datetime(["2024-12-31", "2024-12-31", "2024-12-31", "2023-12-31"]),
        "annee": [2024, 2024, 2024, 2023],
        "score": [90.0, 50.0, 10.0, 70.0],
        "niveau": ["Très élevé", "Modéré", "Faible", "Modéré"],
        "E_A": [-0.5, 0.3, 0.8, 0.1],      # Alpha : capitaux propres négatifs
        "L_A": [1.5, 0.7, 0.2, 0.9],
        "ROA": [-0.4, 0.02, 0.15, -0.1],
    })


def test_rang_de_risque_dans_l_annee():
    df = prepare(_scores())
    a2024 = df[(df.cik == "0000000001") & (df.annee == 2024)].iloc[0]
    assert a2024["risque_E_A"] == 100        # E_A le plus faible de 2024 = le plus risqué
    assert a2024["risque_L_A"] == 100        # L_A le plus élevé de 2024 = le plus risqué
    assert df["niveau"].cat.ordered


def test_signaux_et_points_forts():
    df = prepare(_scores())
    alpha = df[(df.cik == "0000000001") & (df.annee == 2024)].iloc[0]
    gamma = df[(df.cik == "0000000003")].iloc[0]
    assert any("Capitaux propres" in s for s in signaux(alpha))
    assert signaux(gamma) == [] and len(points_forts(gamma, seuil=40)) > 0   # 3 entreprises : meilleur rang = 33 %


def test_derniers_bilans_et_libelles():
    df = prepare(_scores())
    assert len(derniers_bilans(df)) == 3
    assert libelle_entreprise(df)["0000000001"] == "Alpha (1)"
    assert annee_par_defaut(df, minimum=2) == 2024


def test_textes_cas_particuliers():
    from app.logic import bilan_ancien, resume_executif, texte_proba
    res = {"nom": "X", "end": pd.Timestamp("2015-12-31"), "score": 80.0, "niveau": "Élevé",
           "CP_deja_negatifs": True, "p_CP_negatifs": 0.5, "rang_CP_negatifs": 100.0,
           "p_perte_CP": 0.5, "rang_perte_CP": 90.0, "p_actifs_chute": 0.1, "rang_actifs_chute": 10.0,
           "p_treso_moitie": 0.2, "rang_treso_moitie": 60.0}
    assert texte_proba(res, "CP_negatifs") == "Déjà négatifs" and texte_proba(res, "perte_CP") == "50 %"
    assert "déjà négatifs" in resume_executif(res) and "perte de capitaux propres" in resume_executif(res)
    assert bilan_ancien(res, 2026) is not None and bilan_ancien({**res, "end": pd.Timestamp("2025-12-31")}, 2026) is None
