"""
Tests automatiques : python -m pytest
Lancés aussi à chaque git push par GitHub Actions (.github/workflows/tests.yml).
"""
import numpy as np
import pandas as pd
import pytest

from src.build_flows import build_flows
from src.build_wide import build_wide
from src.events import add_split, evenements
from src.features import build_model_table, feature_names, log_var, ratio
from tests.synthetic import make_long


@pytest.fixture(scope="module")
def donnees():
    bilan, flux = make_long()
    wide = build_wide(bilan, verbose=False)
    flows = build_flows(flux, verbose=False)
    return bilan, flux, wide, flows, build_model_table(wide, flows, verbose=False)


# ------------------------------------------------------------------ fonctions de base
def test_ratio_denominateur_nul_ou_negatif():
    r = ratio(pd.Series([5.0, 5.0, 5.0]), pd.Series([0.0, -2.0, 2.0]))
    assert np.isnan(r[0]) and np.isnan(r[1]) and r[2] == 2.5


def test_log_var_valeurs_non_positives():
    v = log_var(pd.Series([10.0, 0.0, 5.0]), pd.Series([5.0, 5.0, -1.0]))
    assert v[0] == pytest.approx(np.log(2)) and np.isnan(v[1]) and np.isnan(v[2])


# ------------------------------------------------------------------ règles de nettoyage
def test_bilan_point_in_time(donnees):
    bilan, _, wide, _, _ = donnees
    # l'entreprise 1 (i=0) a une valeur corrigée l'année suivante : on garde la PREMIÈRE publiée
    cik, end = "0000000001", pd.Timestamp("2015-12-31")
    premiere = bilan[(bilan.cik == cik) & (bilan.concept == "Assets") & (bilan.end == end)].sort_values("filed").val.iloc[0]
    assert wide.loc[(wide.cik == cik) & (wide.end == end), "Assets"].iloc[0] == premiere


def test_bilan_seulement_annuel_et_sans_date_future(donnees):
    _, _, wide, _, _ = donnees
    assert (wide["end"].dt.month == 12).all()                  # pas de bilan trimestriel (30 juin)
    assert wide["end"].max() <= pd.Timestamp("2025-12-31")     # pas de date future
    assert not wide.duplicated(["cik", "end"]).any()


def test_negatifs_impossibles(donnees):
    _, _, wide, _, _ = donnees
    assert (wide["LiabilitiesCurrent"].dropna() >= 0).all()
    assert (wide["StockholdersEquity"] < 0).any()               # les capitaux propres, eux, peuvent l'être


def test_flux_seulement_annuels(donnees):
    _, flux, _, flows, _ = donnees
    cik = "0000000001"                                          # i=0 a un flux trimestriel en plus
    ni = flows.loc[(flows.cik == cik) & (flows.end == pd.Timestamp("2015-12-31")), "NetIncomeLoss"].iloc[0]
    annuel = flux[(flux.cik == cik) & (flux.concept == "NetIncomeLoss") & (flux.end == pd.Timestamp("2015-12-31"))
                  & (flux.start == pd.Timestamp("2015-01-01"))].sort_values("filed").val.iloc[0]
    assert ni == annuel


# ------------------------------------------------------------------ ⭐ absence de fuite
def test_features_n_utilisent_pas_le_futur(donnees):
    """Effacer toutes les données après une date ne doit modifier AUCUNE feature."""
    _, _, wide, flows, complet = donnees
    coupure = pd.Timestamp("2018-12-31")
    tronque = build_model_table(wide[wide.end <= coupure], flows[flows.end <= coupure], verbose=False)
    feats = feature_names(complet)
    a = complet[complet.end <= coupure].set_index(["cik", "end"]).sort_index()[feats]
    t = tronque.set_index(["cik", "end"]).sort_index()[feats]
    pd.testing.assert_frame_equal(a, t)


def test_cibles_regardent_bien_le_futur(donnees):
    """À l'inverse, les cibles de la dernière année avant la coupure DOIVENT disparaître."""
    _, _, wide, flows, complet = donnees
    coupure = pd.Timestamp("2018-12-31")
    tronque = build_model_table(wide[wide.end <= coupure], flows[flows.end <= coupure], verbose=False)
    assert complet.loc[complet.end == coupure, "y_A"].notna().any()
    assert tronque.loc[tronque.end == coupure, "y_A"].isna().all()


def test_evenements_et_decoupage(donnees):
    *_, table = donnees
    t = add_split(table)
    assert set(t["split"]) <= {"train", "val", "test", "prediction"}
    assert t.loc[t.split == "train", "end"].dt.year.max() <= 2019
    for nom, (evt, ok) in evenements(t).items():
        assert ok.sum() > 0 and 0 < evt[ok].mean() < 1, nom


# ------------------------------------------------------------------ pipeline complète
def test_entrainement_et_score_bout_en_bout(donnees, tmp_path):
    from src.score import run as run_score
    from src.train import run as run_train
    *_, table = donnees
    metrics, eval_models, prod_models = run_train(table, verbose=False)
    assert len(metrics) == 4 and metrics["AUC modèle"].between(0, 1).all()
    scores, verif = run_score(table, prod_models, eval_models, tmp_path / "absent", verbose=False)
    assert scores["score"].between(0, 100).all()
    assert set(scores["niveau"].dropna()) <= {"Faible", "Modéré", "Élevé", "Très élevé"}
    assert (scores["nom"] == "(nom inconnu)").all()           # pas de JSON ici : noms absents, sans planter
    # chaque année : environ 10 % des entreprises en « Très élevé »
    part = scores.groupby("annee")["niveau"].apply(lambda s: (s == "Très élevé").mean())
    assert part.between(0.05, 0.2).all()
