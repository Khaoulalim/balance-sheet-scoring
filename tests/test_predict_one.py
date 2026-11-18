"""Analyse d'un JSON importé (src/predict_one.py), au format SEC companyfacts."""
import json

import pandas as pd
import pytest

from src.build_flows import build_flows
from src.build_wide import build_wide
from src.features import build_model_table
from src.predict_one import AnalyseImpossible, analyser
from src.score import scorer
from src.train import run as run_train
from tests.synthetic import make_long


def vers_json(bilan: pd.DataFrame, flux: pd.DataFrame, cik: str, nom: str) -> bytes:
    """Reconstruit un fichier companyfacts (facts -> us-gaap -> concept -> units -> USD) à partir des lignes."""
    facts = {}
    for df in (bilan, flux):
        for r in df[df.cik == cik].itertuples():
            fait = {"end": r.end.strftime("%Y-%m-%d"), "val": r.val, "accn": r.accn, "fy": r.fy,
                    "fp": r.fp, "form": r.form, "filed": r.filed.strftime("%Y-%m-%d")}
            if pd.notna(r.start):
                fait["start"] = r.start.strftime("%Y-%m-%d")
            facts.setdefault(r.concept, {"units": {"USD": []}})["units"]["USD"].append(fait)
    return json.dumps({"cik": int(cik), "entityName": nom, "facts": {"us-gaap": facts}}).encode()


@pytest.fixture(scope="module")
def contexte():
    bilan, flux = make_long()
    table = build_model_table(build_wide(bilan, verbose=False), build_flows(flux, verbose=False), verbose=False)
    _, _, prod = run_train(table, verbose=False)
    return bilan, flux, prod, scorer(table, prod)


def test_analyse_d_un_json(contexte):
    bilan, flux, prod, ref = contexte
    res = analyser(vers_json(bilan, flux, "0000000005", "Demo Corp"), prod, ref)
    assert res["nom"] == "Demo Corp" and res["end"] == pd.Timestamp("2025-12-31")
    assert 0 <= res["score"] <= 100 and res["niveau"] in {"Faible", "Modéré", "Élevé", "Très élevé"}
    assert res["faits"]["Total actif"] > 0
    # même entreprise, même année : mêmes probabilités que la pipeline
    ligne = ref[(ref.cik == "0000000005") & (ref.end == res["end"])].iloc[0]
    for e in prod:
        assert res[f"p_{e}"] == pytest.approx(ligne[f"p_{e}"])


def test_json_inexploitable(contexte):
    *_, prod, ref = contexte
    vide = json.dumps({"cik": 1, "entityName": "Vide", "facts": {"us-gaap": {}}}).encode()
    with pytest.raises(AnalyseImpossible):
        analyser(vide, prod, ref)
