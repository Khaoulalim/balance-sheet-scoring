"""
Pipeline complète, en une commande.

    ingest   : JSON SEC -> tables longues (bilan + flux)            src/ingest.py       (~1 h)
    wide     : bilan -> une ligne par clôture annuelle               src/build_wide.py
    flows    : flux -> une ligne par exercice                        src/build_flows.py
    features : 30 features (t) + 6 cibles (t -> t+1)                 src/features.py
    train    : modèles d'évaluation (<= 2021) + test 2022-2024
               + modèles de production (<= 2024)                     src/train.py
    score    : score de risque 0-100 de chaque bilan                 src/score.py

Exemples :
    python run_pipeline.py --src "E:/DSI/projet analytics edge"          # tout
    python run_pipeline.py --from wide                                   # sans relancer l'ingestion
    python run_pipeline.py --only train score                            # seulement ces étapes
"""
from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

from src import config

ETAPES = ["ingest", "wide", "flows", "features", "train", "score"]


def etape_ingest(src: Path, max_files: int) -> None:
    from src.ingest import ingest
    ingest(src, config.FACTS_LONG, ["10-K", "10-K/A"], max_files, concept_set="bilan")
    ingest(src, config.FLOWS_LONG, ["10-K", "10-K/A"], max_files, concept_set="flux")


def main() -> None:
    p = argparse.ArgumentParser(description="Pipeline balance-sheet-scoring")
    p.add_argument("--src", default=str(config.SRC_JSON), help="dossier des JSON SEC (ou companyfacts.zip)")
    p.add_argument("--from", dest="depuis", choices=ETAPES, default="ingest", help="reprendre à cette étape")
    p.add_argument("--only", nargs="+", choices=ETAPES, help="lancer seulement ces étapes")
    p.add_argument("--max-files", type=int, default=0, help="test rapide sur N entreprises (ingestion)")
    a = p.parse_args()

    etapes = a.only or ETAPES[ETAPES.index(a.depuis):]
    src = Path(a.src)
    for d in [config.INTERIM, config.PROCESSED, config.MODELS, config.REPORTS]:
        d.mkdir(parents=True, exist_ok=True)

    modules = {"wide": "src.build_wide", "flows": "src.build_flows",
               "features": "src.features", "train": "src.train"}
    debut = time.time()
    for e in etapes:
        t0 = time.time()
        print(f"\n=== {e} ===")
        if e == "ingest":
            etape_ingest(src, a.max_files)
        elif e == "score":
            _score(src)
        else:
            importlib.import_module(modules[e]).main()
        print(f"  ({time.time() - t0:.0f} s)")
    print(f"\n✅ Pipeline terminée en {(time.time() - debut) / 60:.1f} min : {', '.join(etapes)}")


def _score(src: Path) -> None:
    import joblib
    import pandas as pd
    from src.score import run
    load = lambda kind: {n: joblib.load(config.MODELS / f"{kind}_{n}.joblib") for n in config.POIDS}
    scores, verif = run(pd.read_parquet(config.MODEL_TABLE), load("prod"), load("eval"), src)
    scores.to_parquet(config.SCORES, index=False)
    verif.to_csv(config.REPORTS / "score_par_niveau_test.csv")
    print(f"  -> {config.SCORES}")


if __name__ == "__main__":
    main()
