"""
Configuration centrale du projet : chemins, seuils, réglages.
Tous les choix ci-dessous viennent des notebooks (01 à 08) — le commentaire dit d'où.
Changer un paramètre ici = le changer partout, sans toucher au code.
"""
from pathlib import Path

# ----------------------------------------------------------------------------- chemins
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

SRC_JSON = Path("E:/DSI/projet analytics edge")       # dossier des JSON SEC (modifiable : --src)

FACTS_LONG = INTERIM / "facts_long"                    # bilan, format long (ingest --set bilan)
FLOWS_LONG = INTERIM / "flows_long"                    # flux, format long (ingest --set flux)
BALANCE_WIDE = INTERIM / "balance_sheet_wide.parquet"
FLOWS_WIDE = INTERIM / "flows_wide.parquet"
MODEL_TABLE = PROCESSED / "model_table.parquet"
SCORES = PROCESSED / "scores.parquet"

# ----------------------------------------------------------------------------- nettoyage
MAX_GAP_DAYS = 550            # 01 : écart filed - end, pics 1-2 de l'histogramme (bilan + comparatif)
ANNUAL_GAP = (330, 400)       # 02 : écart entre deux clôtures annuelles (exercices de 52/53 semaines)
FLOW_DURATION = (350, 380)    # 06 : durée d'un flux annuel (exclut trimestres et 9 mois)
MIN_ASSETS = 1e6              # 03 : exclut les sociétés coquilles et erreurs d'échelle
INVALID_CIK = "0000000000"    # 04 : CIK vide dans certains JSON

# ----------------------------------------------------------------------------- découpage temporel
TRAIN_END, VAL_END, TEST_END = 2019, 2021, 2024   # année du bilan t

# ----------------------------------------------------------------------------- modèles
BASE_PARAMS = dict(max_iter=300, l2_regularization=1.0, random_state=0)
PARAMS = {                    # 07 : meilleurs réglages trouvés sur la validation
    "perte_CP":     dict(learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=100),
    "treso_moitie": dict(learning_rate=0.10, max_leaf_nodes=31, min_samples_leaf=50),
    "actifs_chute": dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100),
    "CP_negatifs":  dict(learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=300),
}
REGLE_VAL = {                 # 07 : meilleure règle à une variable, choisie sur la validation
    "perte_CP": ("RE_A", -1), "treso_moitie": ("d_C", 1),
    "actifs_chute": ("RE_A", -1), "CP_negatifs": ("E_A", -1),
}

# ----------------------------------------------------------------------------- score
POIDS = {"CP_negatifs": 0.35, "perte_CP": 0.25, "actifs_chute": 0.25, "treso_moitie": 0.15}
NIVEAUX = ([0, 50, 75, 90, 100.01], ["Faible", "Modéré", "Élevé", "Très élevé"])
RATIOS_AFFICHES = ["taille", "L_A", "E_A", "C_A", "RE_A", "AC_LC", "ROA", "marge_op",
                   "CFO_A", "couv_interets", "d_CA"]
