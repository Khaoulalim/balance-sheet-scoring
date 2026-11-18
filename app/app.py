"""
Dashboard — Alerte précoce sur la santé financière des entreprises cotées (SEC).

Lancer :  streamlit run app/app.py
Prérequis : python run_pipeline.py  (produit data/processed/scores.parquet et reports/*.csv)
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.logic import (EVENEMENTS, NIVEAUX, RATIOS, annee_par_defaut, derniers_bilans,  # noqa: E402
                       libelle_entreprise, points_forts, prepare, signaux)
from src import config  # noqa: E402

st.set_page_config(page_title="Alerte santé financière", page_icon="📉", layout="wide")
COULEUR = {"Faible": "🟢", "Modéré": "🟡", "Élevé": "🟠", "Très élevé": "🔴"}


@st.cache_data
def charger():
    if not config.SCORES.exists():
        return None, None, None
    df = prepare(pd.read_parquet(config.SCORES))
    lire = lambda nom: pd.read_csv(config.REPORTS / nom, index_col=0) if (config.REPORTS / nom).exists() else None
    return df, lire("metrics_test.csv"), lire("score_par_niveau_test.csv")


df, metrics, par_niveau = charger()
if df is None:
    st.error("Fichier de scores introuvable. Lance d'abord : `python run_pipeline.py --from wide`")
    st.stop()

noms = libelle_entreprise(df)

# ------------------------------------------------------------------------------------------- en-tête
st.title("📉 Alerte précoce sur la santé financière des entreprises")
st.caption("Entreprises cotées aux États-Unis · données SEC EDGAR (rapports annuels 10-K) · "
           "probabilité de forte dégradation du bilan l'année suivante")

onglet_vue, onglet_fiche, onglet_comp, onglet_perf = st.tabs(
    ["Vue d'ensemble", "Fiche entreprise", "Comparer deux entreprises", "Fiabilité du modèle"])

# ------------------------------------------------------------------------------------------- 1. vue d'ensemble
with onglet_vue:
    annees = sorted(df["annee"].unique(), reverse=True)
    annee = st.selectbox("Année du bilan", annees, index=annees.index(annee_par_defaut(df)))
    an = df[df["annee"] == annee]

    c = st.columns(5)
    c[0].metric("Entreprises analysées", f"{len(an):,}".replace(",", " "))
    for i, niv in enumerate(NIVEAUX, start=1):
        c[i].metric(f"{COULEUR[niv]} {niv}", f"{(an['niveau'] == niv).mean():.0%}")

    st.subheader(f"Les entreprises les plus à risque — bilans {annee}")
    niveaux_choisis = st.multiselect("Niveaux affichés", NIVEAUX, default=["Très élevé"])
    tableau = (an[an["niveau"].isin(niveaux_choisis)]
               .sort_values("score", ascending=False)
               .assign(niveau=lambda d: d["niveau"].astype(str).map(lambda n: f"{COULEUR[n]} {n}"),
                       cloture=lambda d: d["end"].dt.date,
                       **{f"p_{e}": (lambda d, e=e: d[f"p_{e}"] * 100) for e in EVENEMENTS}))
    st.dataframe(
        tableau[["nom", "cloture", "score", "niveau"] + [f"p_{e}" for e in EVENEMENTS]],
        hide_index=True, width="stretch",
        column_config={
            "nom": "Entreprise", "cloture": "Clôture",
            "score": st.column_config.ProgressColumn("Score de risque", min_value=0, max_value=100, format="%.0f"),
            "niveau": "Niveau",
            **{f"p_{e}": st.column_config.NumberColumn(lib, format="%.0f %%")
               for e, lib in EVENEMENTS.items()},
        })
    st.caption("Score = rang de l'entreprise parmi celles de la même année (95 = plus risquée que 95 % d'entre elles). "
               "Les colonnes suivantes sont les probabilités estimées de chaque dégradation l'année suivante.")

# ------------------------------------------------------------------------------------------- 2. fiche entreprise
def fiche(cik: str, cle: str) -> None:
    hist = df[df["cik"] == cik].sort_values("end")
    dernier = hist.iloc[-1]
    niv = str(dernier["niveau"])

    st.markdown(f"### {dernier['nom']}")
    a, b, c = st.columns(3)
    a.metric("Score de risque", f"{dernier['score']:.0f} / 100")
    b.metric("Niveau", f"{COULEUR[niv]} {niv}")
    c.metric("Dernier bilan", str(dernier["end"].date()))

    st.markdown("**Risque de chaque dégradation l'année suivante**")
    for e, lib in EVENEMENTS.items():
        rang = dernier[f"rang_{e}"]
        st.progress(min(max(rang / 100, 0.0), 1.0),
                    text=f"{lib} — probabilité {dernier[f'p_{e}']:.0%} · plus risquée que {rang:.0f} % des entreprises")

    g, d = st.columns(2)
    with g:
        st.markdown("**⚠️ Signaux de fragilité**")
        for s in signaux(dernier) or ["Aucun ratio parmi les 20 % les plus défavorables"]:
            st.markdown(f"- {s}")
    with d:
        st.markdown("**✅ Points forts**")
        for s in points_forts(dernier) or ["Aucun ratio parmi les 20 % les plus favorables"]:
            st.markdown(f"- {s}")

    if len(hist) > 1:
        st.markdown("**Évolution du score de risque**")
        st.line_chart(hist.assign(annee=hist["annee"].astype(str)).set_index("annee")["score"], height=220)

    with st.expander("Ratios détaillés"):
        st.dataframe(pd.DataFrame({
            "Ratio": [lib for lib, _ in RATIOS.values()],
            "Valeur": [dernier.get(r) for r in RATIOS],
            "Rang de risque dans l'année (100 = le plus risqué)": [dernier.get(f"risque_{r}") for r in RATIOS],
        }).round(3), hide_index=True, width="stretch", key=f"ratios_{cle}")


def choisir(label: str, cle: str, defaut: int = 0) -> str:
    options = sorted(noms, key=noms.get)
    return st.selectbox(label, options, index=defaut, format_func=noms.get, key=cle,
                        placeholder="Tape le nom d'une entreprise…")


with onglet_fiche:
    fiche(choisir("Rechercher une entreprise", "fiche"), "fiche")

# ------------------------------------------------------------------------------------------- 3. comparaison
with onglet_comp:
    st.caption("La question d'une banque : deux entreprises demandent un prêt, laquelle est la moins risquée ?")
    g, d = st.columns(2)
    with g:
        fiche(choisir("Entreprise A", "comp_a"), "a")
    with d:
        fiche(choisir("Entreprise B", "comp_b", defaut=1), "b")

# ------------------------------------------------------------------------------------------- 4. fiabilité
with onglet_perf:
    st.markdown("Évaluation honnête : modèles appris sur les bilans **≤ 2021**, testés sur **2022-2024**, "
                "années jamais vues pendant l'entraînement (configuration figée avant le test).")
    if metrics is not None:
        st.markdown("**Performance de chaque modèle (test 2022-2024)**")
        st.dataframe(metrics, width="stretch")
        st.caption("AUC : 0,5 = hasard, 1 = parfait. Lift top 10 % : parmi les 10 % d'entreprises signalées, "
                   "combien de fois plus de dégradations que la moyenne.")
    if par_niveau is not None:
        st.markdown("**Taux réel de dégradation selon le niveau attribué (test 2022-2024)**")
        st.dataframe(par_niveau.rename(columns=EVENEMENTS), width="stretch")
    st.info("Limites : prédictions moins fiables pour les petites entreprises (< 100 M$) ; "
            "le bilan seul ne dit pas tout (gouvernance, secteur, conjoncture) ; "
            "scores historiques recalculés a posteriori avec le modèle de production.")
