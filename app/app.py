"""
Dashboard — Alerte précoce sur la santé financière des entreprises cotées (SEC).

Lancer :  streamlit run app/app.py
Prérequis : python run_pipeline.py  (produit data/processed/scores.parquet, models/prod_*.joblib, reports/*.csv)
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.logic import (EVENEMENTS, NIVEAUX, RATIOS, annee_par_defaut, bilan_ancien, couleur, texte_proba,  # noqa: E402
                       libelle_entreprise, points_forts, prepare, rapport_html, resume_executif,
                       risque_ratios, signaux)
from src import config  # noqa: E402
from src.predict_one import FAITS, AnalyseImpossible, analyser  # noqa: E402

st.set_page_config(page_title="Alerte santé financière", page_icon="📉", layout="wide")
COULEUR_NIVEAU = {"Faible": "🟢", "Modéré": "🟡", "Élevé": "🟠", "Très élevé": "🔴"}


# ============================================================================== données
@st.cache_data
def charger():
    if not config.SCORES.exists():
        return None, None, None, None
    df = prepare(pd.read_parquet(config.SCORES))
    lire = lambda nom: pd.read_csv(config.REPORTS / nom, index_col=0) if (config.REPORTS / nom).exists() else None
    faits = None
    if config.BALANCE_WIDE.exists():
        faits = pd.read_parquet(config.BALANCE_WIDE, columns=["cik", "end"] + list(FAITS))
    return df, lire("metrics_test.csv"), lire("score_par_niveau_test.csv"), faits


@st.cache_resource
def modeles_production():
    return {n: joblib.load(config.MODELS / f"prod_{n}.joblib") for n in config.POIDS
            if (config.MODELS / f"prod_{n}.joblib").exists()}


df, metrics, par_niveau, faits_bilan = charger()
if df is None:
    st.error("Fichier de scores introuvable. Lance d'abord : `python run_pipeline.py --from wide`")
    st.stop()
noms = libelle_entreprise(df)


def analyse_depuis_base(cik: str) -> dict:
    """Le dernier bilan d'une entreprise de la base, au même format qu'une analyse de JSON importé."""
    hist = df[df["cik"] == cik].sort_values("end")
    res = hist.iloc[-1].to_dict()
    res["niveau"] = str(res["niveau"])
    res["annee_reference"] = int(res["annee"])
    res["n_reference"] = int((df["annee"] == res["annee"]).sum())
    res["historique_scores"] = hist.set_index(hist["annee"].astype(str))["score"]
    if faits_bilan is not None:
        f = faits_bilan[(faits_bilan["cik"] == cik) & (faits_bilan["end"] == res["end"])]
        res["faits"] = {lib: (f[col].iloc[0] if len(f) else float("nan")) for col, lib in FAITS.items()}
    return res


# ============================================================================== affichage d'une analyse
def cartes_faits(res: dict) -> None:
    faits = res.get("faits") or {}
    if not faits:
        return
    st.markdown("**Chiffres du dernier bilan**")
    cols = st.columns(3)
    for i, (lib, v) in enumerate(faits.items()):
        with cols[i % 3].container(border=True):
            st.caption(lib)
            st.markdown(f"**{v:,.0f} $**".replace(",", " ") if pd.notna(v) else "—")


def afficher(res: dict, cle: str, compact: bool = False, cartes: bool = True) -> None:
    niv = res["niveau"]
    st.markdown(f"### {res['nom']}")
    a, b, c = st.columns(3)
    a.metric("Score de risque", f"{res['score']:.0f} / 100")
    b.metric("Niveau", f"{COULEUR_NIVEAU[niv]} {niv}")
    c.metric("Dernier bilan", str(pd.Timestamp(res["end"]).date()))

    alerte = bilan_ancien(res, int(df["annee"].max()))
    if alerte:
        st.warning(f"⚠️ {alerte}")
    resume = resume_executif(res)
    {"Faible": st.success, "Modéré": st.info, "Élevé": st.warning, "Très élevé": st.error}[niv](resume)

    if cartes and not compact:
        cartes_faits(res)

    st.markdown("**Risque de dégradation l'année suivante**")
    cols = st.columns(2 if compact else 4)
    for i, (e, lib) in enumerate(EVENEMENTS.items()):
        rang = res[f"rang_{e}"]
        with cols[i % len(cols)].container(border=True):
            st.markdown(f"{couleur(rang)} **{lib}**")
            st.markdown(f"Probabilité **{texte_proba(res, e)}**" if texte_proba(res, e)[0].isdigit()
                        else f"**{texte_proba(res, e)}** (situation actuelle)")
            if not texte_proba(res, e)[0].isdigit():
                st.caption("Événement déjà survenu : rang de risque maximal")
            else:
                st.caption(f"Plus risquée que {rang:.0f} % des entreprises" if rang >= 50
                           else f"Moins risquée que {100 - rang:.0f} % des entreprises")

    g, d = st.columns(2)
    with g:
        st.markdown("**Signaux de fragilité**")
        for s in signaux(res) or []:
            st.warning(s)
        if not signaux(res):
            st.caption("Aucun ratio parmi les 20 % les plus défavorables.")
    with d:
        st.markdown("**Points forts**")
        for s in points_forts(res) or []:
            st.success(s)
        if not points_forts(res):
            st.caption("Aucun ratio parmi les 20 % les plus favorables.")

    hist = res.get("historique_scores")
    if hist is not None and len(hist) > 1 and not compact:
        st.markdown("**Évolution du score de risque**")
        st.line_chart(hist, height=200)

    with st.expander("Ratios détaillés"):
        st.dataframe(pd.DataFrame({
            "Ratio": [lib for lib, _ in RATIOS.values()],
            "Valeur": [res.get(r) for r in RATIOS],
            "Rang de risque dans l'année (100 = le plus risqué)": [res.get(f"risque_{r}") for r in RATIOS],
        }).round(3), hide_index=True, width="stretch", key=f"ratios_{cle}")

    st.download_button("📄 Télécharger le rapport", rapport_html(res),
                       file_name=f"analyse_risque_{res['nom'][:40].replace(' ', '_')}.html",
                       mime="text/html", key=f"rapport_{cle}",
                       help="Rapport HTML : s'ouvre dans le navigateur, imprimable en PDF (Ctrl + P).")


def choisir(label: str, cle: str, defaut: int = 0) -> str:
    options = sorted(noms, key=noms.get)
    return st.selectbox(label, options, index=defaut, format_func=noms.get, key=cle,
                        placeholder="Tape le nom d'une entreprise…")


# ============================================================================== page
st.title("📉 Alerte précoce sur la santé financière des entreprises")
st.caption("Entreprises cotées aux États-Unis · données SEC EDGAR (rapports annuels 10-K) · "
           "probabilité de forte dégradation du bilan l'année suivante")

t_import, t_vue, t_fiche, t_comp, t_perf = st.tabs(
    ["Analyser un dossier", "Vue d'ensemble", "Fiche entreprise", "Comparer deux entreprises", "Fiabilité du modèle"])

# ------------------------------------------------------------------------------ 1. analyser un dossier (JSON)
with t_import:
    st.markdown("Importe le fichier **JSON SEC** (companyfacts) d'une entreprise, même absente de la base, "
                "pour obtenir son analyse de risque. Fichiers disponibles sur "
                "`data.sec.gov/api/xbrl/companyfacts/CIK##########.json`.")
    fichier = st.file_uploader("Fichier companyfacts (.json)", type="json")
    modeles = modeles_production()
    if fichier is not None:
        if len(modeles) < len(config.POIDS):
            st.error("Modèles de production introuvables. Lance d'abord : `python run_pipeline.py --from train`")
        else:
            try:
                res = analyser(fichier.getvalue(), modeles, df)
                res.update(risque_ratios(res, df[df["annee"] == res["annee_reference"]]))
                st.success(f"Fichier lu : **{res['nom']}** · {res['historique']} bilan(s) annuel(s) exploitable(s) "
                           f"· comparaison avec {res['n_reference']:,} entreprises de {res['annee_reference']}"
                           .replace(",", " "))
                cartes_faits(res)
                if st.button("Lancer l'analyse", type="primary", key="lancer"):
                    st.session_state["analyse_ok"] = fichier.name
                if st.session_state.get("analyse_ok") == fichier.name:
                    st.divider()
                    afficher(res, "import", cartes=False)   # cartes déjà affichées au-dessus, mais gardées pour le rapport
            except AnalyseImpossible as err:
                st.error(f"Analyse impossible : {err}")
            except ValueError:
                st.error("Ce fichier n'est pas un JSON SEC companyfacts valide.")

# ------------------------------------------------------------------------------ 2. vue d'ensemble
with t_vue:
    annees = sorted(df["annee"].unique(), reverse=True)
    annee = st.selectbox("Année du bilan", annees, index=annees.index(annee_par_defaut(df)))
    an = df[df["annee"] == annee]

    c = st.columns(5)
    c[0].metric("Entreprises analysées", f"{len(an):,}".replace(",", " "))
    for i, niv in enumerate(NIVEAUX, start=1):
        c[i].metric(f"{COULEUR_NIVEAU[niv]} {niv}", f"{(an['niveau'] == niv).mean():.0%}")

    st.subheader(f"Les entreprises les plus à risque — bilans {annee}")
    niveaux_choisis = st.multiselect("Niveaux affichés", NIVEAUX, default=["Très élevé"])
    tableau = (an[an["niveau"].isin(niveaux_choisis)]
               .sort_values("score", ascending=False)
               .assign(niveau=lambda d: d["niveau"].astype(str).map(lambda n: f"{COULEUR_NIVEAU[n]} {n}"),
                       cloture=lambda d: d["end"].dt.date,
                       **{f"p_{e}": (lambda d, e=e: d[f"p_{e}"] * 100) for e in EVENEMENTS})
               .assign(p_CP_negatifs=lambda d: d["p_CP_negatifs"].where(~d["CP_deja_negatifs"])))
    st.dataframe(
        tableau[["nom", "cloture", "score", "niveau"] + [f"p_{e}" for e in EVENEMENTS]],
        hide_index=True, width="stretch",
        column_config={
            "nom": "Entreprise", "cloture": "Clôture",
            "score": st.column_config.ProgressColumn("Score de risque", min_value=0, max_value=100, format="%.0f"),
            "niveau": "Niveau",
            **{f"p_{e}": st.column_config.NumberColumn(court, help=EVENEMENTS[e], format="%.0f %%")
               for e, court in {"CP_negatifs": "P(CP < 0)", "perte_CP": "P(perte CP)",
                                "actifs_chute": "P(actifs −20 %)", "treso_moitie": "P(tréso ÷ 2)"}.items()},
        })
    st.caption("Score = rang de l'entreprise parmi celles de la même année (95 = plus risquée que 95 % d'entre elles). "
               "Les colonnes P(…) sont les probabilités estimées de chaque dégradation l'année suivante "
               "(P(CP < 0) vide : capitaux propres déjà négatifs).")

# ------------------------------------------------------------------------------ 3. fiche entreprise
with t_fiche:
    afficher(analyse_depuis_base(choisir("Rechercher une entreprise", "fiche")), "fiche")

# ------------------------------------------------------------------------------ 4. comparaison
with t_comp:
    st.caption("La question d'une banque : deux entreprises demandent un prêt, laquelle est la moins risquée ?")
    g, d = st.columns(2)
    with g:
        afficher(analyse_depuis_base(choisir("Entreprise A", "comp_a")), "a", compact=True)
    with d:
        afficher(analyse_depuis_base(choisir("Entreprise B", "comp_b", defaut=1)), "b", compact=True)

# ------------------------------------------------------------------------------ 5. fiabilité
with t_perf:
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
