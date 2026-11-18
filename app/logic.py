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
    "taille":        ("Petite taille (log10 des actifs)", -1),
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


# ------------------------------------------------------------------ cartes, résumé, rapport
def couleur(rang: float) -> str:
    """Code couleur d'un rang de risque (0-100), aligné sur les niveaux du score."""
    if pd.isna(rang):
        return "⚪"
    return "🟢" if rang < 50 else "🟡" if rang < 75 else "🟠" if rang < 90 else "🔴"


def risque_ratios(valeurs: dict, reference: pd.DataFrame) -> dict:
    """Rang de risque (0-100) de chaque ratio d'une entreprise par rapport aux entreprises de référence."""
    out = {}
    for r, (_, sens) in RATIOS.items():
        v, ref = valeurs.get(r), reference[r].dropna() if r in reference else pd.Series(dtype=float)
        out[f"risque_{r}"] = float(((sens * ref) < sens * v).mean() * 100) if pd.notna(v) and len(ref) else float("nan")
    return out


def pct(p: float) -> str:
    """0,07 -> « 7 % » (écriture française)."""
    return f"{p * 100:.0f} %"


def position(score: float) -> str:
    """Formule lisible du rang : « plus risquée que 92 % » ou « moins risquée que 89 % »."""
    if score >= 50:
        return f"plus risquée que {score:.0f} % des entreprises de la même année"
    return f"moins risquée que {100 - score:.0f} % des entreprises de la même année"


def resume_executif(res: dict) -> str:
    """Une phrase de synthèse pour le banquier."""
    pire = max(EVENEMENTS, key=lambda e: res.get(f"rang_{e}", 0) or 0)
    phrase = f"Niveau de risque {res['niveau'].lower()} : score de {res['score']:.0f}/100, {position(res['score'])}. "
    if res.get("CP_deja_negatifs"):
        phrase += "Les capitaux propres sont déjà négatifs : les dettes dépassent les actifs. "
        pire = max((e for e in EVENEMENTS if e != "CP_negatifs"), key=lambda e: res.get(f"rang_{e}", 0) or 0)
    if res["niveau"] in ("Élevé", "Très élevé"):
        phrase += (f"Principal point d'attention : {EVENEMENTS[pire].lower()} "
                   f"(probabilité {pct(res[f'p_{pire}'])}). Analyse approfondie recommandée.")
    elif res["niveau"] == "Modéré":
        phrase += f"À surveiller en priorité : {EVENEMENTS[pire].lower()}."
    else:
        phrase += "Pas de signal de dégradation marqué pour l'année suivante."
    return phrase


def texte_proba(res: dict, e: str) -> str:
    """Probabilité affichée, ou « Déjà négatifs » si l'événement a déjà eu lieu."""
    if e == "CP_negatifs" and res.get("CP_deja_negatifs"):
        return "Déjà négatifs"
    return pct(res[f"p_{e}"])


def bilan_ancien(res: dict, derniere_annee: int) -> str | None:
    """Avertissement si le dernier bilan date de plus d'un an avant la dernière année disponible."""
    annee = pd.Timestamp(res["end"]).year
    if annee < derniere_annee - 1:
        return (f"Dernier bilan publié : {annee}. L'entreprise ne dépose plus de rapport annuel depuis "
                f"(rachat, sortie de cote ou faillite) : l'analyse décrit sa situation en {annee}.")
    return None


def rapport_html(res: dict) -> str:
    """Rapport autonome à télécharger (s'ouvre dans un navigateur, imprimable en PDF)."""
    lignes_risques = "".join(
        f"<tr><td>{couleur(res[f'rang_{e}'])} {lib}</td><td>{texte_proba(res, e)}</td>"
        f"<td>{res[f'rang_{e}']:.0f} %</td></tr>" for e, lib in EVENEMENTS.items())
    faits = res.get("faits") or {}
    lignes_faits = "".join(f"<tr><td>{k}</td><td>{v:,.0f} $</td></tr>".replace(",", " ")
                           for k, v in faits.items() if pd.notna(v))
    liste = lambda items: "".join(f"<li>{s}</li>" for s in items) or "<li>Aucun</li>"
    return f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>Analyse de risque — {res['nom']}</title>
<style>body{{font-family:Arial,sans-serif;max-width:820px;margin:32px auto;color:#222;line-height:1.45}}
h1{{font-size:22px}}h2{{font-size:16px;margin-top:24px}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}}.score{{font-size:28px;font-weight:bold}}
.note{{color:#666;font-size:12px;margin-top:28px}}</style></head><body>
<h1>Analyse de risque financier — {res['nom']}</h1>
<p>Dernier bilan : {pd.Timestamp(res['end']).date()} · comparé à {res.get('n_reference', '')} entreprises
de l'année {res.get('annee_reference', pd.Timestamp(res['end']).year)}</p>
<p class="score">Score {res['score']:.0f}/100 — {res['niveau']}</p>
<p><b>Résumé :</b> {resume_executif(res)}</p>
<h2>Risque de dégradation l'année suivante</h2>
<table><tr><th>Événement</th><th>Probabilité</th><th>Plus risquée que</th></tr>{lignes_risques}</table>
<h2>Chiffres du dernier bilan</h2><table>{lignes_faits}</table>
<h2>Signaux de fragilité</h2><ul>{liste(signaux(res))}</ul>
<h2>Points forts</h2><ul>{liste(points_forts(res))}</ul>
<p class="note">Modèles HistGradientBoosting appris sur les bilans SEC (10-K). Évaluation sur 2022-2024, années
non vues : AUC de 0,71 à 0,93 selon l'événement. Outil d'aide à la décision : ne remplace pas l'analyse
d'un crédit (gouvernance, secteur, garanties, conjoncture).</p></body></html>"""
