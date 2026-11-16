# Balance Sheet Scoring — prédire l'évolution du bilan des entreprises

> Refonte personnelle du projet Analytics Edge (EMINES, équipe de 6), avec corrections méthodologiques.

## Problématique
Une banque reçoit deux demandes de prêt : laquelle est la moins risquée ?
On prédit l'évolution du bilan à la période suivante (t → t+1), puis on en tire un score de solvabilité et de liquidité.

## Données
SEC EDGAR — Company Facts (XBRL, US-GAAP).
Télécharger `companyfacts.zip` depuis https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
et le dézipper dans `data/raw/companyfacts/` (ou passer `--src` vers le dossier). Les données ne sont pas versionnées.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline
| Étape | Commande | Sortie |
|---|---|---|
| 1. | 1. Ingestion | `python -m src.ingest --src <dossier_json>` | `data/interim/facts_long/` — 3,0 M lignes, 14 081 entreprises, 25 concepts (10-K, USD) — ~35 min |
| 2. Table wide (dédup) | à venir | `data/interim/balance_sheet_wide.parquet` |
| 3. Cibles t→t+1 + features | à venir | |
| 4. Split temporel | à venir | |
| 5. Baselines + HGB | à venir | |
| 6. Score + dashboard | à venir | |

## Ce que j'ai corrigé par rapport à la version d'équipe
_(à compléter au fur et à mesure)_

## Résultats
_(à compléter)_
