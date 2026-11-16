"""
Étape 1 — Ingestion : companyfacts.zip (SEC) -> table longue filtrée (Parquet).

Lit soit le dossier de JSON dézippés, soit le zip directement. Garde uniquement :
  - la taxonomie us-gaap
  - l'unité USD
  - les concepts de bilan listés dans src/concepts.py
  - les formulaires choisis (10-K par défaut)

Sortie : data/interim/facts_long.parquet
  colonnes : cik, concept, val, end, filed, form, fy, fp, accn

Usage :
  python -m src.ingest                                   # dossier data/raw/companyfacts
  python -m src.ingest --src "E:/DSI/projet analytics edge" --max-files 100
  python -m src.ingest --src data/raw/companyfacts.zip   # lit le zip sans dézipper
  python -m src.ingest --forms 10-K 10-K/A 10-Q 10-Q/A
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.concepts import BALANCE_SHEET_CONCEPTS

COLUMNS = ["cik", "concept", "val", "end", "filed", "form", "fy", "fp", "accn"]


def parse_company(raw: bytes, concepts: set[str], forms: set[str]) -> list[dict]:
    """Transforme le JSON d'une entreprise en lignes (format long)."""
    data = json.loads(raw)
    cik = str(data.get("cik", "")).zfill(10)
    usgaap = data.get("facts", {}).get("us-gaap", {})

    rows = []
    for concept, payload in usgaap.items():
        if concept not in concepts:
            continue
        for fact in payload.get("units", {}).get("USD", []):
            if fact.get("form") not in forms:
                continue
            rows.append({
                "cik": cik,
                "concept": concept,
                "val": fact.get("val"),
                "end": fact.get("end"),
                "filed": fact.get("filed"),
                "form": fact.get("form"),
                "fy": fact.get("fy"),
                "fp": fact.get("fp"),
                "accn": fact.get("accn"),
            })
    return rows


def _iter_sources(src: Path, max_files: int):
    """Renvoie des paires (nom, lecteur) que src soit un .zip ou un dossier de JSON."""
    if src.is_dir():
        files = sorted(src.glob("*.json"))
        if max_files:
            files = files[:max_files]
        for f in files:
            yield f.name, f.read_bytes
    else:
        zf = zipfile.ZipFile(src)
        members = sorted(m for m in zf.namelist() if m.endswith(".json"))
        if max_files:
            members = members[:max_files]
        for name in members:
            yield name, (lambda n=name: zf.read(n))


def ingest(src: Path, out_path: Path, forms: list[str], max_files: int = 0) -> pd.DataFrame:
    if not src.exists():
        raise FileNotFoundError(f"Introuvable : {src}")
    concepts = set(BALANCE_SHEET_CONCEPTS)
    forms_set = set(forms)
    all_rows: list[dict] = []

    sources = list(_iter_sources(src, max_files))
    for name, read in tqdm(sources, desc="Entreprises"):
        try:
            all_rows.extend(parse_company(read(), concepts, forms_set))
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
            print(f"[skip] {name}: {e}")

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"\n✅ {len(df):,} lignes | {df['cik'].nunique():,} entreprises | "
          f"{df['concept'].nunique()} concepts -> {out_path}")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/raw/companyfacts",
                   help="Dossier de JSON dézippés OU fichier companyfacts.zip")
    p.add_argument("--out", default="data/interim/facts_long.parquet")
    p.add_argument("--forms", nargs="+", default=["10-K", "10-K/A"])
    p.add_argument("--max-files", type=int, default=0)
    a = p.parse_args()
    ingest(Path(a.src), Path(a.out), a.forms, a.max_files)


if __name__ == "__main__":
    main()
