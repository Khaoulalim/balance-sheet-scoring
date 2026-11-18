"""
Étape 1 — Ingestion : companyfacts.zip (SEC) -> table longue filtrée (Parquet).

Lit soit le dossier de JSON dézippés, soit le zip directement. Garde uniquement :
  - la taxonomie us-gaap
  - l'unité USD
  - les concepts de bilan listés dans src/concepts.py
  - les formulaires choisis (10-K par défaut)

Sortie : data/interim/facts_long/part-*.parquet  (relire : pd.read_parquet("data/interim/facts_long"))
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
import re
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.concepts import BALANCE_SHEET_CONCEPTS

COLUMNS = ["cik", "concept", "val", "end", "filed", "form", "fy", "fp", "accn"]


def cik_from_name(name: str) -> str | None:
    """'CIK0000001750.json' -> '0000001750' (le nom du fichier est la source la plus fiable)."""
    m = re.search(r"CIK(\d{10})", name)
    return m.group(1) if m else None


def parse_company(raw: bytes, concepts: set[str], forms: set[str],
                  cik_hint: str | None = None) -> list[dict]:
    """Transforme le JSON d'une entreprise en lignes (format long).
    cik_hint : CIK lu dans le nom du fichier. Certains JSON ont un champ "cik" vide ou à 0 :
    sans ce garde-fou, des entreprises différentes se retrouvaient fusionnées sous '0000000000'."""
    data = json.loads(raw)
    cik = cik_hint or str(data.get("cik") or "").zfill(10)
    if not cik.strip("0"):          # CIK vide ou nul -> on ne peut pas identifier l'entreprise
        return []
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


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
    return df


def ingest(src: Path, out_dir: Path, forms: list[str], max_files: int = 0,
           batch_size: int = 1000) -> None:
    """Écrit le résultat par blocs (part-00000.parquet, ...) pour ne pas saturer la RAM.
    Relire ensuite avec : pd.read_parquet("data/interim/facts_long")"""
    if not src.exists():
        raise FileNotFoundError(f"Introuvable : {src}")
    concepts = set(BALANCE_SHEET_CONCEPTS)
    forms_set = set(forms)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):   # repartir de zéro à chaque exécution
        old.unlink()

    rows: list[dict] = []
    n_rows, n_parts, n_skipped = 0, 0, 0
    ciks: set[str] = set()
    concepts_seen: set[str] = set()

    def flush() -> None:
        nonlocal rows, n_rows, n_parts
        if not rows:
            return
        df = _to_frame(rows)
        df.to_parquet(out_dir / f"part-{n_parts:05d}.parquet", index=False)
        n_rows += len(df)
        n_parts += 1
        rows = []

    sources = list(_iter_sources(src, max_files))
    for i, (name, read) in enumerate(tqdm(sources, desc="Entreprises"), start=1):
        try:
            company_rows = parse_company(read(), concepts, forms_set, cik_from_name(name))
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
            n_skipped += 1
            print(f"[skip] {name}: {e}")
            continue
        if company_rows:
            ciks.add(company_rows[0]["cik"])
            concepts_seen.update(r["concept"] for r in company_rows)
            rows.extend(company_rows)
        if i % batch_size == 0:
            flush()
    flush()

    print(f"\n✅ {n_rows:,} lignes | {len(ciks):,} entreprises | "
          f"{len(concepts_seen)} concepts | {n_parts} fichiers -> {out_dir}"
          + (f" | {n_skipped} fichiers ignorés" if n_skipped else ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/raw/companyfacts",
                   help="Dossier de JSON dézippés OU fichier companyfacts.zip")
    p.add_argument("--out", default="data/interim/facts_long",
                   help="Dossier de sortie (fichiers part-*.parquet)")
    p.add_argument("--forms", nargs="+", default=["10-K", "10-K/A"])
    p.add_argument("--max-files", type=int, default=0)
    a = p.parse_args()
    ingest(Path(a.src), Path(a.out), a.forms, a.max_files)


if __name__ == "__main__":
    main()
