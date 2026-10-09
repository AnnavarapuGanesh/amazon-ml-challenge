"""TSV loading helpers with low memory footprint."""
from pathlib import Path
from typing import Dict, Set

import pandas as pd

SRC_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in SRC_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[SRC_COLS].copy()
    df["entity_id"] = df["entity_id"].str.strip()
    df["country"] = df["country"].str.strip()
    return df[df["entity_id"] != ""]


def read_targets(data_dir, split: str) -> pd.DataFrame:
    parts = [read_source(Path(data_dir) / f"{split}_source{k}.tsv") for k in (2, 3)]
    return pd.concat(parts, ignore_index=True).drop_duplicates("entity_id", keep="first")


def partition_by_country(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {c: g.reset_index(drop=True) for c, g in df.groupby("country", sort=True)}


def read_ground_truth(path) -> Dict[str, Set[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    gt: Dict[str, Set[str]] = {}
    for s1, raw in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        s1 = s1.strip()
        raw = raw.strip()
        gt[s1] = {x.strip() for x in raw.split(",") if x.strip()} if raw and raw.lower() != "nan" else set()
    return gt
