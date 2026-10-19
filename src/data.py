import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load_counts():
    """Raw read counts, genes x samples."""
    return pd.read_csv(DATA / "GSE218039_18942R_featureCounts.txt.gz", sep="\t", index_col="geneid")


def load_meta():
    text = gzip.open(DATA / "GSE218039_family.soft.gz", "rt").read()
    rows = []
    for block in text.split("^SAMPLE = ")[1:]:
        get = lambda key: re.search(rf"{key}: (\w)", block).group(1)
        rows.append({
            "sample": re.search(r"!Sample_title = (\S+)", block).group(1),
            "sex": get("Sex"),
            "preeclampsia": get("preeclampsia") == "Y",
            "rop": get("rop") == "Y",
        })
    return pd.DataFrame(rows).set_index("sample")


def log_cpm(counts):
    return np.log2(counts / counts.sum() * 1e6 + 1)


def load():
    """Returns log2(CPM+1) expression (samples x genes), ROP labels and sample metadata."""
    counts = load_counts()
    meta = load_meta().loc[counts.columns]
    expr = log_cpm(counts).T
    return expr, meta["rop"].astype(int).to_numpy(), meta


def symbols():
    """Ensembl ID -> gene symbol for the candidate genes (from mygene.info)."""
    df = pd.read_csv(DATA / "gene_symbols.csv", index_col="geneid")
    return df["symbol"].dropna().to_dict()


def name(gene, table=None):
    table = table if table is not None else symbols()
    return table.get(gene, gene)
