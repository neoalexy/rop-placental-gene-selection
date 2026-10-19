import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# genes reported by Huang et al. (2026) on the same cohort
KNOWN = {
    "MMP25": "ENSG00000008516",
    "CAPG": "ENSG00000042493",
    "MCAM": "ENSG00000076706",
    "B2M": "ENSG00000166710",
}


def ttest(expr, y):
    """Student t-test per gene, ROP vs control, on log-CPM values."""
    rop, ctrl = expr[y == 1], expr[y == 0]
    _, p = stats.ttest_ind(rop, ctrl)
    res = pd.DataFrame({"log2fc": rop.mean() - ctrl.mean(), "p": p}, index=expr.columns).dropna()
    res["padj"] = multipletests(res["p"], method="fdr_bh")[1]
    return res.sort_values("p")


def pool(expr, y, p=0.05, lfc=1.0, known=True):
    """Candidate genes: p < 0.05 and |log2FC| > 1, plus the known genes if requested.

    Genes are returned in the column order of `expr`, not by p-value.
    """
    deg = ttest(expr, y)
    keep = set(deg.index[(deg["p"] < p) & (deg["log2fc"].abs() > lfc)])
    if known:
        keep |= set(KNOWN.values()) & set(deg.index)
    return [g for g in expr.columns if g in keep]
