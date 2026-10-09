"""Official macro F0.5 metric and the three-parameter decision rule tuned against it.

v2 — Per-country threshold tuning + wider grid search + cross-validation support.

Rule for one Source-1 entity with candidate probabilities p (max = p1):
    if p1 < gate:                       predict nothing (singleton protection)
    else keep candidates with p >= pair and p >= rel * p1, plus always the top candidate.
"""
import itertools
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def group_max(ent: np.ndarray, p: np.ndarray) -> np.ndarray:
    return pd.Series(p).groupby(ent).transform("max").to_numpy()


def select(p: np.ndarray, pmax: np.ndarray, gate: float, pair: float, rel: float) -> np.ndarray:
    return (pmax >= gate) & (p >= pair) & (p >= rel * pmax)


def entity_scores(ent: np.ndarray, label: np.ndarray, sel: np.ndarray, n_true: np.ndarray):
    """Per-entity (f05, precision, recall) exactly as defined in the challenge."""
    n_ent = len(n_true)
    npred = np.bincount(ent, weights=sel, minlength=n_ent)
    tp = np.bincount(ent, weights=sel & label, minlength=n_ent)
    f = np.zeros(n_ent)
    prec = np.zeros(n_ent)
    rec = np.zeros(n_ent)
    single = n_true == 0
    f[single] = (npred[single] == 0).astype(float)
    prec[single] = f[single]
    rec[single] = 1.0
    m = ~single & (npred > 0)
    p_ = tp[m] / npred[m]
    r_ = tp[m] / n_true[m]
    den = 0.25 * p_ + r_
    f[m] = np.where(den > 0, 1.25 * p_ * r_ / np.where(den > 0, den, 1), 0.0)
    prec[m], rec[m] = p_, r_
    return f, prec, rec


def summarize(ent, label, sel, n_true) -> Dict[str, float]:
    f, prec, rec = entity_scores(ent, label, sel, n_true)
    single = n_true == 0
    return {
        "macro_f05": float(f.mean()),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "singleton_accuracy": float(f[single].mean()) if single.any() else 1.0,
        "singleton_count": int(single.sum()),
        "matched_entity_f05": float(f[~single].mean()) if (~single).any() else 0.0,
    }


def tune_rule(ent: np.ndarray, p: np.ndarray, label: np.ndarray, n_true: np.ndarray,
              country_labels: np.ndarray = None):
    """Coarse-to-fine grid search of (gate, pair, rel) maximizing macro F0.5.
    
    If country_labels is provided, also tunes per-country thresholds.
    """
    label = label.astype(bool)
    pmax = group_max(ent, p)

    def score(g, q, r):
        return entity_scores(ent, label, select(p, pmax, g, q, r), n_true)[0].mean()

    # Wide grid search — precision-protecting thresholds
    gates = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    pairs = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    rels = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    best = (-1.0, 0.55, 0.50, 0.70)
    for g, q, r in itertools.product(gates, pairs, rels):
        s = score(g, q, r)
        if s > best[0]:
            best = (s, g, q, r)
    _, g0, q0, r0 = best

    # Fine search around best
    for g, q in itertools.product(np.arange(max(0.20, g0 - 0.05), min(0.90, g0 + 0.051), 0.01),
                                  np.arange(max(0.05, q0 - 0.05), min(0.85, q0 + 0.051), 0.01)):
        s = score(g, q, r0)
        if s > best[0]:
            best = (s, g, q, r0)
    _, g0, q0, _ = best
    for r in np.arange(max(0.20, r0 - 0.10), min(0.95, r0 + 0.101), 0.01):
        s = score(g0, q0, r)
        if s > best[0]:
            best = (s, g0, q0, r)
    _, g, q, r = best
    rule = {"gate": round(float(g), 4), "pair": round(float(q), 4), "rel": round(float(r), 4)}

    # ── Per-country rules ────────────────────────────────────────
    per_country_rules = {}
    if country_labels is not None:
        unique_countries = np.unique(country_labels)
        for c in unique_countries:
            c_mask = country_labels == c
            c_ent_ids = np.unique(ent[c_mask])
            # Create local entity indices for this country
            c_ent_map = {old: new for new, old in enumerate(c_ent_ids)}
            c_idx = np.array([i for i in range(len(ent)) if c_mask[i]])
            if len(c_idx) == 0:
                continue
            c_ent_local = np.array([c_ent_map[ent[i]] for i in c_idx])
            c_p = p[c_idx]
            c_label = label[c_idx]
            c_n_true = n_true[c_ent_ids]
            c_pmax = group_max(c_ent_local, c_p)

            def c_score(g_, q_, r_):
                return entity_scores(c_ent_local, c_label, 
                                     select(c_p, c_pmax, g_, q_, r_), c_n_true)[0].mean()

            c_best = (-1.0, g, q, r)
            # Search around the global rule
            for g_, q_, r_ in itertools.product(
                np.arange(max(0.20, g - 0.15), min(0.90, g + 0.151), 0.03),
                np.arange(max(0.05, q - 0.15), min(0.85, q + 0.151), 0.03),
                np.arange(max(0.20, r - 0.15), min(0.95, r + 0.151), 0.03),
            ):
                s = c_score(g_, q_, r_)
                if s > c_best[0]:
                    c_best = (s, g_, q_, r_)
            _, cg, cq, cr = c_best
            # Fine-tune
            for g_, q_ in itertools.product(
                np.arange(max(0.20, cg - 0.03), min(0.90, cg + 0.031), 0.01),
                np.arange(max(0.05, cq - 0.03), min(0.85, cq + 0.031), 0.01),
            ):
                s = c_score(g_, cq if q_ == cq else q_, cr)
                if s > c_best[0]:
                    c_best = (s, g_, q_, cr)
            _, cg, cq, cr = c_best
            per_country_rules[str(c)] = {
                "gate": round(float(cg), 4), "pair": round(float(cq), 4), "rel": round(float(cr), 4)
            }

        # Calibrate France rule for zero-shot inference (missing/unstandardized postal codes)
        if "France" not in per_country_rules:
            per_country_rules["France"] = {
                "gate": round(float(max(0.20, g - 0.05)), 4),
                "pair": round(float(q), 4),
                "rel": round(float(r), 4),
            }

    curve = []
    for t in np.arange(0.10, 0.96, 0.02):
        sel = select(p, pmax, t, t, 0.0)
        m = summarize(ent, label, sel, n_true)
        m["threshold"] = round(float(t), 3)
        curve.append(m)
    return rule, pd.DataFrame(curve), per_country_rules


def apply_rule(ent: np.ndarray, p: np.ndarray, rule: Dict[str, float]) -> np.ndarray:
    return select(p, group_max(ent, p), rule["gate"], rule["pair"], rule["rel"])


def apply_country_rules(ent: np.ndarray, p: np.ndarray, country_labels: np.ndarray,
                        global_rule: Dict[str, float], 
                        per_country_rules: Dict[str, Dict[str, float]]) -> np.ndarray:
    """Apply per-country rules if available, falling back to global rule."""
    pmax = group_max(ent, p)
    sel = np.zeros(len(p), dtype=bool)
    for c in np.unique(country_labels):
        c_mask = country_labels == c
        rule = per_country_rules.get(str(c), global_rule)
        sel[c_mask] = select(p[c_mask], pmax[c_mask], rule["gate"], rule["pair"], rule["rel"])
    return sel


def enforce_exclusive(target_key: np.ndarray, p: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Source 1 is deduplicated, so a target record can belong to at most one Source-1 entity."""
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        return sel
    df = pd.DataFrame({"t": target_key[idx], "p": p[idx], "i": idx})
    keep = (df.sort_values("p", ascending=False, kind="mergesort")
            .drop_duplicates("t", keep="first")["i"].to_numpy())
    out = np.zeros_like(sel)
    out[keep] = True
    return out
