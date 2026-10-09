"""End-to-end orchestration: memory-bounded training, rule tuning, and streaming test inference.

v2 — Key changes:
  • default top_k raised to 40 (was 20), k_max to 50
  • per-country threshold tuning integrated
  • country-stratified validation split
  • support for larger training sizes with chunked processing
"""
import array
import gc
import json
import time
from dataclasses import asdict, dataclass
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .blocking import BlockingIndex, keys_for_table, record_keys
from .data import read_ground_truth, read_source
from .decision import (apply_country_rules, apply_rule, enforce_exclusive,
                       entity_scores, summarize, tune_rule)
from .features import BASE_FEATURES, IdfTable, add_context_features, pair_features
from .model import MatchModel
from .preprocess import _addr_parts, _name_parts, preprocess_records


@dataclass
class Config:
    train_dir: str
    test_dir: str
    output_dir: str = "output"
    artifacts_dir: str = "artifacts"
    n_train_s1: int = 300_000   # raised from 80k — use more training data
    k_max: int = 50             # raised from 25
    top_k: int = 40             # raised from 20 — wider candidate retrieval
    recall_keep: float = 0.98
    min_top_k: int = 10
    score_ratio_min: float = 0.03  # lowered from 0.05 — keep more marginal candidates
    alpha_max_recall_loss: float = 0.003
    exclusive_targets: str = "auto"
    val_frac: float = 0.2
    seed: int = 42
    s1_chunk: int = 20_000
    per_country_rules: bool = True  # NEW: per-country threshold tuning


def log(msg: str) -> None:
    print(msg, flush=True)


class CountryContext:
    """Memory-bounded streaming target pool using file byte offsets and array-based inverted index."""

    def __init__(self, country: str, data_dir: str, split: str = "train", max_records: Optional[int] = None) -> None:
        t0 = time.time()
        self.country = country
        self.data_dir = data_dir
        self.split = split
        self.index = BlockingIndex()
        self.offsets = array.array("Q")
        self.file_nums = array.array("B")
        self.file_handles = []
        self.col_maps = []

        seen = set()
        count = 0
        name_df: Dict[str, int] = {}
        addr_df: Dict[str, int] = {}

        for file_k, src_num in enumerate((2, 3)):
            path = Path(data_dir) / f"{split}_source{src_num}.tsv"
            if not path.exists():
                continue
            fh = open(path, "rb")
            self.file_handles.append(fh)
            header_line = fh.readline().decode("utf-8", "replace")
            header = header_line.strip().split("\t")
            col_map = {col.lower().strip(): i for i, col in enumerate(header)}
            self.col_maps.append(col_map)
            id_idx = col_map.get("entity_id", 0)
            name_idx = col_map.get("business_name", 1)
            addr_idx = col_map.get("business_address", 2)
            ctry_idx = col_map.get("country", 3)

            while True:
                offset = fh.tell()
                raw_line = fh.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", "replace")
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= max(id_idx, ctry_idx):
                    continue
                if parts[ctry_idx].strip() != country:
                    continue
                eid = parts[id_idx].strip()
                if not eid or eid in seen:
                    continue
                seen.add(eid)

                name = parts[name_idx].strip() if name_idx < len(parts) else ""
                addr = parts[addr_idx].strip() if addr_idx < len(parts) else ""

                n_clean, n_core, n_toks, n_key, legal, digits, phonetics, anchor, bigrams, n_comp, n_comps = _name_parts(name, country)
                a_clean, a_toks, postal, nums, missing, segments = _addr_parts(addr, country)
                keys = record_keys(n_toks, n_core, n_key, postal, nums, a_toks, phonetics, anchor, bigrams, n_comp, n_comps, segments)

                i = count
                self.offsets.append(offset)
                self.file_nums.append(file_k)

                if i % 3 == 0:
                    for tok in set(n_toks):
                        name_df[tok] = name_df.get(tok, 0) + 1
                    for tok in set(a_toks):
                        addr_df[tok] = addr_df.get(tok, 0) + 1

                for t, vals in keys:
                    self.index.add(t, vals, i)

                count += 1
                if count % 1_000_000 == 0:
                    log(f"[{country}] streaming index: {count:,} records processed...")
                if max_records and count >= max_records:
                    break
            if max_records and count >= max_records:
                break

        self.n_records = count
        self.index.build_weights(count)
        self.name_idf = IdfTable([], count)
        self.name_idf.df = name_df
        self.addr_idf = IdfTable([], count)
        self.addr_idf.df = addr_df
        del seen, name_df, addr_df
        gc.collect()

        log(f"[{country}] target pool ready: {count:,} records, "
            f"{sum(self.index.stats().values()):,} blocking keys ({time.time() - t0:.0f}s)")

    def retrieve(self, s1_tbl: Dict[str, list], k: int, log_every: int = 0, alpha: float = 0.0):
        si, ti, sc, rk, mx = [], [], [], [], []
        n = len(s1_tbl["entity_id"])
        t0 = time.time()
        for i in range(n):
            cands = self.index.query(keys_for_table(s1_tbl, i), k)
            if cands:
                top = cands[0][1]
                floor = alpha * top
                for r, (j, s) in enumerate(cands, 1):
                    if s < floor:
                        break
                    si.append(i)
                    ti.append(j)
                    sc.append(s)
                    rk.append(r)
                    mx.append(top)
            if log_every and (i + 1) % log_every == 0:
                log(f"[{self.country}] blocking {i + 1:,}/{n:,} ({(i + 1) / max(1e-6, time.time() - t0):.0f} entities/s)")
        return (np.asarray(si, np.int64), np.asarray(ti, np.int64), np.asarray(sc, np.float32),
                np.asarray(rk, np.float32), np.asarray(mx, np.float32))

    def features(self, s1_tbl, si, ti, sc, rk, mx) -> Tuple[np.ndarray, List[str]]:
        if len(si) == 0:
            return np.zeros((0, len(BASE_FEATURES)), dtype=np.float32), []
        unique_t, inv_t = np.unique(ti, return_inverse=True)

        t_sub_ids = []
        t_sub_names = []
        t_sub_addrs = []
        for j in unique_t:
            file_k = self.file_nums[j]
            offset = self.offsets[j]
            fh = self.file_handles[file_k]
            fh.seek(offset)
            line = fh.readline().decode("utf-8", "replace")
            parts = line.rstrip("\r\n").split("\t")
            cmap = self.col_maps[file_k]
            t_sub_ids.append(parts[cmap["entity_id"]].strip())
            name = parts[cmap["business_name"]].strip() if cmap["business_name"] < len(parts) else ""
            addr = parts[cmap["business_address"]].strip() if cmap["business_address"] < len(parts) else ""
            t_sub_names.append(name)
            t_sub_addrs.append(addr)

        t_tbl = {
            "entity_id": t_sub_ids,
            "name_clean": [], "name_core": [], "name_tokens": [], "name_key": [],
            "legal": [], "name_digits": [], "name_phonetics": [], "name_anchor": [], "name_bigrams": [],
            "name_compact": [], "name_compact_sorted": [],
            "addr_clean": [], "addr_tokens": [], "postal": [], "nums": [], "addr_missing": [], "addr_segments": []
        }
        for name, addr in zip(t_sub_names, t_sub_addrs):
            nc, ncor, ntok, nkey, leg, dig, ph, anc, big, ncomp, ncomps = _name_parts(name, self.country)
            ac, atok, post, num, miss, seg = _addr_parts(addr, self.country)
            t_tbl["name_clean"].append(nc)
            t_tbl["name_core"].append(ncor)
            t_tbl["name_tokens"].append(ntok)
            t_tbl["name_key"].append(nkey)
            t_tbl["legal"].append(leg)
            t_tbl["name_digits"].append(dig)
            t_tbl["name_phonetics"].append(ph)
            t_tbl["name_anchor"].append(anc)
            t_tbl["name_bigrams"].append(big)
            t_tbl["name_compact"].append(ncomp)
            t_tbl["name_compact_sorted"].append(ncomps)
            t_tbl["addr_clean"].append(ac)
            t_tbl["addr_tokens"].append(atok)
            t_tbl["postal"].append(post)
            t_tbl["nums"].append(num)
            t_tbl["addr_missing"].append(miss)
            t_tbl["addr_segments"].append(seg)

        feats = pair_features(s1_tbl, t_tbl, si, inv_t, sc, rk, mx, self.name_idf, self.addr_idf)
        t_ids_for_pairs = [t_sub_ids[inv] for inv in inv_t]
        return feats, t_ids_for_pairs

    def get_eids(self, unique_t: np.ndarray) -> List[str]:
        eids = []
        for j in unique_t:
            file_k = self.file_nums[j]
            offset = self.offsets[j]
            fh = self.file_handles[file_k]
            fh.seek(offset)
            line = fh.readline().decode("utf-8", "replace")
            parts = line.rstrip("\r\n").split("\t")
            cmap = self.col_maps[file_k]
            eids.append(parts[cmap["entity_id"]].strip())
        return eids

    def close(self):
        for fh in self.file_handles:
            try:
                fh.close()
            except Exception:
                pass


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------
def prepare_training(cfg: Config) -> dict:
    t_start = time.time()
    train_dir = Path(cfg.train_dir)
    gt = read_ground_truth(train_dir / "train_ground_truth.tsv")
    t_count = Counter(t for v in gt.values() for t in v)
    multi_rate = sum(1 for c in t_count.values() if c > 1) / max(1, len(t_count))
    log(f"Ground truth: {len(gt):,} Source-1 rows; {len(t_count):,} distinct matched targets, "
        f"{multi_rate:.4%} multi-matched")
    del t_count

    s1_all = read_source(train_dir / "train_source1.tsv")
    s1_all = s1_all[s1_all["entity_id"].isin(gt.keys())].drop_duplicates("entity_id")
    if cfg.n_train_s1 and len(s1_all) > cfg.n_train_s1:
        rng = np.random.default_rng(cfg.seed)
        sampled_ids = []
        c_groups = s1_all.groupby("country")
        n_per_c = cfg.n_train_s1 // len(c_groups)
        for _, grp in c_groups:
            c_ids = grp["entity_id"].to_numpy()
            n_take = min(len(c_ids), n_per_c)
            sampled_ids.extend(rng.choice(c_ids, n_take, replace=False))
        s1 = s1_all[s1_all["entity_id"].isin(set(sampled_ids))].reset_index(drop=True)
    else:
        s1 = s1_all.reset_index(drop=True)
    del s1_all
    n_true = np.array([len(gt[e]) for e in s1["entity_id"]], dtype=np.float64)
    log(f"Training entities: {len(s1):,} sampled ({int((n_true == 0).sum()):,} singletons, {int(n_true.sum()):,} true links)")

    ent_pos = {e: i for i, e in enumerate(s1["entity_id"])}
    parts = []

    for country, s1_c in s1.groupby("country", sort=True):
        ctx = CountryContext(country, cfg.train_dir, split="train")
        if ctx.n_records == 0:
            log(f"[{country}] no target records found")
            ctx.close()
            del ctx
            gc.collect()
            continue

        s1_tbl = preprocess_records(s1_c)
        t0 = time.time()
        si, ti, sc, rk, mx = ctx.retrieve(s1_tbl, cfg.k_max, log_every=15_000)
        log(f"[{country}] retrieved {len(si):,} pairs for {len(s1_c):,} entities ({time.time() - t0:.0f}s)")

        t0 = time.time()
        base, t_ids_for_pairs = ctx.features(s1_tbl, si, ti, sc, rk, mx)
        s1_ids = s1_tbl["entity_id"]
        ent = np.fromiter((ent_pos[s1_ids[i]] for i in si), dtype=np.int64, count=len(si))
        label = np.fromiter((t in gt[s1_ids[i]] for i, t in zip(si, t_ids_for_pairs)), dtype=bool, count=len(si))
        tid = np.asarray(t_ids_for_pairs, dtype=object)
        # Track country for each pair
        country_arr = np.full(len(si), country, dtype=object)
        log(f"[{country}] features computed ({time.time() - t0:.0f}s)")
        parts.append((ent, rk, label, base, tid, country_arr))

        ctx.close()
        del ctx, s1_tbl
        gc.collect()

    if parts:
        ent = np.concatenate([p[0] for p in parts])
        rank = np.concatenate([p[1] for p in parts])
        label = np.concatenate([p[2] for p in parts])
        base = np.vstack([p[3] for p in parts])
        tid = np.concatenate([p[4] for p in parts])
        pair_countries = np.concatenate([p[5] for p in parts])
    else:
        tid = np.zeros(0, dtype=object)
        ent, rank, label = np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, bool)
        base = np.zeros((0, len(BASE_FEATURES)), np.float32)
        pair_countries = np.zeros(0, dtype=object)

    log(f"Total training candidate pairs: {len(ent):,} ({int(label.sum()):,} positive) in {(time.time() - t_start) / 60:.1f} min")
    return {
        "ent": ent, "rank": rank, "label": label, "base": base, "n_true": n_true, "tid": tid,
        "gt_target_multi_rate": multi_rate,
        "ent_ids": s1["entity_id"].to_numpy(), "ent_country": s1["country"].to_numpy(),
        "pair_countries": pair_countries,
    }


def recall_curve(td: dict, k_max: int) -> pd.DataFrame:
    total = max(1.0, td["n_true"].sum())
    n_ent = len(td["n_true"])
    cand_counts = np.bincount(td["ent"], minlength=n_ent)
    rows = []
    for k in range(1, k_max + 1):
        kept = td["label"] & (td["rank"] <= k)
        rows.append({"k": k, "recall_ceiling": kept.sum() / total,
                     "avg_candidates": float(np.minimum(cand_counts, k).mean())})
    return pd.DataFrame(rows)


def choose_top_k(td: dict, cfg: Config):
    curve = recall_curve(td, cfg.k_max)
    if cfg.top_k:
        return int(cfg.top_k), curve
    target = cfg.recall_keep * curve["recall_ceiling"].iloc[-1]
    ok = curve[(curve["k"] >= cfg.min_top_k) & (curve["recall_ceiling"] >= target)]
    return int(ok["k"].iloc[0]) if len(ok) else cfg.k_max, curve


def candidate_mask(td: dict, top_k: int, alpha: float) -> np.ndarray:
    return (td["rank"] <= top_k) & (td["base"][:, BASE_FEATURES.index("bscore_ratio")] >= alpha - 1e-6)


def choose_alpha(td: dict, top_k: int, cfg: Config):
    total = max(1.0, td["n_true"].sum())
    n_ent = len(td["n_true"])
    rows = []
    for a in np.round(np.arange(0.0, 0.81, 0.05), 2):
        m = candidate_mask(td, top_k, a)
        rows.append({"alpha": float(a), "recall_ceiling": (td["label"] & m).sum() / total,
                     "avg_candidates": m.sum() / max(1, n_ent)})
    table = pd.DataFrame(rows)
    if cfg.score_ratio_min >= 0:
        return float(cfg.score_ratio_min), table
    floor = (1.0 - cfg.alpha_max_recall_loss) * table["recall_ceiling"].iloc[0]
    return float(table.loc[table["recall_ceiling"] >= floor, "alpha"].max()), table


def train_and_tune(td: dict, top_k: int, alpha: float, cfg: Config) -> dict:
    m = candidate_mask(td, top_k, alpha)
    ent, y, tid = td["ent"][m], td["label"][m], td["tid"][m]
    pair_countries = td["pair_countries"][m] if "pair_countries" in td else None
    X = add_context_features(td["base"][m], ent)
    n_ent = len(td["n_true"])
    
    # Country-stratified validation split
    rng = np.random.default_rng(cfg.seed + 1)
    val_mask = np.zeros(n_ent, dtype=bool)
    ent_countries = td["ent_country"]
    for c in np.unique(ent_countries):
        c_idx = np.flatnonzero(ent_countries == c)
        n_val = max(1, int(len(c_idx) * cfg.val_frac))
        chosen = rng.choice(c_idx, n_val, replace=False)
        val_mask[chosen] = True
    
    is_val = val_mask[ent]
    log(f"Train pairs {int((~is_val).sum()):,} | validation pairs {int(is_val.sum()):,} ({int(val_mask.sum()):,} entities)")

    t0 = time.time()
    log("Fitting baseline matcher...")
    model = MatchModel().fit(X[~is_val], y[~is_val], X[is_val], y[is_val])
    log(f"Baseline model trained in {time.time() - t0:.0f}s (best iteration: {model.best_iteration})")

    # Hard negative mining: identify false positive candidate pairs with high predicted probability
    p_train = model.predict(X[~is_val])
    y_train = y[~is_val]
    hard_neg_mask = (~y_train) & (p_train >= 0.35)
    n_hard = int(hard_neg_mask.sum())
    log(f"Hard negative mining: identified {n_hard:,} false positive pairs (p >= 0.35)")
    if n_hard > 50:
        sample_weights = np.ones(len(y_train), dtype=np.float32)
        sample_weights[hard_neg_mask] = 2.5
        log("Retraining matcher with hard negative re-weighting (2.5x)...")
        t0 = time.time()
        model = MatchModel().fit(X[~is_val], y[~is_val], X[is_val], y[is_val], sample_weight=sample_weights)
        log(f"Refined model trained in {time.time() - t0:.0f}s (best iteration: {model.best_iteration})")

    val_idx = np.flatnonzero(val_mask)
    remap = np.full(n_ent, -1, dtype=np.int64)
    remap[val_idx] = np.arange(len(val_idx))
    ent_v, y_v = remap[ent[is_val]], y[is_val]
    n_true_v = td["n_true"][val_idx]
    p_v = model.predict(X[is_val])
    
    # Get country labels for validation pairs
    val_ent_countries = ent_countries[val_idx]
    val_pair_countries = None
    if pair_countries is not None:
        val_pair_countries = pair_countries[is_val]

    # Tune global rule + per-country rules
    rule, curve, per_country_rules = tune_rule(ent_v, p_v, y_v, n_true_v, 
                                                country_labels=val_pair_countries)
    
    # Evaluate with global rule
    sel = apply_rule(ent_v, p_v, rule)
    metrics = summarize(ent_v, y_v, sel, n_true_v)
    best_single = curve.loc[curve["macro_f05"].idxmax()].to_dict()
    metrics["blocking_recall_ceiling"] = float((y_v).sum() / max(1.0, n_true_v.sum()))
    metrics["avg_candidates"] = float(np.bincount(ent_v, minlength=len(val_idx)).mean())
    exclusive = td["gt_target_multi_rate"] <= 0.005 if cfg.exclusive_targets == "auto" else (cfg.exclusive_targets == "on")
    sel_x = enforce_exclusive(tid[is_val], p_v, sel)
    metrics["macro_f05_with_exclusivity_on_val"] = summarize(ent_v, y_v, sel_x, n_true_v)["macro_f05"]
    metrics["gt_target_multi_rate"] = float(td["gt_target_multi_rate"])
    
    # Evaluate with per-country rules if available
    if per_country_rules and val_pair_countries is not None:
        sel_pc = apply_country_rules(ent_v, p_v, val_pair_countries, rule, per_country_rules)
        pc_metrics = summarize(ent_v, y_v, sel_pc, n_true_v)
        metrics["macro_f05_per_country_rules"] = pc_metrics["macro_f05"]
        log(f"Per-country rules F0.5: {pc_metrics['macro_f05']:.4f} vs global: {metrics['macro_f05']:.4f}")
        # Use per-country rules if they improve
        if pc_metrics["macro_f05"] > metrics["macro_f05"]:
            log("Using per-country rules (better than global)")
        else:
            log("Keeping global rule (per-country didn't improve)")
            per_country_rules = {}

    f, _, _ = entity_scores(ent_v, y_v.astype(bool), sel, n_true_v)
    per_country_df = (pd.DataFrame({
        "country": val_ent_countries, "f05": f,
        "singleton": n_true_v == 0,
        "predicted_empty": np.bincount(ent_v, weights=sel, minlength=len(val_idx)) == 0
    }).groupby("country").agg(
        entities=("f05", "size"),
        macro_f05=("f05", "mean"),
        singleton_rate=("singleton", "mean"),
        predicted_empty_rate=("predicted_empty", "mean")
    ))
    
    log(f"\nPer-country validation scores:")
    for c, row in per_country_df.iterrows():
        cr = per_country_rules.get(str(c), rule)
        log(f"  [{c}] F0.5={row['macro_f05']:.4f}, n={int(row['entities'])}, "
            f"rule=gate={cr['gate']}, pair={cr['pair']}, rel={cr['rel']}")

    return {
        "model": model, "rule": rule, "per_country_rules": per_country_rules,
        "metrics": metrics, "single_threshold_curve": curve,
        "best_single_threshold": best_single, "per_country": per_country_df, "top_k": top_k, "alpha": alpha,
        "exclusive": bool(exclusive),
        "recall_curve": recall_curve(td, int(td["rank"].max()) if len(td["rank"]) else top_k),
        "alpha_table": choose_alpha(td, top_k, cfg)[1],
        "importance": model.feature_importance()
    }


_TABLES = ("single_threshold_curve", "per_country", "importance", "recall_curve", "alpha_table")


def save_artifacts(cfg: Config, result: dict) -> None:
    art = Path(cfg.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    result["model"].save(art / "match_model.pkl")
    meta = {
        "rule": result["rule"], "per_country_rules": result.get("per_country_rules", {}),
        "top_k": result["top_k"], "alpha": result["alpha"],
        "exclusive": result["exclusive"], "metrics": result["metrics"],
        "best_single_threshold": result["best_single_threshold"], "config": asdict(cfg),
        "feature_names": result["model"].feature_names
    }
    (art / "run_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    for name in _TABLES:
        if name in result:
            result[name].to_csv(art / f"{name}.csv", index=(name == "per_country"))


def load_artifacts(cfg: Config):
    art = Path(cfg.artifacts_dir)
    meta = json.loads((art / "run_meta.json").read_text())
    return MatchModel.load(art / "match_model.pkl"), meta


def load_result(cfg: Config) -> dict:
    model, meta = load_artifacts(cfg)
    art = Path(cfg.artifacts_dir)
    result = {
        "model": model, "rule": meta["rule"],
        "per_country_rules": meta.get("per_country_rules", {}),
        "top_k": meta["top_k"], "alpha": meta["alpha"],
        "exclusive": meta["exclusive"],
        "metrics": meta["metrics"], "best_single_threshold": meta["best_single_threshold"]
    }
    for name in _TABLES:
        p = art / f"{name}.csv"
        if p.exists():
            result[name] = pd.read_csv(p, index_col=0 if name == "per_country" else None)
    return result


# --------------------------------------------------------------------------------------------
# Test Inference (Streaming with byte-seek low memory guarantee)
# --------------------------------------------------------------------------------------------
def run_test_inference(cfg: Config, model: MatchModel, top_k: int, rule: Dict[str, float],
                       alpha: float = 0.03, exclusive: bool = True,
                       per_country_rules: Dict[str, Dict[str, float]] = None) -> pd.DataFrame:
    t_start = time.time()
    test_dir, out = Path(cfg.test_dir), Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    s1 = read_source(test_dir / "test_source1.tsv")
    log(f"Test Set: {len(s1):,} Source-1 entities {s1['country'].value_counts().to_dict()}")
    log(f"Inference settings: top_k={top_k}, alpha={alpha}, rule={rule}, "
        f"per_country_rules={per_country_rules}, target_exclusivity={exclusive}")

    written = set()
    stats = []

    with open(out / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as fc, \
            open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")

        for country, s1_c in s1.groupby("country", sort=True):
            s1_c = s1_c[~s1_c["entity_id"].isin(written)].drop_duplicates("entity_id")
            ctx = CountryContext(country, cfg.test_dir, split="test")

            if ctx.n_records == 0:
                log(f"[{country}] no target records -> {len(s1_c):,} entities written as singletons")
                for e in s1_c["entity_id"]:
                    written.add(e)
                    fc.write(f"{e}\t\n")
                    fm.write(f"{e}\t\n")
                stats.append({"country": country, "s1_entities": len(s1_c), "avg_candidates": 0.0,
                              "non_empty_match_rate": 0.0, "avg_matches_when_non_empty": 0.0,
                              "exclusivity_removed": 0})
                ctx.close()
                del ctx
                gc.collect()
                continue

            # Use per-country rule if available
            country_rule = per_country_rules.get(country, rule) if per_country_rules else rule

            ent_ids = list(s1_c["entity_id"])
            p_ent = []
            p_t = []
            p_prob = []
            p_sel = []
            t0 = time.time()
            total_cands = 0

            for start in range(0, len(s1_c), cfg.s1_chunk):
                chunk_df = s1_c.iloc[start: start + cfg.s1_chunk]
                s1_tbl = preprocess_records(chunk_df)
                si, ti, sc, rk, mx = ctx.retrieve(s1_tbl, top_k, alpha=alpha)
                base_row = start
                chunk_cands = [[] for _ in range(len(s1_tbl["entity_id"]))]

                if len(si):
                    feats, t_ids_for_pairs = ctx.features(s1_tbl, si, ti, sc, rk, mx)
                    X = add_context_features(feats, si)
                    prob = model.predict(X)
                    sel = apply_rule(si, prob, country_rule)
                    for i, t in zip(si.tolist(), t_ids_for_pairs):
                        chunk_cands[i].append(t)
                    p_ent.append(si + base_row)
                    p_t.append(ti)
                    p_prob.append(prob.astype(np.float32))
                    p_sel.append(sel)

                for e, cl in zip(s1_tbl["entity_id"], chunk_cands):
                    written.add(e)
                    fc.write(f"{e}\t{','.join(cl)}\n")
                    total_cands += len(cl)

                done = min(start + cfg.s1_chunk, len(s1_c))
                rate = done / max(1e-6, time.time() - t0)
                n_sel = int(sum(int(s.sum()) for s in p_sel))
                log(f"[{country}] {done:,}/{len(s1_c):,} entities | {rate:.0f}/s | "
                    f"ETA {(len(s1_c) - done) / max(rate, 1e-6) / 60:.1f} min | matches {n_sel:,}")

            match_dict = {e: [] for e in ent_ids}
            removed = 0
            if p_ent:
                ent = np.concatenate(p_ent)
                tix = np.concatenate(p_t)
                prob = np.concatenate(p_prob)
                sel = np.concatenate(p_sel)
                if exclusive:
                    kept = enforce_exclusive(tix, prob, sel)
                    removed = int(sel.sum() - kept.sum())
                    sel = kept

                winning_t = tix[sel]
                unique_win, inv_win = np.unique(winning_t, return_inverse=True)
                win_eids = ctx.get_eids(unique_win)
                matched_target_ids = [win_eids[inv] for inv in inv_win]

                for e_idx, tid in zip(ent[sel].tolist(), matched_target_ids):
                    match_dict[ent_ids[e_idx]].append(tid)

                del ent, tix, prob, sel, winning_t, unique_win, inv_win, win_eids, matched_target_ids

            n_match = 0
            n_nonempty = 0
            for e in ent_ids:
                ml = match_dict[e]
                fm.write(f"{e}\t{','.join(ml)}\n")
                n_match += len(ml)
                n_nonempty += bool(ml)

            n_c = len(ent_ids)
            log(f"[{country}] completed: non-empty rate {n_nonempty / max(1, n_c):.1%}, "
                f"exclusivity filtered {removed:,} conflicts ({(time.time() - t0) / 60:.1f} min)")
            stats.append({"country": country, "s1_entities": n_c, "avg_candidates": total_cands / max(1, n_c),
                          "non_empty_match_rate": n_nonempty / max(1, n_c),
                          "avg_matches_when_non_empty": n_match / max(1, n_nonempty),
                          "exclusivity_removed": removed})

            ctx.close()
            del ctx, ent_ids, match_dict, p_ent, p_t, p_prob, p_sel
            gc.collect()

        leftover = [e for e in s1["entity_id"].drop_duplicates() if e not in written]
        for e in leftover:
            written.add(e)
            fc.write(f"{e}\t\n")
            fm.write(f"{e}\t\n")
        if leftover:
            log(f"Wrote {len(leftover):,} leftover Source-1 entities as singletons")

    summary = pd.DataFrame(stats)
    log(f"Test inference finished in {(time.time() - t_start) / 60:.1f} min; {len(written):,} rows written")
    return summary
