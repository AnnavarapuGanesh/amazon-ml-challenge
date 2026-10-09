"""Pairwise + candidate-group context features (country-agnostic for zero-shot transfer)."""
import math
from collections import Counter
from itertools import chain
from typing import Dict

import numpy as np
import pandas as pd

from .preprocess import char_grams

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler

    try:
        from rapidfuzz.process import cpdist
    except ImportError:
        cpdist = None
    SCORERS = {
        "ratio": fuzz.ratio,
        "token_sort": fuzz.token_sort_ratio,
        "token_set": fuzz.token_set_ratio,
        "partial": fuzz.partial_ratio,
        "jw": JaroWinkler.normalized_similarity,
    }
    SCALE = {"ratio": 100.0, "token_sort": 100.0, "token_set": 100.0, "partial": 100.0, "jw": 1.0}
except ImportError:
    import difflib
    cpdist = None

    def _r(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _tsort(a, b):
        return _r(" ".join(sorted(a.split())), " ".join(sorted(b.split())))

    def _tset(a, b):
        sa, sb = set(a.split()), set(b.split())
        inter = " ".join(sorted(sa & sb))
        da = (inter + " " + " ".join(sorted(sa - sb))).strip()
        db = (inter + " " + " ".join(sorted(sb - sa))).strip()
        return max(_r(inter, da), _r(inter, db), _r(da, db))

    def _partial(a, b):
        short, long_ = (a, b) if len(a) <= len(b) else (b, a)
        m = difflib.SequenceMatcher(None, short, long_).find_longest_match(0, len(short), 0, len(long_))
        return m.size / max(1, len(short))

    SCORERS = {"ratio": _r, "token_sort": _tsort, "token_set": _tset, "partial": _partial, "jw": _r}
    SCALE = {k: 1.0 for k in SCORERS}


BASE_FEATURES = [
    "name_ratio", "core_ratio", "core_token_sort", "core_token_set", "core_partial", "core_jw",
    "gram_jaccard", "core_exact", "namekey_exact", "core_len_ratio",
    "compact_exact", "compact_sorted_exact", "name_contains",
    "tok_idf_jaccard", "tok_idf_cov_s1", "tok_idf_cov_t", "shared_tok_count", "max_shared_idf",
    "max_missing_idf", "first_tok_equal", "first_tok_ratio", "acronym_match", "phonetic_match",
    "legal_cmp", "name_digits_cmp",
    "addr_ratio", "addr_token_sort", "addr_token_set", "addr_idf_jaccard", "addr_exact", "num_overlap",
    "num_conflict", "postal_cmp", "num_loc_match", "s1_addr_missing", "t_addr_missing", "is_s3",
    "bscore", "brank", "bscore_ratio", "name_empty",
    "seg_overlap", "has_seg_overlap",
]

CONTEXT_FEATURES = [
    "group_size", "core_token_set_gap", "core_token_set_rank", "tok_idf_jaccard_gap",
    "tok_idf_jaccard_rank", "addr_token_set_gap", "n_strong_in_group",
]

FEATURE_NAMES = BASE_FEATURES + CONTEXT_FEATURES


class IdfTable(dict):
    """Token -> IDF normalized to [0, 1] on one country's target pool (memoized on first use)."""

    def __init__(self, token_lists, n_docs: int) -> None:
        super().__init__()
        self.df = Counter(chain.from_iterable(token_lists))
        self.n1 = n_docs + 1.0
        self.log_n1 = math.log(self.n1) if n_docs > 0 else 1.0

    def __missing__(self, tok: str) -> float:
        v = math.log(self.n1 / (self.df.get(tok, 0) + 1.0)) / self.log_n1
        v = 1.0 if v > 1.0 else (0.0 if v < 0.0 else v)
        self[tok] = v
        return v


def _bulk(name: str, a: list, b: list) -> np.ndarray:
    scorer = SCORERS[name]
    if cpdist is not None and len(a):
        out = cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    else:
        out = np.fromiter((scorer(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))
    return out / np.float32(SCALE[name])


def _acronym(toks) -> str:
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""


def pair_features(s1: Dict[str, list], tg: Dict[str, list], s1_idx, t_idx, bscore, brank, bmax,
                  name_idf: IdfTable, addr_idf: IdfTable) -> np.ndarray:
    si, ti = np.asarray(s1_idx).tolist(), np.asarray(t_idx).tolist()
    n = len(si)
    out = np.zeros((n, len(BASE_FEATURES)), dtype=np.float32)
    if n == 0:
        return out

    col_idx = {name: i for i, name in enumerate(BASE_FEATURES)}

    S, T = s1, tg
    a_clean_list = [S["name_clean"][i] for i in si]
    b_clean_list = [T["name_clean"][j] for j in ti]
    a_core = [S["name_core"][i] for i in si]
    b_core = [T["name_core"][j] for j in ti]
    a_addr = [S["addr_clean"][i] for i in si]
    b_addr = [T["addr_clean"][j] for j in ti]
    a_miss = np.fromiter((S["addr_missing"][i] for i in si), dtype=bool, count=n)
    b_miss = np.fromiter((T["addr_missing"][j] for j in ti), dtype=bool, count=n)
    has_name = np.fromiter((bool(x) and bool(y) for x, y in zip(a_core, b_core)), dtype=bool, count=n)
    has_addr = ~a_miss & ~b_miss

    # Bulk similarities
    out[:, col_idx["name_ratio"]] = _bulk("ratio", a_clean_list, b_clean_list)
    out[:, col_idx["core_ratio"]] = _bulk("ratio", a_core, b_core)
    out[:, col_idx["core_token_sort"]] = _bulk("token_sort", a_core, b_core)
    out[:, col_idx["core_token_set"]] = _bulk("token_set", a_core, b_core)
    out[:, col_idx["core_partial"]] = _bulk("partial", a_core, b_core)
    out[:, col_idx["core_jw"]] = _bulk("jw", a_core, b_core)

    out[:, col_idx["addr_ratio"]] = _bulk("ratio", a_addr, b_addr)
    out[:, col_idx["addr_token_sort"]] = _bulk("token_sort", a_addr, b_addr)
    out[:, col_idx["addr_token_set"]] = _bulk("token_set", a_addr, b_addr)

    # Zero out empty fields
    for c in [col_idx[k] for k in ["name_ratio", "core_ratio", "core_token_sort", "core_token_set", "core_partial", "core_jw"]]:
        out[~has_name, c] = 0.0
    for c in [col_idx[k] for k in ["addr_ratio", "addr_token_sort", "addr_token_set"]]:
        out[~has_addr, c] = 0.0

    S_toks, T_toks = S["name_tokens"], T["name_tokens"]
    S_key, T_key = S["name_key"], T["name_key"]
    S_legal, T_legal = S["legal"], T["legal"]
    S_dig, T_dig = S["name_digits"], T["name_digits"]
    S_ph, T_ph = S["name_phonetics"], T["name_phonetics"]
    S_atoks, T_atoks = S["addr_tokens"], T["addr_tokens"]
    S_post, T_post = S["postal"], T["postal"]
    S_nums, T_nums = S["nums"], T["nums"]
    S_segs = S.get("addr_segments", [()]*len(S["entity_id"]))
    T_segs = T.get("addr_segments", [()]*len(T["entity_id"]))
    T_id = T["entity_id"]
    nidf, aidf = name_idf.__getitem__, addr_idf.__getitem__

    grams_cache: Dict[str, frozenset] = {}

    def grams(s):
        g = grams_cache.get(s)
        if g is None:
            g = frozenset(char_grams(s))
            grams_cache[s] = g
        return g

    (c_gram, c_exact, c_key, c_len, c_comp_exact, c_comp_sorted_exact, c_contains,
     c_jac, c_cov_a, c_cov_b, c_nshared, c_maxsh, c_maxmiss,
     c_first, c_first_ratio, c_acr, c_ph,
     c_addr_jac, c_addr_exact, c_legal, c_digits, c_overlap, c_conflict, c_postal, c_num_loc, c_is_s3,
     c_seg_overlap, c_has_seg_overlap) = [[] for _ in range(28)]

    for r in range(n):
        i, j = si[r], ti[r]
        ac, bc = a_core[r], b_core[r]
        if ac and bc:
            ga, gb = grams(ac), grams(bc)
            inter = len(ga & gb)
            c_gram.append(inter / (len(ga) + len(gb) - inter) if ga and gb else 0.0)
            c_exact.append(1.0 if ac == bc else 0.0)
            c_key.append(1.0 if S_key[i] == T_key[j] else 0.0)
            la, lb = len(ac), len(bc)
            c_len.append(la / lb if la < lb else lb / la)

            # Compact spaceless equality
            ac_nospace = ac.replace(" ", "")
            bc_nospace = bc.replace(" ", "")
            c_comp_exact.append(1.0 if (ac_nospace and ac_nospace == bc_nospace) else 0.0)
            c_comp_sorted_exact.append(1.0 if ("".join(sorted(ac.split())) == "".join(sorted(bc.split()))) else 0.0)
            c_contains.append(1.0 if (min(la, lb) >= 4 and (ac in bc or bc in ac)) else 0.0)

            ta, tb = S_toks[i], T_toks[j]
            sa, sb = set(ta), set(tb)
            shared = sa & sb
            w_a = sum(map(nidf, sa))
            w_b = sum(map(nidf, sb))
            if shared:
                w_sh = sum(map(nidf, shared))
                c_maxsh.append(max(map(nidf, shared)))
            else:
                w_sh = 0.0
                c_maxsh.append(0.0)
            diff = sa ^ sb
            c_maxmiss.append(max(map(nidf, diff)) if diff else 0.0)
            w_un = w_a + w_b - w_sh
            c_jac.append(w_sh / w_un if w_un > 0 else 0.0)
            c_cov_a.append(w_sh / w_a if w_a > 0 else 0.0)
            c_cov_b.append(w_sh / w_b if w_b > 0 else 0.0)
            c_nshared.append(float(len(shared)))

            # First token match & ratio
            if ta and tb:
                c_first.append(1.0 if ta[0] == tb[0] else 0.0)
                la_tok, lb_tok = ta[0], tb[0]
                short_len = min(len(la_tok), len(lb_tok))
                c_first_ratio.append(1.0 if la_tok == lb_tok else (0.7 if la_tok[:short_len] == lb_tok[:short_len] and short_len >= 3 else 0.0))
            else:
                c_first.append(0.0)
                c_first_ratio.append(0.0)

            # Acronym match
            acr_a, acr_b = _acronym(ta), _acronym(tb)
            c_acr.append(1.0 if ((acr_a and (acr_a in sb or acr_a == bc_nospace))
                                 or (acr_b and (acr_b in sa or acr_b == ac_nospace))) else 0.0)

            # Phonetic match
            pha, phb = S_ph[i], T_ph[j]
            c_ph.append(1.0 if (pha and phb and (pha[0] == phb[0] or set(pha) & set(phb))) else 0.0)
        else:
            c_gram.append(0.0)
            c_exact.append(0.0)
            c_key.append(0.0)
            c_len.append(0.0)
            c_comp_exact.append(0.0)
            c_comp_sorted_exact.append(0.0)
            c_contains.append(0.0)
            c_jac.append(0.0)
            c_cov_a.append(0.0)
            c_cov_b.append(0.0)
            c_nshared.append(0.0)
            c_maxsh.append(0.0)
            c_maxmiss.append(0.0)
            c_first.append(0.0)
            c_first_ratio.append(0.0)
            c_acr.append(0.0)
            c_ph.append(0.0)

        # Address similarities
        aa, ba = a_addr[r], b_addr[r]
        c_addr_exact.append(1.0 if (aa and ba and aa == ba and len(aa) > 5) else 0.0)

        if has_addr[r]:
            xa, xb = set(S_atoks[i]), set(T_atoks[j])
            if xa and xb:
                sh = xa & xb
                w_sh = sum(map(aidf, sh)) if sh else 0.0
                w_un = sum(map(aidf, xa)) + sum(map(aidf, xb)) - w_sh
                c_addr_jac.append(w_sh / w_un if w_un > 0 else 0.0)
            else:
                c_addr_jac.append(0.0)
        else:
            c_addr_jac.append(0.0)

        la_, lb_ = S_legal[i], T_legal[j]
        c_legal.append(0.0 if not (la_ and lb_) else (1.0 if la_ == lb_ else -1.0))
        da, db = S_dig[i], T_dig[j]
        c_digits.append(0.0 if not (da and db) else (1.0 if set(da) & set(db) else -1.0))
        na, nb = S_nums[i], T_nums[j]
        if na and nb:
            ov = float(len(set(na) & set(nb)))
            conflict = 1.0 if ov == 0 else 0.0
        else:
            ov, conflict = 0.0, 0.0
        c_overlap.append(ov)
        c_conflict.append(conflict)

        pa, pb = S_post[i], T_post[j]
        postal = 0.0 if not (pa and pb) else (1.0 if pa == pb else (0.5 if pa[:3] == pb[:3] else -1.0))
        c_postal.append(postal)

        loc_a = S_atoks[i][-1] if S_atoks[i] else ""
        loc_b = T_atoks[j][-1] if T_atoks[j] else ""
        num_loc = 1.0 if (ov > 0 and (postal == 1.0 or (loc_a and loc_b and loc_a == loc_b))) else 0.0
        c_num_loc.append(num_loc)

        c_is_s3.append(1.0 if T_id[j].startswith("S3") else 0.0)

        sa_seg = S_segs[i]
        tb_seg = T_segs[j]
        if sa_seg and tb_seg:
            sh_seg = float(len(set(sa_seg) & set(tb_seg)))
            c_seg_overlap.append(sh_seg)
            c_has_seg_overlap.append(1.0 if sh_seg > 0 else 0.0)
        else:
            c_seg_overlap.append(0.0)
            c_has_seg_overlap.append(0.0)

    # Assign to output matrix via col_idx
    out[:, col_idx["gram_jaccard"]] = c_gram
    out[:, col_idx["core_exact"]] = c_exact
    out[:, col_idx["namekey_exact"]] = c_key
    out[:, col_idx["core_len_ratio"]] = c_len
    out[:, col_idx["compact_exact"]] = c_comp_exact
    out[:, col_idx["compact_sorted_exact"]] = c_comp_sorted_exact
    out[:, col_idx["name_contains"]] = c_contains
    out[:, col_idx["tok_idf_jaccard"]] = c_jac
    out[:, col_idx["tok_idf_cov_s1"]] = c_cov_a
    out[:, col_idx["tok_idf_cov_t"]] = c_cov_b
    out[:, col_idx["shared_tok_count"]] = c_nshared
    out[:, col_idx["max_shared_idf"]] = c_maxsh
    out[:, col_idx["max_missing_idf"]] = c_maxmiss
    out[:, col_idx["first_tok_equal"]] = c_first
    out[:, col_idx["first_tok_ratio"]] = c_first_ratio
    out[:, col_idx["acronym_match"]] = c_acr
    out[:, col_idx["phonetic_match"]] = c_ph
    out[:, col_idx["legal_cmp"]] = c_legal
    out[:, col_idx["name_digits_cmp"]] = c_digits
    out[:, col_idx["addr_idf_jaccard"]] = c_addr_jac
    out[:, col_idx["addr_exact"]] = c_addr_exact
    out[:, col_idx["num_overlap"]] = c_overlap
    out[:, col_idx["num_conflict"]] = c_conflict
    out[:, col_idx["postal_cmp"]] = c_postal
    out[:, col_idx["num_loc_match"]] = c_num_loc
    out[:, col_idx["s1_addr_missing"]] = a_miss
    out[:, col_idx["t_addr_missing"]] = b_miss
    out[:, col_idx["is_s3"]] = c_is_s3
    out[:, col_idx["bscore"]] = bscore
    out[:, col_idx["brank"]] = brank
    out[:, col_idx["bscore_ratio"]] = np.asarray(bscore, dtype=np.float32) / np.maximum(np.asarray(bmax, dtype=np.float32), 1e-6)
    out[:, col_idx["name_empty"]] = ~has_name
    out[:, col_idx["seg_overlap"]] = c_seg_overlap
    out[:, col_idx["has_seg_overlap"]] = c_has_seg_overlap
    return out


def add_context_features(base: np.ndarray, group: np.ndarray) -> pd.DataFrame:
    """Computes per-candidate context against other candidates in the same Source-1 group."""
    df = pd.DataFrame(base, columns=BASE_FEATURES)
    g = pd.Series(group)
    df["group_size"] = g.map(g.value_counts()).to_numpy(dtype=np.float32)
    for col, gap_name, rank_name in (
        ("core_token_set", "core_token_set_gap", "core_token_set_rank"),
        ("tok_idf_jaccard", "tok_idf_jaccard_gap", "tok_idf_jaccard_rank"),
        ("addr_token_set", "addr_token_set_gap", None),
    ):
        grp = df[col].groupby(group)
        df[gap_name] = (df[col] - grp.transform("max")).astype(np.float32)
        if rank_name:
            df[rank_name] = grp.rank(method="min", ascending=False).astype(np.float32)
    strong = (df["core_token_set"] >= 0.9).astype(np.float32)
    df["n_strong_in_group"] = strong.groupby(group).transform("sum").astype(np.float32)
    return df[FEATURE_NAMES]
