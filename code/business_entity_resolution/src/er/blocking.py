"""Scalable, high-recall candidate generation: IDF-weighted multi-key inverted index.

v2 — six new key types targeting the 15% of true matches missed by v1:
  TP  : 4-char token prefix       (morphological variants: 'services'/'service')
  TS  : stemmed tokens             (singular/plural: 'traders'/'trading'→'trad')
  G4  : character 4-grams          (more discriminative overlap than 3-grams)
  PHP : phonetic pair compound     (transliteration: 'lakshmi traders'→L250_T636)
  FL  : first+last token compound  (reordered names share first/last tokens)
  TPL : token prefix + locality    (fuzzy name + location compound)

Uses array.array contiguous 32-bit integer postings and cap pruning.
"""
import array
import heapq
import math
import re
from operator import itemgetter
from typing import Dict, List, Tuple

from .preprocess import char_grams, soundex

# ──────────────────────────────────────────────────────────────────
# Stemming: ultra-simple suffix removal for blocking keys only
# ──────────────────────────────────────────────────────────────────
_STEM_RX = re.compile(
    r"(?:ations?|ations|ments?|nesses|ness|ings?|tion|sion|ence|ance|ment"
    r"|ous|ive|ful|less|able|ible|ies|ers?|ors?|ists?|ates?"
    r"|ling|ling|ally|edly|ical|ular|eous"
    r"|es|ed|ly|al|en|s)$"
)


def _stem(tok: str) -> str:
    """Aggressive stem for blocking only — maps morphological variants to same key."""
    if len(tok) <= 4:
        return tok
    s = _STEM_RX.sub("", tok)
    return s if len(s) >= 3 else tok


# ──────────────────────────────────────────────────────────────────
# Key caps and IDF weights
# ──────────────────────────────────────────────────────────────────
KEY_CAPS = {
    # Original keys (some caps raised)
    "T": 6000, "G": 5000, "N": 3000, "NC": 2500, "NCS": 2500,
    "P": 5000, "S": 1500, "A": 3000, "PT": 1500, "PS": 1200, "NP": 800,
    "NCP": 1000, "SNL": 1500, "SNL2": 1200, "SNA": 1200,
    "L": 4000, "NL": 800, "NCL": 800,
    "PH": 5000,   # raised from 3500 — common phonetic codes were being dropped
    "PHT": 1500, "ANC": 3500, "ANCP": 1000, "TB": 2000,
    # New key types
    "TP": 5000,   # token 4-char prefix
    "TS": 5000,   # stemmed token
    "G4": 4000,   # character 4-grams
    "G5": 3000,   # character 5-grams
    "STP": 3000,  # sorted token pairs
    "PHP": 2500,  # phonetic pair compound
    "FL": 2500,   # first + last token
    "TPL": 1200,  # token prefix + locality compound
    "TPP": 1000,  # token prefix + postal compound
}

KEY_WEIGHTS = {
    # Original keys
    "T": 3.0, "G": 0.8, "N": 4.0, "NC": 4.5, "NCS": 4.0,
    "P": 1.5, "S": 1.0, "A": 1.0, "PT": 3.2, "PS": 3.0, "NP": 5.0,
    "NCP": 5.0, "SNL": 4.5, "SNL2": 4.0, "SNA": 3.5,
    "L": 0.6, "NL": 4.0, "NCL": 4.0, "PH": 2.5, "PHT": 4.0, "ANC": 3.5, "ANCP": 5.0, "TB": 3.5,
    # New keys
    "TP": 2.0,    # token prefix — lower weight, broad coverage
    "TS": 2.5,    # stemmed token
    "G4": 0.5,    # 4-grams — low weight, high recall
    "G5": 0.7,    # 5-grams — discriminative n-gram
    "STP": 4.0,   # sorted token pairs — high precision compound
    "PHP": 3.5,   # phonetic pair — high precision
    "FL": 3.0,    # first+last token
    "TPL": 3.5,   # token prefix + locality
    "TPP": 4.0,   # token prefix + postal
}

# Protected compound keys never dropped even if frequent
PROTECTED_KEYS = {
    "PT", "PS", "NP", "NL", "PHT", "ANCP", "NCP", "NCL", "SNL", "SNL2", "SNA",
    "PHP", "FL", "TPL", "TPP", "STP",
}


def _char_4grams(s: str) -> set:
    """Character 4-grams from a string."""
    s = s.replace(" ", "")
    if len(s) < 4:
        return {s} if s else set()
    return {s[i:i + 4] for i in range(len(s) - 3)}


def _char_5grams(s: str) -> set:
    """Character 5-grams from a string."""
    s = s.replace(" ", "")
    if len(s) < 5:
        return {s} if s else set()
    return {s[i:i + 5] for i in range(len(s) - 4)}


def record_keys(name_tokens, name_core, name_key, postal, nums, addr_tokens,
                phonetics, anchor, bigrams, name_compact: str = "", name_compact_sorted: str = "",
                segments: tuple = ()) -> List[Tuple[str, tuple]]:
    if not name_compact and name_core:
        name_compact = name_core.replace(" ", "")
    if not name_compact_sorted and name_tokens:
        name_compact_sorted = "".join(sorted(name_tokens))

    # ── Original keys ───────────────────────────────────────────
    keys = [
        ("T", name_tokens[:5]),
        ("G", tuple(char_grams(name_core[:25]))),
        ("N", (name_key,) if name_key else ()),
        ("NC", (name_compact,) if len(name_compact) >= 3 else ()),
        ("NCS", (name_compact_sorted,) if len(name_compact_sorted) >= 3 else ()),
        ("A", addr_tokens[:4]),
        ("S", nums[:4]),
        ("PH", phonetics[:3]),
        ("ANC", (anchor,) if anchor else ()),
        ("TB", bigrams[:3]),
    ]

    # ── NEW: Token prefix keys (TP) ─────────────────────────────
    # 4-char prefix of tokens catches morphological variants:
    #   "services"/"service" → "serv", "enterprises"/"enterprise" → "ente"
    prefixes = tuple({t[:4] for t in name_tokens[:5] if len(t) >= 5})
    if prefixes:
        keys.append(("TP", prefixes))

    # ── NEW: Stemmed token keys (TS) ─────────────────────────────
    # Ultra-simple stemming: catches singular/plural/tense variants
    stems = tuple({_stem(t) for t in name_tokens[:5] if len(t) >= 4} - set(name_tokens[:5]))
    if stems:
        keys.append(("TS", stems))

    # ── NEW: Character 4-gram and 5-gram keys (G4, G5) ─────────
    g4 = tuple(_char_4grams(name_core[:25]))
    if g4:
        keys.append(("G4", g4))
    g5 = tuple(_char_5grams(name_core[:30]))
    if g5:
        keys.append(("G5", g5))

    # ── NEW: Sorted token pairs (STP) ──────────────────────────
    # For entities with 2-4 tokens, every pair of distinct tokens sorted alphabetically
    # "Raj Enterprises" -> "enterprises_raj"
    # "Enterprises Raj" -> "enterprises_raj"
    # Invariant to word permutations
    if len(name_tokens) >= 2:
        top_toks = name_tokens[:4]
        stp = tuple(
            f"{top_toks[a]}_{top_toks[b]}" if top_toks[a] < top_toks[b] else f"{top_toks[b]}_{top_toks[a]}"
            for a in range(len(top_toks))
            for b in range(a + 1, len(top_toks))
            if top_toks[a] != top_toks[b]
        )
        if stp:
            keys.append(("STP", stp))

    # ── NEW: Phonetic pair compound (PHP) ────────────────────────
    # Concatenate soundex of first two tokens → very precise phonetic fingerprint
    if len(phonetics) >= 2:
        keys.append(("PHP", (f"{phonetics[0]}_{phonetics[1]}",)))

    # ── NEW: First + Last token (FL) ─────────────────────────────
    # For 3+ token names, the first and last distinctive words often survive reordering
    if len(name_tokens) >= 3:
        first = name_tokens[0]
        last = name_tokens[-1]
        if first != last:
            keys.append(("FL", (f"{first}|{last}",)))

    loc = addr_tokens[-1] if addr_tokens else ""
    loc2 = addr_tokens[-2] if len(addr_tokens) >= 2 else ""

    # Street number + locality / address tokens compounds
    for n in nums[:2]:
        if loc:
            keys.append(("SNL", (f"{n}|{loc}",)))
        if loc2:
            keys.append(("SNL2", (f"{n}|{loc2}",)))
        for at in addr_tokens[:4]:
            if len(at) >= 4 and not at.isdigit():
                keys.append(("SNA", (f"{n}|{at}",)))

    if postal:
        keys.append(("P", (postal,)))
        keys.append(("PT", tuple(f"{postal}|{t}" for t in name_tokens[:4])))
        keys.append(("PS", tuple(f"{postal}|{n}" for n in nums[:3])))
        if name_key:
            keys.append(("NP", (f"{name_key}|{postal}",)))
        if name_compact:
            keys.append(("NCP", (f"{name_compact}|{postal}",)))
        if anchor:
            keys.append(("ANCP", (f"{anchor}|{postal}",)))
        if phonetics:
            keys.append(("PHT", tuple(f"{ph}|{postal[:3]}" for ph in phonetics[:2])))
        # NEW: token prefix + postal compound (TPP)
        if prefixes:
            keys.append(("TPP", tuple(f"{p}|{postal}" for p in prefixes[:3])))
    else:
        # Locality fallback when postal is absent (critical for France)
        loc_val = segments[-1] if segments else (addr_tokens[-1] if addr_tokens else "")
        if loc_val:
            keys.append(("L", (loc_val,)))
            if name_key:
                keys.append(("NL", (f"{name_key}|{loc_val}",)))
            if name_compact:
                keys.append(("NCL", (f"{name_compact}|{loc_val}",)))
            if phonetics:
                keys.append(("PHT", tuple(f"{ph}|{loc_val}" for ph in phonetics[:2])))
            if anchor:
                keys.append(("ANCP", (f"{anchor}|{loc_val}",)))
            # NEW: token prefix + locality compound (TPL)
            if prefixes:
                keys.append(("TPL", tuple(f"{p}|{loc_val}" for p in prefixes[:3])))

    return keys


def keys_for_table(tbl: Dict[str, list], i: int):
    comp = tbl["name_compact"][i] if "name_compact" in tbl else ""
    comps = tbl["name_compact_sorted"][i] if "name_compact_sorted" in tbl else ""
    seg = tbl["addr_segments"][i] if "addr_segments" in tbl else ()
    return record_keys(
        tbl["name_tokens"][i], tbl["name_core"][i], tbl["name_key"][i],
        tbl["postal"][i], tbl["nums"][i], tbl["addr_tokens"][i],
        tbl["name_phonetics"][i], tbl["name_anchor"][i], tbl["name_bigrams"][i],
        comp, comps, seg
    )


class BlockingIndex:
    def __init__(self, caps: Dict[str, int] = None, weights: Dict[str, float] = None) -> None:
        self.caps = dict(caps or KEY_CAPS)
        self.weights = dict(weights or KEY_WEIGHTS)
        self.tables: Dict[str, Dict[str, array.array]] = {t: {} for t in self.caps}
        self.over: Dict[str, set] = {t: set() for t in self.caps}
        self.wtabs: Dict[str, List[float]] = {}
        self.n_records = 0

    def add(self, t: str, vals: tuple, i: int):
        table = self.tables.get(t)
        if table is None:
            return
        cap = self.caps[t]
        over = self.over[t]
        for v in vals:
            if v in over:
                continue
            lst = table.get(v)
            if lst is None:
                table[v] = array.array("I", [i])
            elif len(lst) < cap:
                lst.append(i)
            else:
                if t not in PROTECTED_KEYS:
                    over.add(v)
                    del table[v]

    def build_weights(self, n: int):
        self.n_records = n
        if hasattr(self, "over"):
            self.over.clear()
            del self.over
        log_n = math.log(n + 1.0)
        for t, cap in self.caps.items():
            w = self.weights.get(t, 1.0)
            self.wtabs[t] = [0.0] + [w * math.log((n + 1.0) / df) / log_n for df in range(1, cap + 1)]

    def query(self, keys, k: int) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = {}
        get = scores.get
        tables, wtabs = self.tables, self.wtabs
        for t, vals in keys:
            table = tables.get(t)
            wtab = wtabs.get(t)
            if table is None or wtab is None:
                continue
            for v in vals:
                lst = table.get(v)
                if lst is None:
                    continue
                w = wtab[min(len(lst), len(wtab) - 1)]
                if w < 0.10:
                    continue
                for c in lst:
                    scores[c] = get(c, 0.0) + w
        if not scores:
            return []
        return heapq.nlargest(k, scores.items(), key=itemgetter(1))

    def stats(self) -> Dict[str, int]:
        return {t: len(tbl) for t, tbl in self.tables.items()}
