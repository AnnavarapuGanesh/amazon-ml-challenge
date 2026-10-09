"""Country-aware (open-set) normalization, phonetic encoding, and tokenization."""
import re
import sys
import unicodedata
from typing import Dict, List, Tuple

import pandas as pd

LEGAL_FORMS: Dict[str, List[str]] = {
    "US": [
        "limited liability company", "limited liability partnership", "limited partnership",
        "professional corporation", "incorporated", "incorporation", "corporation", "company",
        "limited", "corp", "inc", "llc", "llp", "lp", "ltd", "co", "pllc", "pc", "plc",
    ],
    "India": [
        "private limited company", "private limited", "pvt ltd", "pvt limited", "pvt", "private",
        "limited liability partnership", "public limited company", "public limited", "limited",
        "ltd", "llp", "opc",
    ],
    "France": [
        "societe par actions simplifiee unipersonnelle", "societe par actions simplifiee",
        "societe a responsabilite limitee", "entreprise unipersonnelle a responsabilite limitee",
        "societe anonyme", "societe civile immobiliere", "societe en nom collectif",
        "societe civile", "sasu", "sarl", "sas", "sa", "eurl", "sci", "snc", "gie", "selarl",
        "sca", "scop", "scp",
    ],
}

OTHER_LEGAL_FORMS = [
    "gmbh", "ag", "kg", "ohg", "ug", "bv", "nv", "vof", "srl", "spa", "sas", "snc", "sl", "slu", "sau",
    "oy", "oyj", "ab", "as", "asa", "aps", "pty ltd", "pty", "pte ltd", "pte", "sdn bhd", "bhd", "kk",
    "sa de cv", "ltda", "eirl", "sp z o o", "sro", "doo", "ad", "plc", "gie",
]

_ALL_FORMS = sorted({f for forms in LEGAL_FORMS.values() for f in forms} | set(OTHER_LEGAL_FORMS),
                    key=len, reverse=True)

LEGAL_CANON = {
    "ltd": "ltd", "limited": "ltd",
    "pvt ltd": "pvt", "pvt limited": "pvt", "private limited": "pvt", "private limited company": "pvt",
    "pvt": "pvt", "private": "pvt",
    "inc": "inc", "incorporated": "inc", "incorporation": "inc",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "llc": "llc", "limited liability company": "llc",
    "llp": "llp", "limited liability partnership": "llp",
    "sarl": "sarl", "societe a responsabilite limitee": "sarl",
    "sas": "sas", "societe par actions simplifiee": "sas",
    "sasu": "sasu", "societe par actions simplifiee unipersonnelle": "sasu",
    "sa": "sa", "societe anonyme": "sa",
    "eurl": "eurl", "entreprise unipersonnelle a responsabilite limitee": "eurl",
    "sci": "sci", "societe civile immobiliere": "sci",
    "snc": "snc", "societe en nom collectif": "snc",
}

NAME_STOPWORDS = {
    "the", "and", "of", "a", "an", "for", "de", "la", "le", "les", "des", "du", "d", "l", "et", "y",
}

PREFIX_TITLES = {
    "sri", "shri", "shree", "smt", "dr", "mr", "mrs", "ms", "prof", "m/s"
}

NAME_ABBREV = {
    "intl": "international", "mfg": "manufacturing", "mfr": "manufacturers", "mfrs": "manufacturers",
    "svc": "services", "svcs": "services", "srvcs": "services", "service": "services",
    "assoc": "associates", "assocs": "associates", "assn": "association", "bros": "brothers",
    "natl": "national", "univ": "university", "hosp": "hospital", "ctr": "center", "centre": "center",
    "mgmt": "management", "mgt": "management", "sys": "systems", "tech": "technologies",
    "technology": "technologies", "techs": "technologies", "engg": "engineering",
    "pharma": "pharmaceuticals", "pharmaceutical": "pharmaceuticals", "devt": "development",
    "grp": "group", "hldgs": "holdings", "hldg": "holding", "inds": "industries",
    "industry": "industries", "enterprise": "enterprises", "entps": "enterprises",
    "solution": "solutions", "trader": "traders", "mt": "mount", "ft": "fort",
    "sri": "shri", "shree": "shri", "sree": "shri", "cie": "compagnie", "ste": "societe",
    "corp": "corporation", "co": "company", "dept": "department", "govt": "government",
}

ADDR_ABBREV = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "pl": "place", "sq": "square", "hwy": "highway", "pkwy": "parkway",
    "ste": "suite", "fl": "floor", "flr": "floor", "apt": "apartment", "bldg": "building",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "nr": "near", "opp": "opposite", "sect": "sector",
    "sec": "sector", "ph": "phase", "mkt": "market", "cir": "circle", "rte": "route",
    "chem": "chemin", "imp": "impasse", "fbg": "faubourg", "stn": "station", "dist": "district",
    "distt": "district", "tq": "taluk", "tal": "taluk", "vill": "village", "vil": "village",
    "blk": "block", "rly": "railway", "ext": "extension", "extn": "extension",
    "hno": "houseno", "h_no": "houseno", "no": "number",
    "pin": "", "zip": "", "cedex": "",
}

POSTAL_PATTERNS = {
    "US": re.compile(r"\b(\d{5})(?:[-\s]\d{4})?\b"),
    "India": re.compile(r"\b(\d{3})\s?(\d{3})\b"),
    "France": re.compile(r"\b(\d{2})\s?(\d{3})\b"),
}
DEFAULT_POSTAL = re.compile(r"\b(\d{4,6})\b")
NULL_VALUES = {"", "none", "null", "nan", "unknown", "n a", "na", "address", "not available", "0", "undefined"}

_URL = re.compile(r"^\s*(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-.]*?)\.(?:com|net|org|in|co|fr|io|biz|info|us|uk)"
                  r"(?:\.[a-z]{2})?/?\s*$", re.IGNORECASE)
_APOS = re.compile(r"['’`´]")
_SINGLE_LETTERS = re.compile(r"\b(?:[a-z0-9] )+[a-z0-9]\b")
_TRANSLIT = str.maketrans({
    "ß": "ss", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ø": "o", "Ø": "o",
    "đ": "d", "Đ": "d", "ł": "l", "Ł": "l", "ı": "i"
})
_NONWORD = re.compile(r"[\W_]+")
_DIGITS = re.compile(r"\d+")

_intern = sys.intern

_SOUNDEX_MAP = {
    'B': '1', 'F': '1', 'P': '1', 'V': '1',
    'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
    'D': '3', 'T': '3',
    'L': '4',
    'M': '5', 'N': '5',
    'R': '6'
}


def soundex(name: str) -> str:
    """Standard Soundex encoding mapping phonetically similar transliterations."""
    if not name:
        return ""
    clean = "".join(ch for ch in name.upper() if ch.isalpha())
    if not clean:
        return ""
    first = clean[0]
    encoded = [first]
    prev = _SOUNDEX_MAP.get(first, '0')
    for char in clean[1:]:
        code = _SOUNDEX_MAP.get(char, '0')
        if code != '0':
            if code != prev:
                encoded.append(code)
            prev = code
        else:
            prev = '0'
    res = "".join(encoded)[:4]
    return res.ljust(4, '0')


def _alt(forms: List[str]) -> str:
    return "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))


_PREFIX_FORMS_BY_COUNTRY = {
    "France": [f for f in LEGAL_FORMS["France"] if len(f) >= 3],
    "India": ["private limited company", "private limited", "pvt ltd", "pvt limited", "pvt", "private", "public limited", "limited", "ltd", "llp", "opc"],
    "US": ["limited liability company", "corporation", "incorporated", "limited", "company", "corp", "inc", "llc", "llp", "ltd", "co"],
}
_PREFIX_RX_BY_COUNTRY = {
    c: re.compile(rf"^(?:(?:{_alt(f)})\s)+")
    for c, f in _PREFIX_FORMS_BY_COUNTRY.items()
}
_HONORIFIC_RX = re.compile(r"^(?:(?:m\s*s|ms|m/s|shri|shree|sri|sree|smt|dr|prof|om|jai|mr|mrs)\s+)+")
_DOMAIN_TAIL_RX = re.compile(r"\s+(?:com|net|org|in|co\s+in|fr|io|biz|info|us|uk)$")

_NO_PREFIX = re.compile(r"(?!x)x")
_LEGAL_RX = {
    c: (re.compile(rf"(?:\s(?:{_alt(f)}))+$"), _PREFIX_RX_BY_COUNTRY.get(c, _NO_PREFIX))
    for c, f in LEGAL_FORMS.items()
}
_LEGAL_RX_DEFAULT = (re.compile(rf"(?:\s(?:{_alt(_ALL_FORMS)}))+$"), _PREFIX_RX_BY_COUNTRY.get("France", _NO_PREFIX))


def normalize_text(text) -> str:
    """Lower-case, strip accents and punctuation; keeps non-latin letters."""
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    if not s.isascii():
        s = s.translate(_TRANSLIT)
        s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = s.lower().replace("&", " and ").replace("@", " at ")
    s = _APOS.sub("", s)
    s = _NONWORD.sub(" ", s)
    return s.strip()


def _strip_legal(name: str, country: str) -> Tuple[str, str]:
    suffix_rx, _ = _LEGAL_RX.get(country, _LEGAL_RX_DEFAULT)
    prefix_rx = _PREFIX_RX_BY_COUNTRY.get(country, _PREFIX_RX_BY_COUNTRY["France"])
    form = ""
    m = suffix_rx.search(" " + name)
    if m and m.start() > 0:
        form = m.group(0).strip()
        name = (" " + name)[: m.start()].strip()
    m = prefix_rx.match(name + " ")
    if m and m.end() < len(name) + 1:
        if not form:
            form = m.group(0).strip()
        name = (name + " ")[m.end():].strip()
    if form:
        last = form.split(" ")
        canon = LEGAL_CANON.get(form) or LEGAL_CANON.get(last[-1], form)
        form = canon
    return name, form


def _name_parts(raw_name, country: str):
    raw = "" if raw_name is None else str(raw_name)
    m = _URL.match(raw)
    clean = normalize_text(m.group(1) if m else raw)
    if " " in clean:
        clean = _SINGLE_LETTERS.sub(lambda x: x.group(0).replace(" ", ""), clean)
    if clean in NULL_VALUES:
        clean = ""
    if clean and _DOMAIN_TAIL_RX.search(clean):
        clean = _DOMAIN_TAIL_RX.sub("", clean).strip()
    if clean and _HONORIFIC_RX.match(clean + " "):
        clean = _HONORIFIC_RX.sub("", clean + " ").strip()
    core, form = _strip_legal(clean, country) if clean else ("", "")
    raw_toks = core.split()
    toks = []
    for t in raw_toks:
        t = NAME_ABBREV.get(t, t)
        if t and t not in NAME_STOPWORDS and t not in PREFIX_TITLES:
            toks.append(t)
    if not toks and core:
        toks = [NAME_ABBREV.get(t, t) for t in raw_toks if t]
    toks = tuple(_intern(t) for t in dict.fromkeys(toks))
    core_name = " ".join(toks)
    name_key = " ".join(sorted(toks))
    digits = tuple(sorted({d for d in _DIGITS.findall(core_name)}))

    # Phonetic soundex of top tokens
    phonetics = tuple(_intern(soundex(t)) for t in toks[:3] if len(t) >= 2)

    # Anchor brand token (first distinctive word)
    anchor = ""
    for t in toks:
        if len(t) >= 3 and not t.isdigit():
            anchor = _intern(t)
            break

    # Token bi-grams
    bigrams = tuple(_intern(f"{toks[i]}_{toks[i+1]}") for i in range(len(toks) - 1))

    # Spaceless compact names
    name_compact = _intern(core_name.replace(" ", "")) if core_name else ""
    name_compact_sorted = _intern("".join(sorted(toks))) if toks else ""

    return clean, core_name, toks, name_key, form, digits, phonetics, anchor, bigrams, name_compact, name_compact_sorted


def _addr_parts(raw_addr, country: str):
    raw = "" if raw_addr is None else str(raw_addr)
    norm = normalize_text(raw)
    if norm in NULL_VALUES or len(norm) < 3:
        return "", (), "", (), True, ()
    pat = POSTAL_PATTERNS.get(country, DEFAULT_POSTAL)
    postal, span = "", None
    for m in pat.finditer(norm):
        postal, span = "".join(g for g in m.groups() if g), m.span()
    rest = norm if span is None else norm[: span[0]] + " " + norm[span[1]:]
    nums = tuple(dict.fromkeys(_DIGITS.findall(rest)))[:4]
    words = []
    for t in norm.split():
        t = ADDR_ABBREV.get(t, t)
        if t:
            words.append(t)
    clean = " ".join(words)
    toks = tuple(_intern(t) for t in dict.fromkeys(w for w in words if len(w) >= 2 and not w.isdigit()))

    # Extract distinct clean comma segments for locality blocking
    segments = []
    if "," in raw:
        for seg in raw.split(","):
            s_norm = normalize_text(seg)
            if s_norm and 2 <= len(s_norm) <= 35 and not s_norm.isdigit() and s_norm not in NULL_VALUES:
                segments.append(_intern(s_norm))
    segments = tuple(dict.fromkeys(segments))[:5]

    return clean, toks, postal, nums, False, segments


def preprocess_records(df: pd.DataFrame) -> Dict[str, list]:
    """Returns a compact column store (dict of equal-length lists) for records."""
    ids = df["entity_id"].astype(str).str.strip().tolist()
    names = df["business_name"].tolist() if "business_name" in df else [""] * len(ids)
    addrs = df["business_address"].tolist() if "business_address" in df else [""] * len(ids)
    countries = (
        df["country"].fillna("").astype(str).str.strip().tolist() if "country" in df else [""] * len(ids)
    )
    cols = {k: [] for k in (
        "name_clean", "name_core", "name_tokens", "name_key", "legal", "name_digits",
        "name_phonetics", "name_anchor", "name_bigrams", "name_compact", "name_compact_sorted",
        "addr_clean", "addr_tokens", "postal", "nums", "addr_missing", "addr_segments",
    )}
    for raw_name, raw_addr, country in zip(names, addrs, countries):
        n_clean, n_core, n_toks, n_key, legal, digits, phonetics, anchor, bigrams, n_comp, n_comps = _name_parts(raw_name, country)
        a_clean, a_toks, postal, nums, missing, segments = _addr_parts(raw_addr, country)
        cols["name_clean"].append(n_clean)
        cols["name_core"].append(n_core)
        cols["name_tokens"].append(n_toks)
        cols["name_key"].append(n_key)
        cols["legal"].append(legal)
        cols["name_digits"].append(digits)
        cols["name_phonetics"].append(phonetics)
        cols["name_anchor"].append(anchor)
        cols["name_bigrams"].append(bigrams)
        cols["name_compact"].append(n_comp)
        cols["name_compact_sorted"].append(n_comps)
        cols["addr_clean"].append(a_clean)
        cols["addr_tokens"].append(a_toks)
        cols["postal"].append(postal)
        cols["nums"].append(nums)
        cols["addr_missing"].append(missing)
        cols["addr_segments"].append(segments)
    cols["entity_id"] = ids
    cols["country"] = countries
    return cols


def char_grams(core_name: str, n: int = 3) -> set:
    s = core_name.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i: i + n] for i in range(len(s) - n + 1)}
