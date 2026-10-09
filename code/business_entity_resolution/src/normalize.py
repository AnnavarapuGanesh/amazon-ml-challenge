import os
import re
import pandas as pd
from unidecode import unidecode
from concurrent.futures import ThreadPoolExecutor

LEGAL_SUFFIXES_TO_STRIP = [
    r'\bprivate limited\b', r'\bpvt ltd\b', r'\bpvt\b', r'\bltd\b', r'\blimited\b',
    r'\bincorporated\b', r'\binc\b', r'\bcorporation\b', r'\bcorp\b', r'\bllc\b',
    r'\bllp\b', r'\bco\b', r'\bcompany\b', r'\blp\b', r'\bpc\b', r'\bp c\b',
    r'\bgroup\b', r'\bholdings\b', r'\benterprises\b', r'\bsolutions\b',
    r'\bservices\b', r'\bsvc\b', r'\btechnologies\b', r'\btech\b',
    r'\bconsultancy\b', r'\bconsulting\b', r'\binternational\b', r'\bintl\b'
]

PREFIX_TITLES_TO_STRIP = [
    r'^\bsri\b', r'^\bsmt\b', r'^\bm/s\b', r'^\bmr\b', r'^\bmrs\b', r'^\bdr\b'
]

LEGAL_SYNONYMS = {
    r'\bcorporation\b': 'corp',
    r'\blimited\b': 'ltd',
    r'\bprivate\b': 'pvt',
    r'\bincorporated\b': 'inc',
    r'\bcompany\b': 'co',
    r'\bassociates\b': 'assoc',
    r'\btechnologies\b': 'tech',
    r'\bservices\b': 'svc',
    r'\band\b': '&'
}

STREET_ABBREVIATIONS = {
    r'\broad\b': 'rd',
    r'\bstreet\b': 'st',
    r'\bavenue\b': 'ave',
    r'\bdrive\b': 'dr',
    r'\bboulevard\b': 'blvd',
    r'\blane\b': 'ln',
    r'\bcourt\b': 'ct',
    r'\bplace\b': 'pl',
    r'\bsquare\b': 'sq',
    r'\bsuite\b': 'ste',
    r'\bapartment\b': 'apt',
    r'\bhouse number\b': 'hno',
    r'\bh no\b': 'hno',
    r'\bno\b': 'num'
}

PIN_POSTAL_REGEX = re.compile(r'\b(?:\d{5,6}(?:-\d{4})?)\b')
LANDMARK_REGEX = re.compile(r'\b(?:near|nr|opp|opposite|behind|c/o|c o)\s+[^,]+', re.IGNORECASE)

def normalize_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = unidecode(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_name(name: str):
    norm = normalize_text(name)
    if not norm:
        return "", "", ""
    
    std_name = norm
    for pat, repl in LEGAL_SYNONYMS.items():
        std_name = re.sub(pat, repl, std_name)
    std_name = re.sub(r'\s+', ' ', std_name).strip()

    core_name = std_name
    for pat in PREFIX_TITLES_TO_STRIP:
        core_name = re.sub(pat, '', core_name).strip()
    for pat in LEGAL_SUFFIXES_TO_STRIP:
        core_name = re.sub(pat, '', core_name).strip()
    core_name = re.sub(r'\s+', ' ', core_name).strip()
    if not core_name:
        core_name = std_name

    tokens = core_name.split()
    tokens.sort()
    sorted_core_name = " ".join(tokens)

    return std_name, core_name, sorted_core_name

def normalize_address(address: str):
    if not isinstance(address, str) or not address:
        return "", "", "", ""
    
    raw_unidecoded = unidecode(address).lower()
    
    pin_match = PIN_POSTAL_REGEX.findall(raw_unidecoded)
    postal_code = pin_match[0] if pin_match else ""
    
    landmark_match = LANDMARK_REGEX.findall(raw_unidecoded)
    landmark = landmark_match[0].strip() if landmark_match else ""
    
    norm_addr = re.sub(r'[^a-z0-9\s]', ' ', raw_unidecoded)
    for pat, repl in STREET_ABBREVIATIONS.items():
        norm_addr = re.sub(pat, repl, norm_addr)
    norm_addr = re.sub(r'\s+', ' ', norm_addr).strip()
    
    residual_addr = norm_addr
    if postal_code:
        residual_addr = re.sub(re.escape(postal_code), '', residual_addr)
    residual_addr = re.sub(r'\s+', ' ', residual_addr).strip()
    
    return norm_addr, residual_addr, postal_code, landmark

def normalize_dataframe(df: pd.DataFrame, label: str = "DataFrame") -> pd.DataFrame:
    df = df.copy()
    print(f"  [Normalize] {label}: {len(df)} rows...", flush=True)
    
    df['business_name'] = df['business_name'].fillna('').astype(str)
    df['business_address'] = df['business_address'].fillna('').astype(str)
    df['country'] = df['country'].fillna('').astype(str)
    
    num_workers = min(16, (os.cpu_count() or 4) * 2)
    
    # Memoize unique name normalization in parallel
    unique_names = df['business_name'].unique()
    print(f"  [Normalize] {label}: {len(unique_names)} unique names across {num_workers} threads...", flush=True)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        name_vals = list(executor.map(normalize_name, unique_names, chunksize=2000))
    name_map = dict(zip(unique_names, name_vals))
    
    name_results = df['business_name'].map(name_map)
    df['norm_name'] = [t[0] for t in name_results]
    df['core_name'] = [t[1] for t in name_results]
    df['sorted_core_name'] = [t[2] for t in name_results]
    
    # Memoize unique address normalization in parallel
    unique_addrs = df['business_address'].unique()
    print(f"  [Normalize] {label}: {len(unique_addrs)} unique addresses across {num_workers} threads...", flush=True)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        addr_vals = list(executor.map(normalize_address, unique_addrs, chunksize=2000))
    addr_map = dict(zip(unique_addrs, addr_vals))
    
    addr_results = df['business_address'].map(addr_map)
    df['norm_address'] = [t[0] for t in addr_results]
    df['residual_address'] = [t[1] for t in addr_results]
    df['postal_code'] = [t[2] for t in addr_results]
    df['landmark'] = [t[3] for t in addr_results]
    
    print(f"  [Normalize] {label} DONE!", flush=True)
    return df

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("Testing parallel memoized normalize.py...")
    test_names = [
        "Baba Technologies Private Limited",
        "Baba టెక్నాలజీస్ ప్రైవేట్ లిమిటెడ్",
        "Encore Cabs Pvt Ltd",
        "Sri ENCORE CABS PVT LTD.",
        "Straight Edge Book Store P.C."
    ]
    for name in test_names:
        print(f"RAW : {name}")
        print(f"NORM: {normalize_name(name)}")
        print("-" * 40)
