"""Submission validation conforming to official competition rules."""
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Set


def _read_ids(path: Path) -> Set[str]:
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        return {line.split("\t", 1)[0].strip() for line in f if line.strip()}


def _read_lists(path: Path, header: str, problems: list) -> Dict[str, list]:
    rows: Dict[str, list] = {}
    with open(path, "r", encoding="utf-8") as f:
        got = f.readline().rstrip("\r\n")
        if got != header:
            problems.append(f"{path.name}: header {got!r} != {header!r}")
        for n, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                problems.append(f"{path.name}:{n}: expected 2 tab-separated columns, got {len(parts)}")
                continue
            s1, ids = parts[0], [x for x in parts[1].split(",") if x]
            if s1 in rows:
                problems.append(f"{path.name}:{n}: duplicate source1_entity_id {s1}")
            if len(set(ids)) != len(ids):
                problems.append(f"{path.name}:{n}: duplicate ids in list for {s1}")
            rows[s1] = ids
    return rows


def validate_outputs(output_dir, test_dir, max_report: int = 10) -> bool:
    output_dir, test_dir = Path(output_dir), Path(test_dir)
    problems: list = []
    match = _read_lists(output_dir / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids", problems)
    cand = _read_lists(output_dir / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids", problems)
    s1_ids = _read_ids(test_dir / "test_source1.tsv")

    for name, rows in (("matching_results.tsv", match), ("candidate_pairs.tsv", cand)):
        missing, extra = s1_ids - rows.keys(), rows.keys() - s1_ids
        if missing:
            problems.append(f"{name}: {len(missing):,} Source-1 test entities missing")
        if extra:
            problems.append(f"{name}: {len(extra):,} unknown Source-1 ids")
        bad = sum(1 for ids in rows.values() for x in ids
                  if not (x.startswith("S2-") or x.startswith("S3-")))
        if bad:
            problems.append(f"{name}: {bad:,} ids do not start with S2- or S3-")
    not_subset = sum(1 for s1, ids in match.items() if not set(ids) <= set(cand.get(s1, ())))
    if not_subset:
        problems.append(f"{not_subset:,} entities have matches outside their candidate list")

    n = max(1, len(match))
    non_empty = sum(1 for v in match.values() if v)
    print(f"matching_results.tsv : {len(match):,} rows | non-empty {non_empty:,} ({non_empty / n:.1%}) | "
          f"avg matches {sum(map(len, match.values())) / n:.2f}")
    print(f"candidate_pairs.tsv  : {len(cand):,} rows | avg candidates / Source-1 entity "
          f"{sum(map(len, cand.values())) / max(1, len(cand)):.2f}")
    if problems:
        print(f"VALIDATION FAILED ({len(problems)} problems):")
        for p in problems[:max_report]:
            print("  -", p)
        return False
    print("VALIDATION PASSED: all challenge format rules satisfied.")
    return True


def find_official_validator(test_dir) -> Optional[Path]:
    for parent in Path(test_dir).resolve().parents:
        cand = parent / "utils" / "validate_submission.py"
        if cand.exists():
            return cand
    return None


def run_official_validator(output_dir, test_dir) -> Optional[int]:
    script = find_official_validator(test_dir)
    if script is None:
        print("Official utils/validate_submission.py not found next to the dataset; skipped.")
        return None
    cmd = [sys.executable, str(script), "--matching", str(Path(output_dir) / "matching_results.tsv"),
           "--candidate", str(Path(output_dir) / "candidate_pairs.tsv"), "--test-dir", str(test_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout[-4000:])
    if res.stderr.strip():
        print(res.stderr[-2000:])
    print(f"Official validator exit code: {res.returncode} ({'PASS' if res.returncode == 0 else 'FAIL'})")
    return res.returncode
