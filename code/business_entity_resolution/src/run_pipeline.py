"""End-to-end reproduction: data -> blocking -> matching -> output/*.tsv (+ validation)."""
import argparse
import sys
from pathlib import Path

# Add src directory to path
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from er.pipeline import (Config, choose_alpha, choose_top_k, load_artifacts, log, prepare_training,  # noqa: E402
                         run_test_inference, save_artifacts, train_and_tune)
from er.validate import run_official_validator, validate_outputs  # noqa: E402
from er.package import build_submission_zip, methodology_markdown  # noqa: E402


def find_default_dir(marker: str, search_roots: list) -> Path:
    for root in search_roots:
        if root.exists():
            for p in root.rglob(marker):
                return p.parent
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", default=None, help="Path to dataset/train folder")
    ap.add_argument("--test-dir", default=None, help="Path to dataset/test folder")
    ap.add_argument("--output-dir", default=None, help="Directory to save output TSVs")
    ap.add_argument("--artifacts-dir", default="artifacts", help="Directory for model artifacts")
    ap.add_argument("--n-train-s1", type=int, default=300_000, help="Number of S1 training entities")
    ap.add_argument("--top-k", type=int, default=40, help="0 = choose from recall curve; or specify e.g. 40")
    ap.add_argument("--alpha", type=float, default=0.03, help="Score-ratio cutoff for candidate filtering")
    ap.add_argument("--reuse-artifacts", action="store_true", help="Skip training, load saved model")
    ap.add_argument("--team-name", default="Amritians", help="Team name for final zip package")
    args = ap.parse_args()

    # Search default paths if not provided
    base_dir = Path.cwd() if (Path.cwd() / "student_resource").exists() else SRC_DIR.parents[2]
    search_dirs = [base_dir, SRC_DIR.parents[1], Path.cwd()]

    train_dir = Path(args.train_dir) if args.train_dir else find_default_dir("train_ground_truth.tsv", search_dirs)
    test_dir = Path(args.test_dir) if args.test_dir else find_default_dir("test_source1.tsv", search_dirs)
    output_dir = Path(args.output_dir) if args.output_dir else (base_dir / "output")

    if train_dir is None or not train_dir.exists():
        raise FileNotFoundError("Could not locate train_ground_truth.tsv. Please pass --train-dir.")
    if test_dir is None or not test_dir.exists():
        raise FileNotFoundError("Could not locate test_source1.tsv. Please pass --test-dir.")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    log(f"Train directory : {train_dir}")
    log(f"Test directory  : {test_dir}")
    log(f"Output directory: {output_dir}")
    log(f"Artifacts dir   : {artifacts_dir}")

    cfg = Config(
        train_dir=str(train_dir), test_dir=str(test_dir), output_dir=str(output_dir),
        artifacts_dir=str(artifacts_dir), n_train_s1=args.n_train_s1, top_k=args.top_k,
        score_ratio_min=args.alpha,
    )

    if args.reuse_artifacts and (artifacts_dir / "run_meta.json").exists():
        log("Reusing existing model and artifacts...")
        model, meta = load_artifacts(cfg)
        top_k = meta["top_k"]
        rule = meta["rule"]
        alpha = meta["alpha"]
        exclusive = meta["exclusive"]
        per_country_rules = meta.get("per_country_rules", {})
        result = {"model": model, "rule": rule, "per_country_rules": per_country_rules,
                  "top_k": top_k, "alpha": alpha, "exclusive": exclusive, "metrics": meta["metrics"]}
    else:
        td = prepare_training(cfg)
        top_k, curve = choose_top_k(td, cfg)
        log("\nRecall curve @ k:")
        log(curve.to_string(index=False))
        alpha, alpha_table = choose_alpha(td, top_k, cfg)
        log(f"\nChosen top_k = {top_k}, score-ratio cutoff alpha = {alpha}")
        result = train_and_tune(td, top_k, alpha, cfg)
        log(f"\nValidation Metrics: {result['metrics']}")
        log(f"Tuned Decision Rule: {result['rule']}")
        if result.get("per_country_rules"):
            log(f"Per-Country Rules: {result['per_country_rules']}")
        save_artifacts(cfg, result)
        model = result["model"]
        rule = result["rule"]
        per_country_rules = result.get("per_country_rules", {})
        exclusive = result["exclusive"]
        del td
        import gc; gc.collect()

    # Run streaming inference
    summary = run_test_inference(cfg, model, top_k, rule, alpha, exclusive,
                                per_country_rules=per_country_rules)
    log("\nTest inference summary per country:")
    log(summary.to_string(index=False))

    # Validate output format
    ok = validate_outputs(cfg.output_dir, cfg.test_dir)
    run_official_validator(cfg.output_dir, cfg.test_dir)

    # Build methodology doc and final submission zip
    methodology = methodology_markdown(args.team_name, result, summary)
    (base_dir / "Documentation_template.md").write_text(methodology, encoding="utf-8")
    zip_path = build_submission_zip(args.team_name, base_dir, output_dir, SRC_DIR, methodology)
    log(f"\nSubmission zip generated: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
