import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def build_dataset_patterns(include_train: bool) -> list[str]:
    patterns = [
        "README.md",
        ".gitattributes",
        "data/test_task-*",
        "data/test_website-*",
        "data/test_domain-*",
    ]
    if include_train:
        patterns.append("data/train-*")
    return patterns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the static Multimodal-Mind2Web test splits and official score file."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root directory for local dataset artifacts.",
    )
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Also download the train split parquet shards.",
    )
    parser.add_argument(
        "--skip-scores",
        action="store_true",
        help="Skip downloading the official scores_all_data.pkl file.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    multimodal_dir = output_root / "multimodal_mind2web"
    multimodal_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Multimodal-Mind2Web into: {multimodal_dir}")
    snapshot_download(
        repo_id="osunlp/Multimodal-Mind2Web",
        repo_type="dataset",
        local_dir=str(multimodal_dir),
        local_dir_use_symlinks=False,
        allow_patterns=build_dataset_patterns(args.include_train),
    )

    if not args.skip_scores:
        scores_dir = output_root / "mind2web_aux"
        scores_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading official candidate scores into: {scores_dir}")
        snapshot_download(
            repo_id="osunlp/Mind2Web",
            repo_type="dataset",
            local_dir=str(scores_dir),
            local_dir_use_symlinks=False,
            allow_patterns=["scores_all_data.pkl", "README.md"],
        )

    print("Done.")


if __name__ == "__main__":
    main()
