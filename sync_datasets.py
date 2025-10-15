import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Set, Tuple
import re
import requests
from huggingface_hub import HfApi, list_repo_files, create_repo


LOBSTER_DATASETS = {
    "AMZN": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "AAPL": {"levels": [1, 5, 10, 30, 50], "date": "2012-06-21"},
    "GOOG": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "INTC": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "MSFT": {"levels": [1, 5, 10, 30, 50], "date": "2012-06-21"},
    "SPY": {"levels": [30, 50], "date": "2012-06-21"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sync LOBSTER datasets")
    parser.add_argument("--repo-id", type=str, required=True, help="HF repo ID")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--keep-local", action="store_true", help="Keep local files")
    parser.add_argument("--download-only", action="store_true", help="No HF upload")
    parser.add_argument(
        "--tickers",
        nargs="+",
        choices=list(LOBSTER_DATASETS.keys()),
        help="Specific tickers",
    )
    parser.add_argument("--private", action="store_true", help="Private repo")
    parser.add_argument(
        "--force-readme",
        action="store_true",
        help="Force update README even if no changes",
    )
    return parser.parse_args()


def get_local_datasets() -> Set[Tuple[str, int]]:
    local = set()
    for path in Path(".").glob("LOBSTER_SampleFile_*"):
        if path.is_dir():
            parts = path.name.split("_")
            if len(parts) >= 5:
                ticker = parts[2]
                levels = int(parts[4])
                local.add((ticker, levels))
    return local


def get_hf_datasets(repo_id: str) -> Set[Tuple[str, int]]:
    try:
        files = list_repo_files(repo_id, repo_type="dataset")

        # Track datasets with both message and orderbook files
        dataset_files = {}
        for file in files:
            match = re.match(
                r"LOBSTER_SampleFile_(\w+)_(\d{4}-\d{2}-\d{2})_(\d+)/.*_(message|orderbook)_\d+\.csv",
                file,
            )
            if match:
                ticker, date, levels, file_type = match.groups()
                key = (ticker, int(levels))
                if key not in dataset_files:
                    dataset_files[key] = {"message": False, "orderbook": False}
                dataset_files[key][file_type] = True

        # Only include datasets with both files
        hf_datasets = set()
        for key, files_present in dataset_files.items():
            if files_present["message"] and files_present["orderbook"]:
                hf_datasets.add(key)
            else:
                ticker, levels = key
                missing = [k for k, v in files_present.items() if not v]
                print(
                    f"Warning: {ticker} ({levels} levels) missing {', '.join(missing)} file(s)"
                )

        return hf_datasets
    except Exception as e:
        print(f"Warning: HF repo access failed: {e}")
        print("Assuming empty repo")
        return set()


def get_available_datasets(tickers: List[str] = None) -> Set[Tuple[str, int]]:
    available = set()
    for ticker, info in LOBSTER_DATASETS.items():
        if tickers and ticker not in tickers:
            continue
        for level in info["levels"]:
            available.add((ticker, level))
    return available


def download_lobster_dataset(ticker: str, levels: int, dest_dir: Path) -> bool:
    date = LOBSTER_DATASETS[ticker]["date"]
    dir_name = f"LOBSTER_SampleFile_{ticker}_{date}_{levels}"
    full_path = dest_dir / dir_name

    zip_filename = f"LOBSTER_SampleFile_{ticker}_{date}_{levels}.zip"
    zip_url = f"https://lobsterdata.com/info/sample/{zip_filename}"

    print(f"  Downloading {ticker} ({levels} levels)...")

    try:
        print(f"    Fetching {zip_filename}...")
        response = requests.get(zip_url, timeout=120)
        response.raise_for_status()
        print(f"    Downloaded ({len(response.content) / 1024 / 1024:.1f} MB)")

        print("    Extracting...")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = Path(tmp_file.name)

        try:
            with tempfile.TemporaryDirectory() as tmp_extract_dir:
                with zipfile.ZipFile(tmp_path, "r") as zip_ref:
                    zip_ref.extractall(tmp_extract_dir)

                tmp_extract_path = Path(tmp_extract_dir)
                csv_files = list(tmp_extract_path.glob("*.csv"))

                if not csv_files:
                    print("    Error: No CSV files in archive")
                    return False

                full_path.mkdir(parents=True, exist_ok=True)
                for csv_file in csv_files:
                    shutil.move(str(csv_file), str(full_path / csv_file.name))

                print(f"    Extracted {len(csv_files)} files to {dir_name}")
            return True
        finally:
            tmp_path.unlink()

    except requests.exceptions.HTTPError as e:
        print(f"    Error: Download failed: {e}")
        return False
    except zipfile.BadZipFile:
        print("    Error: Invalid ZIP archive")
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def upload_to_hf(repo_id: str, dataset_dir: Path) -> bool:
    try:
        api = HfApi()
        print(f"  Uploading {dataset_dir.name}...")
        api.upload_folder(
            folder_path=str(dataset_dir),
            path_in_repo=dataset_dir.name,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"  Uploaded {dataset_dir.name}")
        return True
    except Exception as e:
        print(f"  Error: Upload failed: {e}")
        return False


def cleanup_local_dataset(dataset_dir: Path):
    try:
        shutil.rmtree(dataset_dir)
        print(f"  Cleaned up: {dataset_dir.name}")
    except Exception as e:
        print(f"  Error: Cleanup failed: {e}")


def create_hf_repository(repo_id: str, private: bool = False) -> bool:
    try:
        create_repo(
            repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True
        )
        print(f"Repository ready: {repo_id}")
        return True
    except Exception as e:
        print(f"Error: Repo creation failed: {e}")
        return False


def generate_readme(hf_datasets: Set[Tuple[str, int]]) -> str:
    ticker_datasets = {}
    for ticker, levels in sorted(hf_datasets):
        if ticker not in ticker_datasets:
            ticker_datasets[ticker] = []
        ticker_datasets[ticker].append(levels)

    ticker_list = []
    for ticker in sorted(ticker_datasets.keys()):
        levels = sorted(ticker_datasets[ticker])
        levels_str = ", ".join(str(level) for level in levels)
        ticker_list.append(f"- **{ticker}**: {levels_str} levels")

    configs = []
    for ticker in sorted(ticker_datasets.keys()):
        for level in sorted(ticker_datasets[ticker]):
            date = LOBSTER_DATASETS[ticker]["date"]
            config_name = f"{ticker}_{level}levels"

            configs.append(
                {
                    "name": config_name,
                    "message_path": f"LOBSTER_SampleFile_{ticker}_{date}_{level}/*_message_{level}.csv",
                    "orderbook_path": f"LOBSTER_SampleFile_{ticker}_{date}_{level}/*_orderbook_{level}.csv",
                }
            )

    yaml_configs = []
    for config in configs:
        yaml_configs.append(f"""  - config_name: {config["name"]}
    data_files:
      - split: message
        path: "{config["message_path"]}"
      - split: orderbook
        path: "{config["orderbook_path"]}\"""")

    yaml_section = "---\nconfigs:\n" + "\n".join(yaml_configs) + "\n---"

    readme = f"""{yaml_section}

# LOBSTER Sample Data

LOBSTER sample L3 order book data for June 21, 2012

## Dataset Description

LOBSTER provides limit order book data from NASDAQ TotalView-ITCH messages

{chr(10).join(ticker_list)}

## File Structure

Each tickerlevel combination has two CSV files:

1. **Message file** (`*_message_*.csv`): Order book events
   - Columns: time, event_type, order_id, size, price, direction
   - Types: 1=New, 2=Cancel, 3=Delete, 4=Exec(Visible), 5=Exec(Hidden), 7=Halt

2. **Orderbook file** (`*_orderbook_*.csv`): Order book snapshots
   - Columns: ask_price_1, ask_size_1, bid_price_1, bid_size_1, ... (up to N levels)
   - Prices in fixed-point (multiply by 10^-4)

## Source

Data provided by [LOBSTER](https://lobsterdata.com/)
"""
    return readme


def upload_readme(repo_id: str, hf_datasets: Set[Tuple[str, int]]) -> bool:
    try:
        api = HfApi()
        readme_content = generate_readme(hf_datasets)
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("  README.md updated")
        return True
    except Exception as e:
        print(f"  Error: README upload failed: {e}")
        return False


def upload_gitattributes(repo_id: str) -> bool:
    try:
        api = HfApi()
        gitattributes_content = "*.csv -diff -merge\n"
        api.upload_file(
            path_or_fileobj=gitattributes_content.encode(),
            path_in_repo=".gitattributes",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("  .gitattributes uploaded")
        return True
    except Exception as e:
        print(f"  Error: .gitattributes upload failed: {e}")
        return False


def main():
    args = parse_args()

    print("=" * 60)
    print("LOBSTER Dataset Sync")
    print("=" * 60)
    print()

    if not args.dry_run and not args.download_only:
        print("Creating/verifying HF repository...")
        print()
        if not create_hf_repository(args.repo_id, args.private):
            print("Repo creation failed. Exiting.")
            return
        print()

    print("Checking current state...")
    print()

    available = get_available_datasets(args.tickers)
    local = get_local_datasets()
    hf = get_hf_datasets(args.repo_id)

    print(f"Available on lobsterdata.com: {len(available)}")
    print(f"Local: {len(local)}")
    print(f"HuggingFace: {len(hf)}")
    print()

    missing_from_local = available - local
    missing_from_hf = available - hf
    ready_to_upload = local - hf
    can_cleanup = local & hf

    print("Analysis:")
    print()
    print(f"To download: {len(missing_from_local)}")
    if missing_from_local:
        for ticker, levels in sorted(missing_from_local):
            print(f"  - {ticker} ({levels} levels)")
    print()

    print(f"To upload: {len(missing_from_hf)}")
    if missing_from_hf:
        for ticker, levels in sorted(missing_from_hf):
            print(f"  - {ticker} ({levels} levels)")
    print()

    print(f"Can cleanup (already on HF): {len(can_cleanup)}")
    if can_cleanup and not args.keep_local:
        for ticker, levels in sorted(can_cleanup):
            print(f"  - {ticker} ({levels} levels)")
    print()

    if args.dry_run:
        print("DRY RUN - No changes made")
        return

    if missing_from_local:
        print("Downloading from lobsterdata.com...")
        print()
        for ticker, levels in sorted(missing_from_local):
            success = download_lobster_dataset(ticker, levels, Path("."))
            if success:
                print(f"  Downloaded {ticker} ({levels} levels)")
                if (ticker, levels) in missing_from_hf:
                    ready_to_upload.add((ticker, levels))
            else:
                print(f"  Failed {ticker} ({levels} levels)")
        print()

    if not args.download_only and ready_to_upload:
        print("Uploading to HuggingFace...")
        print()
        for ticker, levels in sorted(ready_to_upload):
            date = LOBSTER_DATASETS[ticker]["date"]
            dir_name = f"LOBSTER_SampleFile_{ticker}_{date}_{levels}"
            dataset_dir = Path(dir_name)

            if not dataset_dir.exists():
                print(f"  Skipping {dir_name} (not found)")
                continue

            success = upload_to_hf(args.repo_id, dataset_dir)
            if success and not args.keep_local:
                can_cleanup.add((ticker, levels))
        print()

    if not args.keep_local and can_cleanup:
        print("Cleaning up local datasets...")
        print()
        for ticker, levels in sorted(can_cleanup):
            date = LOBSTER_DATASETS[ticker]["date"]
            dir_name = f"LOBSTER_SampleFile_{ticker}_{date}_{levels}"
            dataset_dir = Path(dir_name)
            if dataset_dir.exists():
                cleanup_local_dataset(dataset_dir)
        print()

    if (
        not args.dry_run
        and not args.download_only
        and (ready_to_upload or missing_from_hf or args.force_readme)
    ):
        print("Updating metadata...")
        print()
        final_hf = get_hf_datasets(args.repo_id)
        upload_readme(args.repo_id, final_hf)
        upload_gitattributes(args.repo_id)
        print()

    print("=" * 60)
    print("Sync complete")
    if not args.dry_run and not args.download_only:
        print(f"https://huggingface.co/datasets/{args.repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
