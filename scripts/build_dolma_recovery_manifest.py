#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


CONTENT_RANGE_RE = re.compile(r"/(\d+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a whole-object recovery manifest for Dolma shards missed by Storage Transfer."
    )
    parser.add_argument("--selected-urls", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default="dolma-v1_7-400b")
    parser.add_argument("--bucket", default="fractal-llm-data-us-central1-81f2add4")
    parser.add_argument("--size-workers", type=int, default=32)
    parser.add_argument("--size-timeout", type=float, default=30.0)
    return parser.parse_args()


def source_name(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    try:
        index = parts.index("dolma-v1_7")
    except ValueError as exc:
        raise ValueError(f"url does not look like a Dolma v1.7 object: {url}") from exc
    if index + 1 >= len(parts):
        raise ValueError(f"url is missing Dolma source path: {url}")
    return parts[index + 1]


def transfer_object_name(url: str, *, dataset_id: str) -> str:
    parsed = urlparse(url)
    source = source_name(url)
    source_path = parsed.path.lstrip("/")
    return f"datasets/{dataset_id}/raw/{source}/{parsed.netloc}/{source_path}"


def load_urls(path: Path) -> list[str]:
    urls = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not urls:
        raise SystemExit(f"no URLs found in {path}")
    return sorted(urls)


def fetch_size(url: str, timeout: float) -> int:
    proc = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--head",
            "--max-time",
            str(timeout),
            "--header",
            "Range: bytes=0-0",
            "--header",
            "Accept-Encoding: identity",
            url,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for line in proc.stdout.splitlines():
        if line.lower().startswith("content-range:"):
            match = CONTENT_RANGE_RE.search(line.strip())
            if match:
                return int(match.group(1))
            raise RuntimeError(f"invalid Content-Range for {url}: {line}")
    raise RuntimeError(f"missing Content-Range for {url}")


def fetch_sizes(urls: list[str], *, workers: int, timeout: float) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fetch_size, url, timeout): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                sizes[url] = future.result()
            except Exception as exc:  # noqa: BLE001 - make all metadata failures visible.
                print(f"warning: failed to fetch size for {url}: {exc}", file=sys.stderr)
    missing = sorted(set(urls) - set(sizes))
    if missing:
        raise SystemExit(f"failed to fetch sizes for {len(missing)} URLs")
    return sizes


def list_existing_objects(bucket: str, dataset_id: str) -> dict[str, int]:
    prefix = f"gs://{bucket}/datasets/{dataset_id}/raw/"
    proc = subprocess.run(
        ["gcloud", "storage", "ls", "-l", "-r", prefix],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    sizes: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) < 3 or not fields[0].isdigit() or not fields[-1].startswith(f"gs://{bucket}/"):
            continue
        name = fields[-1][len(f"gs://{bucket}/") :]
        sizes[name] = int(fields[0])
    return sizes


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    urls = load_urls(args.selected_urls)
    sizes = fetch_sizes(urls, workers=args.size_workers, timeout=args.size_timeout)
    existing = list_existing_objects(args.bucket, args.dataset_id)

    all_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for url in urls:
        object_name = transfer_object_name(url, dataset_id=args.dataset_id)
        expected_size = sizes[url]
        existing_size = existing.get(object_name)
        if existing_size == expected_size:
            status = "present"
        elif existing_size is None:
            status = "missing"
        else:
            status = "size_mismatch"
        row = {
            "url": url,
            "bucket": args.bucket,
            "object": object_name,
            "source": source_name(url),
            "expected_size": expected_size,
            "existing_size": existing_size,
            "status": status,
        }
        all_rows.append(row)
        status_counts[status] += 1
        if status != "present":
            missing_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "all_selected_recovery_status.jsonl", all_rows)
    write_jsonl(args.output_dir / "missing_recovery_manifest.jsonl", missing_rows)

    summary = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_id": args.dataset_id,
        "bucket": args.bucket,
        "selected_urls": len(urls),
        "missing_urls": len(missing_rows),
        "status_counts": dict(status_counts),
        "selected_bytes": sum(row["expected_size"] for row in all_rows),
        "missing_bytes": sum(row["expected_size"] for row in missing_rows),
        "present_bytes": sum(row["expected_size"] for row in all_rows if row["status"] == "present"),
    }
    (args.output_dir / "recovery_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "missing_recovery_manifest.jsonl")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
