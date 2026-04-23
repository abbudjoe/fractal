#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_REFERENCE_TOKENS = 3_000_000_000_000
DEFAULT_TARGET_TOKENS = 400_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build source-preserving Storage Transfer URL-list manifests for a Dolma scale subset."
    )
    parser.add_argument("--url-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default="dolma-v1_7-400b")
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--reference-total-tokens", type=int, default=DEFAULT_REFERENCE_TOKENS)
    parser.add_argument("--target-compressed-bytes", type=int)
    parser.add_argument("--size-cache", type=Path)
    parser.add_argument("--fetch-sizes", action="store_true")
    parser.add_argument("--head-workers", type=int, default=32)
    parser.add_argument("--head-timeout", type=float, default=30.0)
    parser.add_argument("--bucket-suffix", default="81f2add4")
    parser.add_argument("--canonical-region", default="us-central1")
    parser.add_argument(
        "--include-sources",
        help="Comma-separated Dolma source names to include. Useful for small transfer smokes.",
    )
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


def object_name(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    try:
        index = parts.index("dolma-v1_7")
    except ValueError as exc:
        raise ValueError(f"url does not look like a Dolma v1.7 object: {url}") from exc
    return "/".join(parts[index + 1 :])


def load_urls(path: Path) -> list[str]:
    urls = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not urls:
        raise SystemExit(f"no URLs found in {path}")
    return sorted(urls)


def load_size_cache(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"size cache must be a JSON object: {path}")
    out: dict[str, int] = {}
    for url, size in payload.items():
        if isinstance(url, str) and isinstance(size, int) and size >= 0:
            out[url] = size
    return out


def save_size_cache(path: Path | None, cache: dict[str, int]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(cache.items())), indent=2) + "\n", encoding="utf-8")


def fetch_content_length(url: str, timeout: float) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        header = response.headers.get("Content-Length")
        if header is None:
            raise RuntimeError("missing Content-Length")
        return int(header)


def populate_sizes(urls: list[str], cache: dict[str, int], *, workers: int, timeout: float) -> dict[str, int]:
    missing = [url for url in urls if url not in cache]
    if not missing:
        return cache
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_content_length, url, timeout): url for url in missing}
        for future in as_completed(futures):
            url = futures[future]
            try:
                cache[url] = future.result()
            except Exception as exc:  # noqa: BLE001 - report all URL metadata failures.
                print(f"warning: failed to fetch size for {url}: {exc}", file=sys.stderr)
            completed += 1
            if completed % 100 == 0:
                print(f"fetched sizes for {completed}/{len(missing)} missing URLs", file=sys.stderr, flush=True)
    return cache


def select_by_fraction(urls_by_source: dict[str, list[str]], fraction: float) -> list[str]:
    selected: list[str] = []
    for source, urls in sorted(urls_by_source.items()):
        count = max(1, math.ceil(len(urls) * fraction))
        selected.extend(sorted(urls)[:count])
    return sorted(selected)


def select_by_bytes(
    urls_by_source: dict[str, list[str]],
    sizes: dict[str, int],
    target_bytes: int,
) -> list[str]:
    total_bytes = sum(sizes[url] for urls in urls_by_source.values() for url in urls if url in sizes)
    if total_bytes <= 0:
        raise SystemExit("cannot select by bytes: no URL sizes available")
    fraction = min(1.0, target_bytes / total_bytes)
    selected: list[str] = []
    for source, urls in sorted(urls_by_source.items()):
        source_urls = [url for url in sorted(urls) if url in sizes]
        source_total = sum(sizes[url] for url in source_urls)
        source_target = max(1, math.ceil(source_total * fraction))
        running = 0
        for url in source_urls:
            if running >= source_target:
                break
            selected.append(url)
            running += sizes[url]
    return sorted(selected)


def write_tsv(path: Path, urls: list[str], sizes: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("TsvHttpData-1.0\n")
        for url in sorted(urls):
            size = sizes.get(url)
            if size is None:
                handle.write(f"{url}\n")
            else:
                handle.write(f"{url}\t{size}\n")


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.target_tokens <= 0:
        raise SystemExit("--target-tokens must be positive")
    if args.reference_total_tokens <= 0:
        raise SystemExit("--reference-total-tokens must be positive")

    all_urls = load_urls(args.url_list)
    include_sources = (
        {source.strip() for source in args.include_sources.split(",") if source.strip()}
        if args.include_sources
        else None
    )
    urls = [
        url
        for url in all_urls
        if include_sources is None or source_name(url) in include_sources
    ]
    if not urls:
        raise SystemExit(f"source filter selected no URLs: {sorted(include_sources or [])}")
    urls_by_source: dict[str, list[str]] = defaultdict(list)
    for url in urls:
        urls_by_source[source_name(url)].append(url)

    size_cache = load_size_cache(args.size_cache)
    if args.fetch_sizes:
        size_cache = populate_sizes(
            urls,
            size_cache,
            workers=max(1, args.head_workers),
            timeout=args.head_timeout,
        )
        save_size_cache(args.size_cache, size_cache)

    if args.target_compressed_bytes is not None:
        selected = select_by_bytes(urls_by_source, size_cache, args.target_compressed_bytes)
        selection_basis = "target-compressed-bytes"
        target_bytes = args.target_compressed_bytes
    elif all(url in size_cache for url in urls):
        total_bytes = sum(size_cache.values())
        token_fraction = min(1.0, args.target_tokens / args.reference_total_tokens)
        target_bytes = math.ceil(total_bytes * token_fraction)
        selected = select_by_bytes(urls_by_source, size_cache, target_bytes)
        selection_basis = "token-fraction-to-compressed-bytes"
    else:
        token_fraction = min(1.0, args.target_tokens / args.reference_total_tokens)
        selected = select_by_fraction(urls_by_source, token_fraction)
        selection_basis = "token-fraction-to-file-count"
        target_bytes = None

    selected_by_source: dict[str, list[str]] = defaultdict(list)
    for url in selected:
        selected_by_source[source_name(url)].append(url)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transfer_dir = args.output_dir / "transfer-url-lists"
    transfer_dir.mkdir(parents=True, exist_ok=True)

    source_rows: list[dict[str, Any]] = []
    for source, source_urls in sorted(selected_by_source.items()):
        tsv_path = transfer_dir / f"{source}.tsv"
        write_tsv(tsv_path, source_urls, size_cache)
        source_rows.append(
            {
                "source": source,
                "url_count": len(source_urls),
                "known_compressed_bytes": sum(size_cache.get(url, 0) for url in source_urls),
                "transfer_tsv": str(tsv_path.relative_to(args.output_dir)),
                "destination_prefix": f"datasets/{args.dataset_id}/raw/{source}/",
            }
        )

    write_lines(args.output_dir / "selected_urls.txt", selected)
    write_lines(args.output_dir / "all_urls.txt", urls)

    known_selected_bytes = sum(size_cache.get(url, 0) for url in selected)
    known_total_bytes = sum(size_cache.get(url, 0) for url in urls)
    reference_url_count = len(all_urls)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_id": args.dataset_id,
        "source_dataset": "allenai/dolma",
        "source_version": "v1_7",
        "source_filter": sorted(include_sources) if include_sources else None,
        "target_tokens": args.target_tokens,
        "reference_total_tokens": args.reference_total_tokens,
        "selection_basis": selection_basis,
        "target_compressed_bytes": target_bytes,
        "reference_url_count": reference_url_count,
        "url_count": len(urls),
        "selected_url_count": len(selected),
        "known_total_compressed_bytes": known_total_bytes,
        "known_selected_compressed_bytes": known_selected_bytes,
        "estimated_selected_tokens": int(args.reference_total_tokens * (known_selected_bytes / known_total_bytes))
        if known_total_bytes > 0
        else int(args.reference_total_tokens * (len(selected) / reference_url_count)),
        "canonical_bucket": f"gs://fractal-llm-data-{args.canonical_region}-{args.bucket_suffix}",
        "canonical_raw_prefix": f"datasets/{args.dataset_id}/raw/",
        "canonical_manifest_prefix": f"datasets/{args.dataset_id}/manifests/",
        "sources": source_rows,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir / "manifest.json")
    print(
        f"selected_urls={len(selected)} estimated_tokens={manifest['estimated_selected_tokens']} "
        f"known_selected_compressed_bytes={known_selected_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
