#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METADATA_TOKEN_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
)


class TokenProvider:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0

    def token(self) -> str:
        with self._lock:
            now = time.time()
            if self._token and now < self._expires_at - 300:
                return self._token
            request = urllib.request.Request(
                METADATA_TOKEN_URL,
                headers={"Metadata-Flavor": "Google"},
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception:
                proc = subprocess.run(
                    ["gcloud", "auth", "print-access-token"],
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                )
                self._token = proc.stdout.strip()
                self._expires_at = now + 1800
                return self._token
            self._token = payload["access_token"]
            self._expires_at = now + int(payload.get("expires_in", 3600))
            return self._token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover missing Dolma shards with whole-object HTTP downloads and GCS resumable uploads."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--work-dir", type=Path, default=Path("/mnt/fractal-dolma-recovery"))
    parser.add_argument("--events-log", type=Path, default=Path("recovery_events.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--final-log-bucket")
    parser.add_argument("--final-log-object")
    parser.add_argument("--summary-object")
    parser.add_argument("--curl-retries", type=int, default=12)
    parser.add_argument("--curl-speed-time", type=int, default=180)
    return parser.parse_args()


def read_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def gcs_json_request(
    token_provider: TokenProvider,
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    ok_missing: bool = False,
) -> dict[str, Any] | tuple[None, int, str]:
    request_headers = dict(headers or {})
    request_headers["Authorization"] = f"Bearer {token_provider.token()}"
    request = urllib.request.Request(url, method=method, data=body, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status == 204:
                return {}
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if ok_missing and exc.code == 404:
            return None, 404, "not found"
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GCS request failed {exc.code}: {detail}") from exc


def gcs_object_size(token_provider: TokenProvider, bucket: str, object_name: str) -> int | None:
    encoded = urllib.parse.quote(object_name, safe="")
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{encoded}?fields=size"
    result = gcs_json_request(token_provider, url, ok_missing=True)
    if isinstance(result, tuple):
        return None
    return int(result["size"])


def start_resumable_upload(
    token_provider: TokenProvider,
    bucket: str,
    object_name: str,
    size: int,
) -> str:
    query_name = urllib.parse.quote(object_name, safe="")
    url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=resumable&name={query_name}"
    metadata = json.dumps({"contentType": "application/json"}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Upload-Content-Type": "application/json",
        "X-Upload-Content-Length": str(size),
        "Authorization": f"Bearer {token_provider.token()}",
    }
    request = urllib.request.Request(url, method="POST", data=metadata, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        location = response.headers.get("Location")
        if not location:
            raise RuntimeError("GCS resumable upload did not return a Location header")
        return location


def upload_file(token_provider: TokenProvider, path: Path, bucket: str, object_name: str, size: int) -> None:
    upload_url = start_resumable_upload(token_provider, bucket, object_name, size)
    subprocess.run(
        [
            "curl",
            "--fail",
            "--show-error",
            "--silent",
            "--location",
            "--request",
            "PUT",
            "--header",
            "Content-Type: application/json",
            "--header",
            f"Content-Length: {size}",
            "--upload-file",
            str(path),
            upload_url,
        ],
        check=True,
    )


def download_file(row: dict[str, Any], path: Path, *, retries: int, speed_time: int) -> None:
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--show-error",
            "--silent",
            "--retry",
            str(retries),
            "--retry-delay",
            "5",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            str(speed_time),
            "--header",
            "Accept-Encoding: identity",
            "--output",
            str(path),
            row["url"],
        ],
        check=True,
    )


class EventWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, event: dict[str, Any]) -> None:
        event = {"time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **event}
        line = json.dumps(event, sort_keys=True)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            print(line, flush=True)


def recover_one(
    row: dict[str, Any],
    *,
    token_provider: TokenProvider,
    work_dir: Path,
    events: EventWriter,
    curl_retries: int,
    curl_speed_time: int,
) -> str:
    bucket = row["bucket"]
    object_name = row["object"]
    expected_size = int(row["expected_size"])
    current_size = gcs_object_size(token_provider, bucket, object_name)
    if current_size == expected_size:
        events.write({"status": "skipped_present", "url": row["url"], "object": object_name, "bytes": expected_size})
        return "skipped_present"

    work_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="dolma-", suffix=".json.gz.partial", dir=work_dir)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        events.write({"status": "download_start", "url": row["url"], "object": object_name, "bytes": expected_size})
        download_file(row, tmp_path, retries=curl_retries, speed_time=curl_speed_time)
        actual_size = tmp_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(f"downloaded size mismatch: expected {expected_size}, got {actual_size}")
        events.write({"status": "upload_start", "url": row["url"], "object": object_name, "bytes": expected_size})
        upload_file(token_provider, tmp_path, bucket, object_name, expected_size)
        uploaded_size = gcs_object_size(token_provider, bucket, object_name)
        if uploaded_size != expected_size:
            raise RuntimeError(f"uploaded size mismatch: expected {expected_size}, got {uploaded_size}")
        events.write({"status": "recovered", "url": row["url"], "object": object_name, "bytes": expected_size})
        return "recovered"
    except Exception as exc:  # noqa: BLE001 - log and keep other workers moving.
        events.write({"status": "failed", "url": row["url"], "object": object_name, "error": str(exc)})
        return "failed"
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def upload_small_file(token_provider: TokenProvider, path: Path, bucket: str, object_name: str) -> None:
    size = path.stat().st_size
    upload_file(token_provider, path, bucket, object_name, size)


def main() -> int:
    args = parse_args()
    rows = read_manifest(args.manifest, args.limit)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    events = EventWriter(args.events_log)
    token_provider = TokenProvider()

    events.write({"status": "run_start", "items": len(rows), "workers": args.workers})
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                recover_one,
                row,
                token_provider=token_provider,
                work_dir=args.work_dir,
                events=events,
                curl_retries=args.curl_retries,
                curl_speed_time=args.curl_speed_time,
            )
            for row in rows
        ]
        for future in as_completed(futures):
            status = future.result()
            counts[status] = counts.get(status, 0) + 1

    summary = {
        "status": "run_complete",
        "items": len(rows),
        "counts": counts,
        "failed": counts.get("failed", 0),
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    events.write(summary)

    if args.final_log_bucket and args.final_log_object:
        upload_small_file(token_provider, args.events_log, args.final_log_bucket, args.final_log_object)
    if args.final_log_bucket and args.summary_object:
        summary_path = args.events_log.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        upload_small_file(token_provider, summary_path, args.final_log_bucket, args.summary_object)

    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
