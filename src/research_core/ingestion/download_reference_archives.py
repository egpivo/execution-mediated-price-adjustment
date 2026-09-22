#!/usr/bin/env python3
"""Full-window immutable download for the two frozen reference sources:
Bybit spot ETHUSDC and OKX spot ETH-USDC native order-book archives.

Endpoint families are exactly those verified in
../../jfqa_public_reference_audit/code/audit_public_availability.py.
No basis leg (USDCUSDT) is downloaded: the frozen primary construction
(sprint spec section 8) uses only these two direct ETH/USDC quotes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time as time_module
import urllib.error
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

# Mechanical extension of download_full_reference.py: START/END are now
# CLI-provided instead of hardcoded so the same unchanged download/URL/retry
# logic below can run against the confirmatory-window dates. No other line
# in this file differs from the discovery-sample original.
START: date
END: date
MAX_RETRIES = 6


def dates_between(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def request(url: str, *, data: bytes | None = None, method: str | None = None):
    headers = {"User-Agent": "jfqa-hf-response-full/1.0"}
    if data is not None:
        headers["Content-Type"] = "application/json"
        headers["Origin"] = "https://www.okx.com"
        headers["Referer"] = "https://www.okx.com/historical-data"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    for attempt in range(MAX_RETRIES):
        try:
            return urllib.request.urlopen(req, timeout=120)
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            transient = isinstance(exc, urllib.error.URLError) or getattr(exc, "code", None) in (429, 500, 502, 503, 504)
            if transient and attempt < MAX_RETRIES - 1:
                time_module.sleep(2.0 * (attempt + 1))
                continue
            raise


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with request(url) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.rename(dest)
    return {"bytes": dest.stat().st_size, "sha256": sha256_file(dest)}


def bybit_urls() -> list[dict]:
    root = "https://quote-saver.bycsi.com/orderbook/spot/ETHUSDC/"
    html = request(root).read().decode("utf-8")
    names = set(re.findall(r'href="([^"]+\.zip)"', html))
    rows = []
    for day in dates_between(START, END):
        name = f"{day.isoformat()}_ETHUSDC_ob200.data.zip"
        if name not in names:
            raise RuntimeError(f"Bybit ETHUSDC missing from listing for {day}: {name}")
        rows.append({"venue": "Bybit", "symbol": "ETHUSDC", "date": day.isoformat(),
                     "url": root + name, "filename": name})
    return rows


def utc_ms(day: date, end_of_day: bool = False) -> int:
    dt = datetime.combine(day, time.max if end_of_day else time.min, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def okx_urls() -> list[dict]:
    endpoint = "https://www.okx.com/priapi/v5/broker/public/trade-data/download-link"
    found: dict[str, dict] = {}
    cursor = START
    while cursor <= END:
        chunk_end = min(cursor + timedelta(days=6), END)
        payload = {
            "module": "4",
            "instType": "SPOT",
            "instQueryParam": {"instIdList": ["ETH-USDC"]},
            "dateQuery": {"dateAggrType": "daily", "begin": str(utc_ms(cursor)),
                          "end": str(utc_ms(chunk_end, end_of_day=True))},
        }
        with request(endpoint, data=json.dumps(payload).encode()) as response:
            body = json.load(response)
        if body.get("code") != "0":
            raise RuntimeError(body)
        for detail in body["data"]["details"]:
            for item in detail["groupDetails"]:
                d = datetime.fromtimestamp(int(item["dateTs"]) / 1000, tz=timezone.utc).date().isoformat()
                found[d] = item
        cursor = chunk_end + timedelta(days=1)

    rows = []
    for day in dates_between(START, END):
        item = found.get(day.isoformat())
        if item is None:
            raise RuntimeError(f"OKX ETH-USDC missing from listing for {day}")
        rows.append({"venue": "OKX", "symbol": "ETH-USDC", "date": day.isoformat(),
                     "url": item["url"], "filename": item["filename"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()

    global START, END
    START, END = args.start, args.end

    plan = bybit_urls() + okx_urls()
    manifest = []
    for i, row in enumerate(plan, 1):
        subdir = "bybit" if row["venue"] == "Bybit" else "okx"
        dest = args.raw_root / subdir / row["filename"]
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if dest.exists() and dest.stat().st_size > 0:
            info = {"bytes": dest.stat().st_size, "sha256": sha256_file(dest)}
            status = "already_present"
        else:
            print(f"[{i}/{len(plan)}] downloading {row['venue']} {row['date']} ...", flush=True)
            info = download(row["url"], dest)
            status = "downloaded"
        manifest.append({**row, "path": str(dest), "status": status,
                          "retrieved_at_utc": retrieved_at, **info})

    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")
    with args.manifest_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"done: {len(manifest)} objects")


if __name__ == "__main__":
    main()
