"""
Lambda that normalizes raw RSS batches into clean, record-per-line JSON.
Triggered by S3 PUT events on the raw bucket.
"""
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List
from urllib.parse import unquote_plus

import boto3

RAW_BUCKET = os.environ["RAW_BUCKET"]
CLEAN_BUCKET = os.environ["CLEAN_BUCKET"]
RAW_PREFIX = os.environ.get("RAW_PREFIX", "raw/")
CLEAN_PREFIX = os.environ.get("CLEAN_PREFIX", "clean/")

s3 = boto3.client("s3")


def _extract_ingest_date(key: str) -> str:
    for part in key.split("/"):
        if part.startswith("ingest_date="):
            return part.split("=", 1)[1]
    return datetime.utcnow().strftime("%Y-%m-%d")


def _normalize_record(record: Dict[str, Any], ingest_date: str) -> Dict[str, Any]:
    """Ensure required keys exist, trim whitespace, and carry ingest date forward."""
    return {
        "id": record.get("id") or str(uuid.uuid4()),
        "title": (record.get("title") or "").strip(),
        "url": (record.get("url") or "").strip(),
        "published": (record.get("published") or "").strip(),
        "summary": (record.get("summary") or "").strip(),
        "ingested_at": record.get("ingested_at") or datetime.utcnow().isoformat() + "Z",
        "source": record.get("source") or "rss",
        "ingest_date": ingest_date,
    }


def _iter_records(payload: bytes) -> Iterable[Dict[str, Any]]:
    parsed = json.loads(payload.decode("utf-8"))
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError("Unexpected payload type")


def handler(event, _context):
    outputs: List[Dict[str, Any]] = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])

        if bucket != RAW_BUCKET or not key.startswith(RAW_PREFIX):
            continue

        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        ingest_date = _extract_ingest_date(key)
        normalized = [_normalize_record(r, ingest_date) for r in _iter_records(body)]

        clean_key = key.replace(RAW_PREFIX, CLEAN_PREFIX, 1)
        if clean_key.endswith(".json"):
            clean_key = clean_key[:-5] + ".jsonl"

        payload = "\n".join(json.dumps(rec, separators=(",", ":")) for rec in normalized)
        s3.put_object(
            Bucket=CLEAN_BUCKET,
            Key=clean_key,
            Body=payload.encode("utf-8"),
            ContentType="application/json",
        )
        outputs.append({"source": key, "target": clean_key, "records": len(normalized)})

    return {"ok": True, "files": outputs}
