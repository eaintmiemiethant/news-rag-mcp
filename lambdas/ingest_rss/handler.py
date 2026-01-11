import os, json, time, uuid, datetime as dt
from urllib.request import urlopen
import xml.etree.ElementTree as ET

RAW_BUCKET = os.environ.get('RAW_BUCKET')
FEEDS = [u.strip() for u in os.environ.get('FEEDS', '').split(',') if u.strip()]

def _parse_rss(xml_bytes):
    # Minimal RSS parser (no external libs to keep package small)
    root = ET.fromstring(xml_bytes)
    channel = root.find('channel')
    if channel is None:
        # Some feeds use namespaces; fallback: scan for item tags
        items = root.findall('.//item')
    else:
        items = channel.findall('item')
    parsed = []
    for it in items[:50]:  # cap per-invoke to keep free-tier friendly
        title = (it.findtext('title') or '').strip()
        link = (it.findtext('link') or '').strip()
        pub = (it.findtext('pubDate') or '').strip()
        desc = (it.findtext('description') or '').strip()
        parsed.append({
            "id": str(uuid.uuid4()),
            "title": title,
            "url": link,
            "published": pub,
            "summary": desc,
            "ingested_at": dt.datetime.utcnow().isoformat() + 'Z',
            "source": "rss"
        })
    return parsed

def handler(event, context):
    # Lazily import boto3 to reduce cold start size
    import boto3
    s3 = boto3.client('s3')

    ymd = dt.datetime.utcnow().strftime('%Y-%m-%d')
    prefix = f"raw/ingest_date={ymd}/"

    total = 0
    for feed in FEEDS:
        try:
            with urlopen(feed, timeout=10) as r:
                xml = r.read()
            items = _parse_rss(xml)
            total += len(items)
            key = prefix + f"rss_{int(time.time())}.json"
            s3.put_object(
                Bucket=RAW_BUCKET,
                Key=key,
                Body=json.dumps(items).encode('utf-8'),
                ContentType='application/json'
            )
        except Exception as e:
            # swallow to keep the schedule healthy; you can log to CloudWatch
            print(f"Failed {feed}: {e}")
    return { "ok": True, "written": total }
