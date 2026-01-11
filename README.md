# News RAG + MCP — AWS Free‑Tier Starter

This repository is a **walking skeleton** for your dissertation project, optimized for the AWS Free Tier. It deploys:
- S3 data lake (raw/clean/curated),
- EventBridge rule + Lambda **ingest_rss** (scheduled),
- Glue database + crawler for raw bucket,
- DynamoDB **alerts** table,
- Step Functions state machine skeleton for the MCP-style decision layer,
- API Gateway stub (HTTP API) with a "health" Lambda,
- (Optional) stubs for **classify** and **RAG** Lambdas you can wire later.

> You can deploy **only the ingestion + storage** to keep costs at $0.

---

## Prereqs

- Node 18+ and AWS CDK v2 installed locally (`npm i -g aws-cdk`).
- Python 3.10+ for Lambda handlers.
- An AWS account with programmatic credentials configured (`aws configure`).
- Recommended region: `ap-southeast-2` (Sydney) or your closest region.

---

## Deploy (first time)

```bash
cd infra/cdk
npm install
# One-time bootstrap for your account/region:
npx cdk bootstrap aws://<ACCOUNT_ID>/<REGION>

# Deploy core stacks (S3 + Glue + EventBridge + ingest lambda + DynamoDB + API stub):
npx cdk deploy --all
```

> After deploy, see CloudWatch Logs for function `IngestRSSFunction` to confirm events flowing.

---

## Folders

- `infra/cdk/` — CDK app (TypeScript) with stacks:
  - `DataLakeStack` — S3 buckets, Glue DB + Crawler
  - `IngestionStack` — EventBridge schedule, `ingest_rss` Lambda, roles
  - `DecisionStack` — Step Functions state machine + DynamoDB `alerts`
  - `ApiStack` — API Gateway HTTP API + health Lambda
- `lambdas/` — Python Lambda handlers
  - `ingest_rss/handler.py` — fetches RSS feed(s), normalizes JSON, writes to S3 (`raw/ingest_date=...`)
  - `clean_transform/handler.py` — triggered by raw S3 writes; flattens batches into newline JSON in the clean bucket
  - `classify_stub/handler.py` — deterministic rule-based placeholder (replace with SageMaker or HF later)
  - `rag_stub/handler.py` — template for retrieval + generation
- `state_machines/decision/sfn_asl.json` — Step Functions ASL template (adjust routing rules/thresholds)
- `data/contracts/` — JSON schemas for raw/curated/summary records
- `api/` — placeholder for FastAPI ECS service (future work)
- `dashboard/` — placeholder (Streamlit/React, deploy elsewhere to stay free)

---

## Tear down

```bash
# Destroys stacks and buckets (empty buckets first if non-empty!)
npx cdk destroy --all
```

> Be careful: destroying stacks is irreversible.

---

## Next steps

1) Wire `classify_stub` and `rag_stub` into a Step Functions pipeline.
2) Replace stubs with Bedrock/SageMaker when ready.
3) Add metrics export (PRF1, ROUGE) to S3 `metrics/` and a dashboard page.
