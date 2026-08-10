# Nodex backend

New FastAPI backend for the project-centric Nodex product. This application is
independent of `backend_old`; legacy modules are not imported.

## Development

```powershell
Copy-Item .env.example .env
uv sync --group dev
uv run uvicorn app.main:app --reload --port 51980
```

The API root is `/api/v1`; health endpoints are `/health/live` and
`/health/ready`. Architecture and migration decisions live in `../docs`.

Agent artifacts and chat images are stored in a private MinIO bucket. Configure
the backend with:

```dotenv
MINIO_ENDPOINT=minio.example.com
MINIO_SECURE=true
MINIO_ACCESS_KEY=your-access-key
MINIO_SECRET_KEY=your-secret-key
MINIO_BUCKET=enerai
MINIO_PRESIGNED_URL_MINUTES=60
```

`MINIO_BUCKET` is optional and defaults to `enerai`. Presigned image URLs are
valid for 60 minutes by default. The service creates the bucket on the first
upload when it does not already exist.

## Modules

- Identity: JWT login plus revocable, hashed `ndx_` API keys.
- Projects: metadata, external operational-data adapter, RDF and point scheme.
- Studio: revisioned graph snapshots and a graph-native Agent surface.
- Agents: provider-neutral run loop, MongoDB entries/lanes/operations/events,
  typed references, MinIO attachments/artifacts, SSE, cancellation
  and restart reconciliation.
- Inspections: persisted policies, scheduled/manual runs and findings.
- Optimizer: GridFS datasets, row-addressable validation and trained models.

Provider adapters intentionally cover only OpenAI, OpenRouter and Bailian. Each
can use Responses or Chat Completions through the normalized runtime.
