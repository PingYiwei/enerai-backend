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

## Modules

- Identity: JWT login plus revocable, hashed `ndx_` API keys.
- Projects: metadata, external operational-data adapter, RDF and point scheme.
- Studio: revisioned graph snapshots and a graph-native Agent surface.
- Agents: provider-neutral run loop, MongoDB entries/lanes/operations/events,
  typed references, multimodal attachments, GridFS artifacts, SSE, cancellation
  and restart reconciliation.
- Inspections: persisted policies, scheduled/manual runs and findings.
- Optimizer: GridFS datasets, row-addressable validation and trained models.

Provider adapters intentionally cover only OpenAI, OpenRouter and Bailian. Each
can use Responses or Chat Completions through the normalized runtime.
