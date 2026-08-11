# Agents module

The module is organized by responsibility while keeping its public HTTP and orchestration entry
points at the package root.

```text
agents/
├── router.py       # FastAPI endpoints and SSE transport
├── schemas.py      # HTTP and persistence-facing Pydantic contracts
├── service.py      # Run coordination and dependency composition
├── prompts/        # Insight/Studio instructions and per-run clock context
├── providers/      # Provider protocol, registry, and API adapters
├── runtime/        # Model loop, context projection, titles, and internal event types
├── storage/        # Mongo/GridFS/MinIO sessions, attachments, and artifacts
└── tools/          # Tool execution contract plus project and Studio tool collections
```

Dependency direction is `router -> service -> runtime/tools/storage -> providers`. Prompt rendering
happens in `service.py` immediately before each model run so its time context is never frozen at
application startup. Provider adapters and tools depend only on the runtime contracts, not on the
HTTP layer.
