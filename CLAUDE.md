# Memory System

Persistent memory across sessions.

## Before Tasks

Retrieve context (skip for trivial tasks):
```bash
./.claude/tools/memory.sh context "<keywords>"
```

## Save Memories

Save immediately when discovering:

| Category | Example |
|----------|---------|
| `architecture` | "Auth service calls user-db via gRPC:50051" |
| `gotcha` | "Redis pool exhausts without explicit close()" |
| `pattern` | "Handlers: validate → process → {data,error,meta}" |
| `config` | "Prod: PostgreSQL 15, Dev: SQLite" |
| `entity` | "UserService.authenticate() handles OAuth/SAML" |
| `decision` | "Redis over Memcached: need pub/sub" |
| `discovery` | "Memory leak from unclosed file handles" |

```bash
./.claude/tools/memory.sh add <category> "<content>" [--tags a,b,c]
```

**Skip**: Trivial info, temp notes, grep-able content, duplicates.

## Other Commands

```bash
./.claude/tools/memory.sh search "<query>"    # Find memories
./.claude/tools/memory.sh list                # List all
./.claude/tools/memory.sh delete <id>         # Remove
./.claude/tools/memory.sh stats               # Counts
```

When memories become stale, verify against codebase and delete/update as needed.
