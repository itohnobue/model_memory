# Memory System

Persistent memory across sessions. **CRITICAL: Always retrieve context before starting non-trivial tasks.**

## Before Tasks (MANDATORY)

ALWAYS retrieve context before starting work (only skip for single-line trivial fixes):
```bash
./.claude/tools/memory.sh context "<keywords>"
```

Use multiple keywords covering: server names, service names, technologies, error types, domain concepts.

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
./.claude/tools/memory.sh search "<query>"    # Find memories (keyword OR search)
./.claude/tools/memory.sh list                # List all
./.claude/tools/memory.sh delete <id>         # Remove
./.claude/tools/memory.sh stats               # Counts
```

When memories become stale, verify against codebase and delete/update as needed.
