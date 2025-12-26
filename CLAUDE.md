# Memory System Instructions

This project has a persistent memory system. Follow these rules.

## MANDATORY: Before Every Task

**Step 1**: Retrieve context before starting work:
```bash
./.claude/tools/memory.sh context "<task keywords>"
```

**Step 2**: Review returned memories to avoid rediscovering known information.

**Step 3**: Skip retrieval ONLY if the task is trivial (< 2 minutes, no domain knowledge needed).

---

## When to Save Memories

Save **immediately** when you discover:

| Trigger | Category | Example |
|---------|----------|---------|
| How components connect | `architecture` | "Auth service calls user-db via gRPC:50051" |
| Non-obvious behavior | `gotcha` | "Redis pool exhausts without explicit close()" |
| Code conventions | `pattern` | "Handlers: validate → process → {data,error,meta}" |
| Environment details | `config` | "Prod: PostgreSQL 15 + pgbouncer, Dev: SQLite" |
| Important code entities | `entity` | "UserService.authenticate() handles OAuth/SAML/password" |
| Design rationale | `decision` | "Redis over Memcached: need pub/sub for realtime" |
| Investigation results | `discovery` | "Memory leak from unclosed file handles in export" |

**Do NOT save**: Trivial info, temp debug notes, easily grep-able content, duplicates.

---

## Commands

```bash
# Add memory
./.claude/tools/memory.sh add <category> "<content>" [--tags a,b,c]

# Search (ranked by relevance + recency)
./.claude/tools/memory.sh search "<query>" [--category X] [--limit N]

# Get context block for a topic
./.claude/tools/memory.sh context "<topic>"

# List memories
./.claude/tools/memory.sh list [--category X] [--limit N]

# Other
./.claude/tools/memory.sh stats              # Show counts
./.claude/tools/memory.sh delete <id>        # Remove memory
./.claude/tools/memory.sh maintain           # Health check
./.claude/tools/memory.sh rebuild            # Force reindex
```

**Categories**: `architecture`, `discovery`, `pattern`, `gotcha`, `config`, `entity`, `decision`, `todo`, `reference`, `context`

---

## Memory Validation

When memories may be stale (after refactoring, major changes, or on request):

1. Load the memory-keeper agent: `.claude/agents/memory-keeper.md`
2. Request: "Validate all memories"
3. Agent will verify each memory against codebase and delete/update as needed

---

## Storage

- **Markdown files**: `knowledge/*.md` (human-readable, git-tracked)
- **SQLite FTS5**: `memory.db` (auto-synced, fast search)
- **Auto-sync**: Database rebuilds from markdown on every query
