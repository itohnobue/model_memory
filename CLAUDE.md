# Memory System

Two-tier memory: **Long-term Knowledge** (`knowledge.md`) and **Session Memory** (`session.md`).

## Long-term Knowledge

Permanent memory across sessions.

### Before Tasks (MANDATORY)

```bash
./.claude/tools/memory.sh context "<keywords>"
```

### Save Discoveries

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

**Skip**: Trivial info, grep-able content, duplicates.

### Mandatory Save Checkpoints

BEFORE completing any task: save relevant findings and include "**Memories saved:** [list]" in response.

### Other Commands

```bash
./.claude/tools/memory.sh search "<query>"
./.claude/tools/memory.sh list [--category CAT]
./.claude/tools/memory.sh delete <id>
```

---

## Session Memory

Temporary work state that survives context compaction. Clear when work completes.

### Categories

| Category | Purpose |
|----------|---------|
| `plan` | Implementation plans, task breakdowns |
| `todo` | Tasks with status: `pending` / `in_progress` / `completed` / `blocked` |
| `progress` | Log of completed work |
| `note` | General session info |
| `blocker` | Issues blocking progress |

### Commands

```bash
# Add
./.claude/tools/memory.sh session add plan "1. Add auth 2. Add tests"
./.claude/tools/memory.sh session add todo "Implement JWT" --status pending

# View
./.claude/tools/memory.sh session show
./.claude/tools/memory.sh session list [--category todo] [--status pending]

# Manage
./.claude/tools/memory.sh session update <id> --status completed
./.claude/tools/memory.sh session delete <id>
./.claude/tools/memory.sh session archive <id> [--category gotcha]  # Move to knowledge
./.claude/tools/memory.sh session clear
```

### When to Use Which

**Session**: Current task plans, WIP todos, progress logs, blockers.
**Long-term**: Architecture, patterns, gotchas, configs, decisions with lasting impact.

### Save Often (CRITICAL)

Session memory survives context compaction. Save state frequently so work can resume from any point:

1. **Start of task**: Save plan with all steps
2. **Before each step**: Mark todo as `in_progress`
3. **After each step**: Log progress, mark todo `completed`
4. **On any blocker**: Save blocker immediately
5. **Before responding**: Update all statuses

If context is compacted or session interrupted, run `session show` to restore full state.
