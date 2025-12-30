# Memory System

Two-tier: **Knowledge** (`knowledge.md`) permanent, **Session** (`session.md`) temporary.

## Before Any Task

```bash
./.claude/tools/memory.sh context "<keywords>"
```

Extract keywords from: entities mentioned, technologies, service names, error types. Use multiple keywords.

**Skip only for:** single-line trivial fixes.

## Save Discoveries

Save **immediately** when discovering something worth remembering:

```bash
./.claude/tools/memory.sh add <category> "<content>" [--tags a,b,c]
```

| Category | Save When |
|----------|-----------|
| `architecture` | System design, service connections, ports |
| `gotcha` | Bugs, pitfalls, non-obvious behavior |
| `pattern` | Code conventions, recurring structures |
| `config` | Environment settings, credentials locations |
| `entity` | Important classes, functions, APIs |
| `decision` | Why choices were made |
| `discovery` | New findings about codebase |
| `todo` | Long-term tasks to remember |
| `reference` | Useful links, documentation |
| `context` | Background info, project context |

**Tags:** Use for cross-cutting concerns (e.g., `--tags redis,production,auth`).

**Skip:** Trivial info, easily grep-able content, duplicates.

**After tasks:** State "**Memories saved:** [list]" or "**Memories saved:** None"

**Stale memories:** If found outdated info, delete with `memory.sh delete <id>`.

## Other Knowledge Commands

```bash
./.claude/tools/memory.sh search "<query>" [--category CAT]
./.claude/tools/memory.sh list [--category CAT]
./.claude/tools/memory.sh delete <id>
./.claude/tools/memory.sh stats
```

---

## Session Memory

Tracks **current task** work. Survives context compaction.

**Use for:** Plans, in-progress todos, blockers, progress logs.
**Use Knowledge for:** Lasting discoveries that apply beyond this task.

**Categories:** `plan`, `todo`, `progress`, `note`, `context`, `decision`, `blocker`

**Statuses:** `pending` → `in_progress` → `completed` | `blocked`

```bash
# Add entries
./.claude/tools/memory.sh session add todo "Task description" --status pending
./.claude/tools/memory.sh session add plan "Step 1... Step 2..."
./.claude/tools/memory.sh session add blocker "Waiting for X"
./.claude/tools/memory.sh session add progress "Completed auth module"

# View
./.claude/tools/memory.sh session show
./.claude/tools/memory.sh session list [--status pending]

# Update
./.claude/tools/memory.sh session update <id> --status completed

# Cleanup
./.claude/tools/memory.sh session delete <id>
./.claude/tools/memory.sh session clear                      # When task fully complete
./.claude/tools/memory.sh session archive <id> [--category CAT]  # Move to knowledge
```

**Recovery:** After context compaction, run `session show` to restore state.
