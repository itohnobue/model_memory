# Memory System

Two-tier: **Knowledge** (`knowledge.md`) permanent, **Session** (`session.md`) temporary.

## Session Isolation

Multiple CLI instances and agents can work in parallel without conflicts. Each session is isolated by name.

**Session resolution priority:**
1. `--session` / `-S` flag (explicit)
2. `MEMORY_SESSION` environment variable
3. `.claude/current_session` pointer file
4. `"default"` fallback

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

### Session Isolation

Switch to a named session for parallel work:

```bash
# Switch session (saves to pointer file for recovery)
./.claude/tools/memory.sh session use feature-auth

# Check current session
./.claude/tools/memory.sh session current

# List all sessions
./.claude/tools/memory.sh session sessions
```

**For agents:** Use environment variable (doesn't update pointer file):

```bash
export MEMORY_SESSION=agent-explore-$$
```

Or use `-S` flag for one-off commands:

```bash
./.claude/tools/memory.sh session -S agent-task add todo "..."
```

### Session Commands

```bash
# Add entries (to current session)
./.claude/tools/memory.sh session add todo "Task description" --status pending
./.claude/tools/memory.sh session add plan "Step 1... Step 2..."
./.claude/tools/memory.sh session add blocker "Waiting for X"
./.claude/tools/memory.sh session add progress "Completed auth module"

# View current session
./.claude/tools/memory.sh session show
./.claude/tools/memory.sh session list [--status pending]

# View all sessions
./.claude/tools/memory.sh session show-all
./.claude/tools/memory.sh session list-all

# Update
./.claude/tools/memory.sh session update <id> --status completed

# Cleanup
./.claude/tools/memory.sh session delete <id>
./.claude/tools/memory.sh session clear                      # Clear current session only
./.claude/tools/memory.sh session clear --all                # Clear ALL sessions
./.claude/tools/memory.sh session archive <id> [--category CAT]  # Move to knowledge
```

### Recovery After Context Compaction

The tool automatically remembers which session you were using via the pointer file:

```bash
./.claude/tools/memory.sh session show
```

This reads from `.claude/current_session` and shows the correct session state.

