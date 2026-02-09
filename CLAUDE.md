# Memory System

Two-tier: **Knowledge** (`knowledge.md`) permanent, **Session** (`session.md`) temporary.

## Quick Decision

| Question | Use |
|----------|-----|
| Will this help in future sessions? | **Knowledge** |
| Is this about current task only? | **Session** |
| Discovered a gotcha/pattern/config? | **Knowledge** |
| Tracking todos/progress/blockers? | **Session** |

## Before Any Task

```bash
./.claude/tools/memory.sh context "<keywords>"
```

Extract keywords from: entities, technologies, service names, error types.

**Skip only for:** single-line trivial fixes.

## Save to Knowledge

```bash
memory.sh add <category> "<content>" [--tags a,b,c]
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

**Tags:** Cross-cutting concerns (e.g., `--tags redis,production,auth`).

**Skip:** Trivial info, easily grep-able content, duplicates.

**After tasks:** State "**Memories saved:** [list]" or "**Memories saved:** None"

**Stale memories:** Delete with `memory.sh delete <id>`.

**Other:** `search "<query>"`, `list [--category CAT]`, `delete <id>`, `stats`

---

## Session Memory

Tracks current task. **Persists until explicitly cleared.**

**Categories:** `plan`, `todo`, `progress`, `note`, `context`, `decision`, `blocker`

**Statuses:** `pending` → `in_progress` → `completed` | `blocked`

### Commands

```bash
# Add
memory.sh session add todo "Task" --status pending
memory.sh session add plan "Step 1... Step 2..."

# View
memory.sh session show
memory.sh session list [--status pending]

# Update/Delete
memory.sh session update <id> --status completed
memory.sh session delete <id>

# Cleanup
memory.sh session clear           # Current session only
memory.sh session clear --all     # ALL sessions
memory.sh session archive <id>    # Move to knowledge
```

### State Checkpoints (Compaction Survival)

Context compaction can erase working state mid-task. Save a checkpoint after **every significant step** so work can continue seamlessly.

**When:** After each step that produces results, makes decisions, or changes direction.

**What to save** (single `context` entry, replace previous checkpoint):

```bash
memory.sh session add context "CHECKPOINT: [task summary] | DONE: [completed steps] | CURRENT: [what you're doing now] | NEXT: [remaining steps] | FILES: [key files involved] | DECISIONS: [important choices made] | BLOCKERS: [if any]"
```

**After compaction** (you'll notice missing conversation history): Run `memory.sh session show` immediately and use the latest checkpoint to restore your working state before continuing.

**Rules:**
- One active checkpoint at a time — delete the previous one before adding a new one
- Keep each checkpoint under 500 chars — be terse, use abbreviations
- Always include DONE and NEXT — these are the minimum needed to continue
- Do NOT skip checkpoints to save time — the cost of losing state is much higher

### Multi-Session (Parallel Work)

Multiple CLI instances/agents can work without conflicts.

**Resolution priority:**
1. `-S` / `--session` flag
2. `MEMORY_SESSION` environment variable
3. `.claude/current_session` pointer file
4. `"default"`

```bash
# Interactive: switch session (saved to pointer file)
memory.sh session use feature-auth

# Agents: use env var (no pointer update)
export MEMORY_SESSION=agent-$$

# One-off: use -S flag
memory.sh -S other session add todo "..."

# View all
memory.sh session sessions      # List sessions
memory.sh session show-all      # Show all content
```
