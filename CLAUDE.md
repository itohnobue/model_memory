# Memory System

Two-tier: **Knowledge** (`knowledge.md`) permanent, **Session** (`session.md`) temporary.

| Question | Use |
|----------|-----|
| Will this help in future sessions? | **Knowledge** |
| Current task only? | **Session** |
| Discovered a gotcha/pattern/config? | **Knowledge** |
| Tracking todos/progress/blockers? | **Session** |

## Knowledge

```bash
memory.sh add <category> "<content>" [--tags a,b,c]
```

| Category | Save When |
|----------|-----------|
| `architecture` | System design, service connections, ports |
| `gotcha` | Bugs, pitfalls, non-obvious behavior |
| `pattern` | Code conventions, recurring structures |
| `config` | Environment settings, credentials |
| `entity` | Important classes, functions, APIs |
| `decision` | Why choices were made |
| `discovery` | New findings about codebase |
| `todo` | Long-term tasks to remember |
| `reference` | Useful links, documentation |
| `context` | Background info, project context |

**Tags:** Cross-cutting concerns (e.g., `--tags redis,production,auth`). **Skip:** Trivial, easily grep-able, duplicates.

**After tasks:** State "**Memories saved:** [list]" or "**Memories saved:** None"

**Other:** `search "<query>"`, `list [--category CAT]`, `delete <id>`, `stats`

## Session

Tracks current task. Persists until cleared.

**Categories:** `plan`, `todo`, `progress`, `note`, `context`, `decision`, `blocker`. **Statuses:** `pending` → `in_progress` → `completed` | `blocked`.

```bash
memory.sh session add todo "Task" --status pending
memory.sh session show                    # View current
memory.sh session update <id> --status completed
memory.sh session delete <id>
memory.sh session clear                   # Current only
memory.sh session clear --all             # ALL sessions
```

## Checkpoints

Save after every significant step. One active checkpoint (delete previous first). Under 500 chars.

```bash
memory.sh session add context "CHECKPOINT: [task] | DONE: [steps] | CURRENT: [now] | NEXT: [remaining] | FILES: [key files] | DECISIONS: [choices] | BUILD/TEST: [commands]"
```

After compaction: run `memory.sh session show` immediately to restore state.

**Rules:** One checkpoint at a time. Always include DONE and NEXT. Don't skip — losing state costs more.

## Multi-Session

Multiple CLI instances work without conflicts. Resolution: `-S` flag > `MEMORY_SESSION` env > `.claude/current_session` file > `"default"`.

```bash
memory.sh session use feature-auth        # Switch session
memory.sh -S other session add todo "..." # One-off
memory.sh session sessions                # List all
```
