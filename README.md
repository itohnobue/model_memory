# Model Memory

Persistent long-term memory for Claude Code that remembers project knowledge across sessions.

## Quick start (highly recommended)

1. **Copy files to your project**: Put `.claude/` folder (with `tools/` inside) into your Claude Code working directory

2. **Add instructions from CLAUDE.md to your model instructions file**: Copy the contents of `CLAUDE.md` into your project's instruction file (create one if it doesn't exist)

3. **Test it**: Ask Claude Code to do any exploration task. It will automatically save important discoveries and retrieve them in future sessions.

The wrapper scripts will automatically install **uv** (if needed), which handles Python and all dependencies.

## What it does and why you may need it (read this first)

Claude Code has a fundamental limitation: **it forgets everything between sessions**. Every time you start a new conversation, Claude rediscovers the same things about your codebase - wasting tokens and your time.

This tool solves that problem by giving Claude a persistent memory that survives across sessions.

**How it works**: When Claude explores your codebase, it saves important discoveries (architecture decisions, gotchas, patterns, configurations) into simple Markdown files. Next session, Claude retrieves relevant memories before starting work - so it already knows what it learned before.

**Real-world benefits I've experienced**:
- Claude stops rediscovering the same architectural patterns over and over
- Debugging is faster because Claude remembers past investigations
- Complex codebases become manageable as knowledge accumulates
- Token usage drops significantly on repeated tasks

The memories are stored as human-readable Markdown files (you can read and edit them yourself) with SQLite FTS5 for fast search. Everything auto-syncs - no manual database management needed.

---

## Features

- **Persistent Knowledge**: Memories survive across Claude Code sessions
- **Smart Search**: BM25 relevance ranking with recency boost
- **Human-Readable**: All memories stored as Markdown files in `knowledge/` folder
- **Auto-Sync**: Database rebuilds from Markdown on every query
- **Zero Setup**: Uses uv with inline dependencies - no manual venv or pip needed
- **Cross-Platform**: Works on macOS, Linux, and Windows

## Usage

```bash
# Add memories
./.claude/tools/memory.sh add discovery "API uses JWT with RS256"
./.claude/tools/memory.sh add gotcha "Redis needs explicit close()" --tags redis,pool

# Search memories
./.claude/tools/memory.sh search "authentication"

# Get context block for a topic
./.claude/tools/memory.sh context "database"

# List and manage
./.claude/tools/memory.sh list --category gotcha
./.claude/tools/memory.sh stats
./.claude/tools/memory.sh delete <id>
```

## Memory Categories

| Category | Use For |
|----------|---------|
| `architecture` | System design, component relationships |
| `discovery` | Exploration findings |
| `pattern` | Code conventions, idioms |
| `gotcha` | Bugs, workarounds, pitfalls |
| `config` | Environment, settings |
| `entity` | Key classes, functions, APIs |
| `decision` | Design rationale |

## Requirements

- **uv**: Installed automatically by wrapper scripts
- **Python 3.11+**: Installed automatically by uv if needed

## Troubleshooting

**uv not found**: Auto-installs on first run. Manual: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**No search results**: Run `./.claude/tools/memory.sh rebuild` to reindex.

**Permission denied**: Run `chmod +x .claude/tools/memory.sh`

## License

MIT
