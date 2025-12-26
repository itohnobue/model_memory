# Model Memory

Persistent long-term memory for Claude Code. Stores project knowledge across sessions using Markdown files + SQLite FTS5 search.

## Features

- **Hybrid storage**: Human-readable Markdown + fast SQLite FTS5 search
- **Smart ranking**: BM25 relevance + recency boost
- **Auto-sync**: Database rebuilds from Markdown on every query
- **Cross-platform**: macOS, Linux, Windows

## Installation

```bash
# Clone
git clone ssh://git@git.aoizora.ru:2222/nobu/model_memory.git

# Make executable (Unix)
chmod +x .claude/tools/memory.sh
```

Requirements: Python 3.11+ (uv auto-installs on first run)

## Quick Start

```bash
# Add memories
./memory.sh add discovery "API uses JWT with RS256"
./memory.sh add gotcha "Redis needs explicit close()" --tags redis,pool

# Search (ranked by relevance + recency)
./memory.sh search "authentication"

# Get context block for Claude
./memory.sh context "database"

# List and manage
./memory.sh list --category gotcha
./memory.sh stats
./memory.sh delete <id>
./memory.sh maintain --max-age 180
```

## Usage with Claude Code

Copy `.claude/` folder and `CLAUDE.md` to your project. Claude follows `CLAUDE.md` instructions automatically.

## Commands

| Command | Description |
|---------|-------------|
| `add <category> <content>` | Add memory (use `--tags a,b,c`) |
| `search <query>` | Search with BM25 + recency ranking |
| `context <topic>` | Get formatted context block |
| `list` | List memories (use `--category`, `--limit`) |
| `stats` | Show statistics |
| `delete <id>` | Delete a memory |
| `maintain` | Health check (use `--max-age N --execute`) |
| `rebuild` | Force reindex from Markdown files |

## Categories

| Category | Use For |
|----------|---------|
| `architecture` | System design, component relationships |
| `discovery` | Exploration findings |
| `pattern` | Code conventions, idioms |
| `gotcha` | Bugs, workarounds, pitfalls |
| `config` | Environment, settings |
| `entity` | Key classes, functions, APIs |
| `decision` | Design rationale |
| `todo` | Pending items |
| `reference` | External links |
| `context` | Project background |

## Options

| Option | Description |
|--------|-------------|
| `-t, --tags` | Comma-separated tags |
| `-l, --limit` | Limit results (default: 10) |
| `-c, --category` | Filter by category |
| `-o, --output` | Format: `text` or `json` |
| `-q, --quiet` | Suppress output |
| `--max-age` | Days threshold (maintain) |
| `--execute` | Actually delete (maintain) |

## Shell Completions

```bash
# Bash
source .claude/tools/completions/memory.bash

# Zsh
source .claude/tools/completions/memory.zsh

# Fish
source .claude/tools/completions/memory.fish

# PowerShell
. .claude/tools/completions/memory.ps1
```

## Project Structure

```
.claude/
├── agents/memory-keeper.md   # Validation/cleanup agent
└── tools/
    ├── memory.py             # Core tool
    ├── memory.sh             # Unix wrapper
    ├── memory.bat            # Windows wrapper
    └── completions/          # Shell completions
knowledge/                    # Markdown memory files
memory.db                     # SQLite FTS5 database
CLAUDE.md                     # AI instructions
```

## Troubleshooting

**uv not found**: Auto-installs on first run. Manual: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**No search results**: Run `./memory.sh rebuild` to reindex.

**Permission denied**: Run `chmod +x .claude/tools/memory.sh`

## License

MIT
