# Model Memory

Two-tier memory system for Claude Code: **long-term knowledge** that persists across sessions, and **session memory** for tracking work-in-progress.

## Quick Start

1. **Copy files**: Put `.claude/` folder into your Claude Code working directory
2. **Add instructions**: Copy `CLAUDE.md` contents into your project's instruction file
3. **Test it**: Claude will automatically save discoveries and retrieve them in future sessions

The wrapper scripts auto-install **uv**, which handles Python and dependencies.

## Why You Need This

Claude Code forgets everything between sessions. This tool gives it persistent memory.

- **Long-term knowledge** (`knowledge.md`): Architecture, gotchas, patterns, configs
- **Session memory** (`session.md`): Plans, todos, progress logs for current work

**Benefits**: No more rediscovering the same patterns. Faster debugging. Lower token usage.

## Features

- **Two-tier storage**: Permanent knowledge + temporary session state
- **Survives compaction**: Session memory persists through context limits
- **Human-readable**: Plain Markdown files you can edit
- **No database**: Pure file-based, keyword search
- **Zero setup**: Auto-installs dependencies via uv
- **Cross-platform**: macOS, Linux, Windows

## Usage

### Long-term Knowledge

```bash
./.claude/tools/memory.sh add gotcha "Redis needs explicit close()" --tags redis
./.claude/tools/memory.sh context "authentication redis"
./.claude/tools/memory.sh search "API"
./.claude/tools/memory.sh list --category gotcha
./.claude/tools/memory.sh delete <id>
```

### Session Memory

```bash
./.claude/tools/memory.sh session add plan "1. Add auth 2. Add tests"
./.claude/tools/memory.sh session add todo "Implement JWT" --status pending
./.claude/tools/memory.sh session show
./.claude/tools/memory.sh session update <id> --status completed
./.claude/tools/memory.sh session archive <id>   # Move to knowledge
./.claude/tools/memory.sh session clear
```

## Categories

**Knowledge**: `architecture`, `discovery`, `pattern`, `gotcha`, `config`, `entity`, `decision`

**Session**: `plan`, `todo`, `progress`, `note`, `blocker`

## Requirements

- **uv**: Auto-installed by wrapper scripts
- **Python 3.11+**: Auto-installed by uv if needed

## License

MIT
