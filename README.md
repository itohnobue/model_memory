# Model Memory

Persistent memory for Claude Code: **knowledge** (permanent) + **session** (temporary).

## Why You Need This

Claude Code forgets everything between sessions and after context compaction. This tool gives it persistent memory.

- **Long-term knowledge** (`knowledge.md`): Architecture, gotchas, patterns, configs
- **Session memory** (`session.md`): Plans, todos, progress — survives compaction

**Benefits**: No rediscovering the same patterns. Faster debugging. Lower token usage. Work resumes seamlessly after interruptions.

## Quick Start

1. Copy `.claude/` folder to your project
2. Add `CLAUDE.md` contents to your project instructions
3. Done — Claude will automatically save discoveries and track work

## Features

- **Two-tier storage**: Permanent knowledge + temporary session state
- **13 commands**: 6 for knowledge, 7 for session
- **Word boundary search**: "log" won't match "catalog"
- **Field weighting**: Category matches rank higher than content
- **Human-readable**: Plain Markdown files
- **No database**: Pure file-based
- **Zero setup**: Auto-installs Python via uv
- **Cross-platform**: macOS, Linux, Windows

## Categories

**Knowledge (10):** `architecture`, `discovery`, `pattern`, `gotcha`, `config`, `entity`, `decision`, `todo`, `reference`, `context`

**Session (7):** `plan`, `todo`, `progress`, `note`, `context`, `decision`, `blocker`

**Statuses (4):** `pending`, `in_progress`, `completed`, `blocked`

## License

MIT
