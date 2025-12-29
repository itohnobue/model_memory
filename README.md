# Model Memory

Two-tier memory system for Claude Code: **long-term knowledge** that persists across sessions, and **session memory** for tracking work-in-progress.

## Quick Start

1. **Copy files**: Put `.claude/` folder into your Claude Code working directory
2. **Add instructions**: Copy `CLAUDE.md` contents into your project's instruction file
3. **Done**: Claude will automatically save discoveries and track work state

The wrapper scripts auto-install **uv**, which handles Python and dependencies.

## Why You Need This

Claude Code forgets everything between sessions and after context compaction. This tool gives it persistent memory.

- **Long-term knowledge** (`knowledge.md`): Architecture, gotchas, patterns, configs
- **Session memory** (`session.md`): Plans, todos, progress — survives compaction

**Benefits**: No rediscovering the same patterns. Faster debugging. Lower token usage. Work resumes seamlessly after interruptions.

## Features

- **Two-tier storage**: Permanent knowledge + temporary session state
- **Survives compaction**: Session memory persists through context limits
- **Human-readable**: Plain Markdown files you can read/edit
- **No database**: Pure file-based with keyword search
- **Zero setup**: Auto-installs dependencies via uv
- **Cross-platform**: macOS, Linux, Windows

## Requirements

- **uv**: Auto-installed by wrapper scripts
- **Python 3.11+**: Auto-installed by uv if needed

## License

MIT
