# Model Memory

Persistent memory for Claude Code: **knowledge** (permanent) + **session** (temporary).

## Why You Need This

Claude Code forgets everything between sessions and after context compaction. This tool gives it persistent memory.

- **Long-term knowledge** (`knowledge.md`): Architecture, gotchas, patterns, configs
- **Session memory** (`session.md`): Plans, todos, progress — survives compaction
- **Session isolation**: Multiple CLI/agents can work in parallel without conflicts

**Benefits**: No rediscovering the same patterns. Faster debugging. Lower token usage. Work resumes seamlessly after interruptions.

## Quick Start

1. Copy `.claude/` folder to your project
2. Add `CLAUDE.md` contents to your project instructions
3. Done — Claude will automatically save discoveries and track work

## Features

- **Two-tier storage**: Permanent knowledge + temporary session state
- **Session isolation**: Multiple sessions can coexist (v5.1+)
- **18 commands**: 6 for knowledge, 12 for session
- **Word boundary search**: "log" won't match "catalog"
- **Field weighting**: Category matches rank higher than content
- **Auto-recovery**: Session pointer survives context compaction
- **Human-readable**: Plain Markdown files
- **No database**: Pure file-based
- **Zero setup**: Auto-installs Python via uv
- **Cross-platform**: macOS, Linux, Windows

## Session Isolation (v5.1+)

Multiple CLI instances and agents can work without conflicts:

```bash
# Switch to a named session
memory.sh session use feature-auth

# Or use environment variable (for agents)
export MEMORY_SESSION=agent-explore

# Or use -S flag for one-off commands
memory.sh session -S other-task add todo "..."
```

Session resolution priority:
1. `--session` / `-S` flag
2. `MEMORY_SESSION` environment variable
3. `.claude/current_session` pointer file
4. `"default"` fallback

## Categories

**Knowledge (10):** `architecture`, `discovery`, `pattern`, `gotcha`, `config`, `entity`, `decision`, `todo`, `reference`, `context`

**Session (7):** `plan`, `todo`, `progress`, `note`, `context`, `decision`, `blocker`

**Statuses (4):** `pending`, `in_progress`, `completed`, `blocked`

## Commands

**Knowledge:**
- `add <category> <content>` - Add a memory
- `search <query>` - Search memories
- `context <topic>` - Get context for topic
- `list` - List memories
- `delete <id>` - Delete a memory
- `stats` - Show statistics

**Session:**
- `session add <cat> <content>` - Add entry
- `session list` - List current session entries
- `session show` - Show current session state
- `session update <id> --status <s>` - Update status
- `session delete <id>` - Delete entry
- `session clear` - Clear current session
- `session clear --all` - Clear ALL sessions
- `session archive <id>` - Move to knowledge
- `session use <name>` - Switch session
- `session current` - Show session info
- `session sessions` - List all sessions
- `session list-all` - List all entries
- `session show-all` - Show all sessions

## License

MIT
