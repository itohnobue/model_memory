# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-12-30

### Changed
- **Breaking**: Switched from multiple category files (`knowledge/*.md`) to single file (`knowledge.md`)
- **Breaking**: Replaced `created_at`/`updated_at` with single `changed_at` timestamp
- Memory entries now include `Category:` and `Changed:` metadata lines
- Simplified codebase by removing multi-file management logic
- Age distribution in `maintain` now based on last change, not creation date
- Recency boost in search now favors recently modified memories

### Removed
- `knowledge/` directory structure (replaced by single `knowledge.md` file)
- `--max-age` and `--execute` options from `maintain` command
- Migration support for old format

## [1.2.0] - 2025-12-28

### Added
- Keyword OR search mode: searches now match ANY keyword by default
- `--mode` CLI argument: choose between `keywords` (default) and `phrase` modes
- `build_or_query()` function for flexible multi-keyword FTS5 queries
- Prefix matching with `*` wildcard for partial keyword matches

### Changed
- Default search mode changed from exact phrase to keyword OR matching
- `context` command now uses keyword mode for better context retrieval
- Updated help text with search mode examples

### Fixed
- Multi-keyword context queries now find relevant memories instead of requiring exact phrases

## [1.1.0] - 2024-12-27

### Added
- `maintain` command for database health checks and cleanup
- Recency-boosted search ranking (newer memories rank higher)
- `--max-age` option to identify old memories
- `--execute` option to delete old memories
- PowerShell completion script
- Database index on `created_at` for faster sorting
- Validation audit workflow in memory-keeper agent

### Changed
- Search results now include `age_days` field
- Help examples updated to show `memory.sh` instead of `memory.py`

### Fixed
- Tags parsing issue with empty strings
- Type hints for `format_output` function

## [1.0.0] - 2024-12-27

### Added
- Initial release
- Hybrid storage: Markdown files + SQLite FTS5
- Auto-sync between files and database
- BM25-ranked full-text search
- 10 categories for organizing knowledge
- Shell completions for bash, zsh, fish
- Cross-platform support (Unix/Windows)
- WAL mode for better concurrency
- Path traversal protection
- FTS5 query escaping
- `--quiet` and `--version` flags
