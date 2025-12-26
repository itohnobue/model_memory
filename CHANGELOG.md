# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
