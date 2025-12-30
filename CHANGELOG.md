# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.0.0] - 2025-12-30 - "Lean Edition"

### Changed
- **Breaking**: Massive simplification - 804 lines (down from 2,419)
- **Breaking**: 13 commands (down from 26)
- Simplified search: word boundaries + basic field weighting (category=2x, tags=1.5x, content=1x)
- Reduced stop words list from 60+ to 20 most common
- Simplified data structures (removed Related, Important, Blocked-by, Status-history fields)

### Removed
- All v4.0 over-engineered features:
  - TF-IDF scoring (replaced with simple weighted scoring)
  - `similar` command (use search instead)
  - `merge` command (delete one, re-add combined)
  - `consolidate` command
  - `reclassify` command (delete and re-add)
  - `tags` command
  - `edit` command (opens $EDITOR - useless for Claude)
  - `update` command (delete and re-add)
  - `backup`, `validate`, `repair` commands (solving non-problems)
  - `search-all` command
  - `session new` command (use session clear)
  - `session search` command (sessions are small, use show)
  - `maintain` command (useless age distribution)
  - Category aliases and sub-categories
  - Custom categories (back to predefined list)
  - Status transition validation
  - Status history tracking
  - Auto-archive keywords
  - Duplicate detection and --force flag
  - Auto-suggest relations
  - Importance and blocked-by fields

### Kept
- Core: add, search, context, list, delete, stats
- Session: add, list, show, update, delete, clear, archive
- Word boundary matching in search
- Basic stop word filtering
- Tags support (on add command)

### Why This Release
An independent architecture audit found v4.0 to be significantly over-engineered:
- 60% of features were rarely/never used
- Code was 4-5x larger than needed
- Many features solved problems that don't exist

The tool's purpose is simple: help Claude remember things between sessions.
Everything else was feature creep.

## [4.0.0] - 2025-12-30

### Added

#### Search Improvements (1A-1D)
- Word boundary matching to prevent 'log' matching 'catalog'
- Stop word filtering (common words like 'the', 'is', 'a' ignored)
- Field-weighted scoring (category > tags > ID > content)
- TF-IDF relevance ranking for better search results
- `--tag` flag for tag-based filtering

#### Duplicate Detection (2B-2D)
- `similar <content>` command to find similar memories
- Auto-suggest merging when adding content similar to existing memories
- `--force` flag to bypass duplicate check

#### Memory Relationships (4A, 4C)
- `Related:` field in memory entries to link related memories
- Auto-suggest relations based on content similarity when adding

#### Memory Consolidation (5A, 5B)
- `merge <id1> <id2>` command to combine memories (--keep first/second/both)
- `consolidate` command to find memories that could be merged

#### Category System (7A-7D)
- Custom categories now allowed (not limited to predefined list)
- Sub-categories via `/` syntax (e.g., `arch/database` -> `architecture-database`)
- `reclassify <id> <category>` command to change memory category
- Category aliases: arch, disc, pat, bug, cfg, conf, ent, dec, ref, ctx

#### Tag Management (8A-8D)
- `tags` command to list all tags with usage counts
- `tags cleanup` to find rarely-used tags
- Tag suggestions when adding new memories
- `--tag` search filter

#### Session Improvements (9C-9D, 10A-10D, 11A-11D, 12A-12B)
- `--important` flag to mark entries that shouldn't be accidentally deleted
- Auto-archive important completed todos when starting new session
- `blocked_by` field to link blocked entries to blockers
- Status transition validation (pending -> in_progress -> completed)
- `--force` flag to override status transition rules
- Status history tracking with timestamps
- Session-ID header in session.md
- `session new` command to start fresh session
- `session search <query>` command
- `search-all <query>` for unified search across knowledge + session

#### Edit Commands (14A-14C)
- `edit <id>` to open memory in $EDITOR
- `update <id> --content <new>` for inline content update
- `update <id> --tags <tags>` for tag update
- `update <id> --related <ids>` for relations update

#### Backup/Recovery (20A-20D)
- Auto-backup before destructive operations (delete, merge, repair)
- `backup` command to create manual backup
- `backup list` to show available backups
- `backup restore:<filename>` to restore from backup
- Auto-prune old backups (keeps last 10)

#### Integrity Validation (21A-21C)
- `validate` command to check knowledge file integrity
- Detection of duplicate IDs, invalid formats, missing fields
- `repair` command with dry-run mode (--execute to apply)
- Auto-quarantine of unrepairable entries

### Changed
- Version bumped to 4.0.0
- Improved help text with new commands documented
- Exit code 2 for warnings (duplicate detection)

### Removed
- Shell completions (deemed unnecessary)

## [3.1.0] - 2025-12-30

### Added
- **Session Memory**: New temporary storage system for work-in-progress state
- `session.md` file for storing plans, todos, progress, and session-specific context
- `SessionEntry` dataclass with optional status field
- Session commands: `session add`, `session list`, `session show`, `session update`, `session delete`, `session clear`, `session archive`
- Session categories: plan, todo, progress, note, context, decision, blocker
- Session statuses for todos: pending, in_progress, completed, blocked
- `--status` and `--content` CLI arguments for session operations
- Archive functionality to promote session entries to permanent knowledge

### Changed
- Updated CLAUDE.md with comprehensive session memory documentation
- Added session.md to .gitignore (temporary/user-specific)

## [3.0.0] - 2025-12-30

### Changed
- **Breaking**: Removed SQLite database entirely - now pure file-based
- Search uses simple keyword matching instead of FTS5/BM25
- Results sorted by match count, then recency (changed_at)
- Simplified codebase by ~400 lines

### Removed
- SQLite database and all database-related code
- `rebuild` command (no longer needed without database)
- `--mode` search option (always uses keyword matching)
- BM25 relevance ranking (replaced with keyword counting)
- Database integrity checks in `maintain` command

### Added
- `tokenize()` function for keyword extraction
- `calculate_match_score()` for simple relevance scoring
- `search_memories()` for file-based search

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
