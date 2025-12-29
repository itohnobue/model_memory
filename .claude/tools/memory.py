#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# -*- coding: utf-8 -*-
"""
Model Memory Tool - Persistent Knowledge Base for Claude Code

A hybrid memory system combining:
- Human-readable Markdown files for knowledge storage
- SQLite FTS5 for fast full-text search
- Auto-sync on every query

Usage:
    memory.py add <category> <content> [--tags tag1,tag2]
    memory.py search <query> [--limit N]
    memory.py context <topic>           # Get context block for a topic
    memory.py list [--category CAT]     # List all memories
    memory.py rebuild                   # Force rebuild index from markdown
    memory.py delete <id>               # Delete a memory by ID
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__version__ = "2.0.0"


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
KNOWLEDGE_FILE = PROJECT_ROOT / "knowledge.md"
DB_PATH = PROJECT_ROOT / "memory.db"

# Categories for organizing knowledge
CATEGORIES = [
    "architecture",   # System design, structure
    "discovery",      # Things learned during exploration
    "pattern",        # Code patterns, conventions
    "gotcha",         # Bugs, workarounds, edge cases
    "config",         # Configuration, environment
    "entity",         # Key classes, functions, APIs
    "decision",       # Design decisions, rationale
    "todo",           # Pending items, follow-ups
    "reference",      # External links, docs
    "context",        # Project-specific context
]

# Regex for valid category names (alphanumeric, underscore, hyphen only)
VALID_CATEGORY_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]*$')


def sanitize_category(category: str) -> str:
    """Sanitize category name to prevent path traversal attacks."""
    # Remove any path separators and dangerous characters
    safe = re.sub(r'[./\\]', '', category)
    # Ensure it matches valid pattern
    if not VALID_CATEGORY_PATTERN.match(safe):
        safe = re.sub(r'[^a-zA-Z0-9_-]', '', safe)
        if not safe or not safe[0].isalpha():
            safe = 'misc'
    return safe.lower()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Memory:
    """A single memory entry."""
    id: str
    category: str
    content: str
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    source_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "content": self.content,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# =============================================================================
# MARKDOWN FILE OPERATIONS
# =============================================================================

def parse_knowledge_file() -> list[Memory]:
    """Parse the knowledge file into memory entries."""
    memories = []
    if not KNOWLEDGE_FILE.exists():
        return memories

    content = KNOWLEDGE_FILE.read_text(encoding="utf-8")

    # Split by memory entry markers (## followed by ID)
    pattern = r'^## \[([^\]]+)\](.*?)(?=^## \[|\Z)'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

    for memory_id, body in matches:
        body = body.strip()
        if not body:
            continue

        # Extract category (required in single-file format)
        category = "misc"
        cat_match = re.search(r'^Category:\s*(.+)$', body, re.MULTILINE)
        if cat_match:
            category = sanitize_category(cat_match.group(1).strip())
            body = re.sub(r'^Category:\s*.+\n?', '', body, flags=re.MULTILINE)

        # Extract tags if present
        tags = []
        tag_match = re.search(r'^Tags:\s*(.+)$', body, re.MULTILINE)
        if tag_match:
            tags = [t.strip() for t in tag_match.group(1).split(',')]
            body = re.sub(r'^Tags:\s*.+\n?', '', body, flags=re.MULTILINE)

        # Extract timestamps
        created_at = ""
        updated_at = ""
        created_match = re.search(r'^Created:\s*(.+)$', body, re.MULTILINE)
        if created_match:
            created_at = created_match.group(1).strip()
            body = re.sub(r'^Created:\s*.+\n?', '', body, flags=re.MULTILINE)

        updated_match = re.search(r'^Updated:\s*(.+)$', body, re.MULTILINE)
        if updated_match:
            updated_at = updated_match.group(1).strip()
            body = re.sub(r'^Updated:\s*.+\n?', '', body, flags=re.MULTILINE)

        # Clean up content
        content_text = body.strip()

        if content_text:
            memories.append(Memory(
                id=memory_id,
                category=category,
                content=content_text,
                tags=tags,
                created_at=created_at,
                updated_at=updated_at,
            ))

    return memories


def write_knowledge_file(memories: list[Memory]) -> None:
    """Write all memories to the knowledge file."""
    lines = ["# Knowledge Base\n"]
    lines.append(f"Last updated: {datetime.now().isoformat()}\n\n")

    for memory in memories:
        lines.append(f"## [{memory.id}]\n")
        lines.append(f"Category: {memory.category}\n")
        if memory.tags:
            lines.append(f"Tags: {', '.join(memory.tags)}\n")
        if memory.created_at:
            lines.append(f"Created: {memory.created_at}\n")
        if memory.updated_at:
            lines.append(f"Updated: {memory.updated_at}\n")
        lines.append(f"\n{memory.content}\n\n")

    KNOWLEDGE_FILE.write_text("".join(lines), encoding="utf-8")


def add_memory_to_file(memory: Memory) -> None:
    """Add or update a memory in the knowledge file."""
    existing_memories = parse_knowledge_file()

    # Check if memory already exists (update) or is new (add)
    found = False
    for i, m in enumerate(existing_memories):
        if m.id == memory.id:
            existing_memories[i] = memory
            found = True
            break

    if not found:
        existing_memories.append(memory)

    write_knowledge_file(existing_memories)


def delete_memory_from_file(memory_id: str) -> bool:
    """Delete a memory from the knowledge file."""
    if not KNOWLEDGE_FILE.exists():
        return False

    memories = parse_knowledge_file()
    original_count = len(memories)
    memories = [m for m in memories if m.id != memory_id]

    if len(memories) < original_count:
        write_knowledge_file(memories)
        return True
    return False


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def get_db_connection() -> sqlite3.Connection:
    """Get database connection with FTS5 support."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrent access
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_database() -> None:
    """Initialize the database with FTS5 table."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Main memories table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT,
            created_at TEXT,
            updated_at TEXT,
            content_hash TEXT
        )
    """)

    # FTS5 virtual table for full-text search
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            id,
            category,
            content,
            tags,
            content='memories',
            content_rowid='rowid'
        )
    """)

    # Triggers to keep FTS in sync
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, id, category, content, tags)
            VALUES (new.rowid, new.id, new.category, new.content, new.tags);
        END
    """)

    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, id, category, content, tags)
            VALUES('delete', old.rowid, old.id, old.category, old.content, old.tags);
        END
    """)

    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, id, category, content, tags)
            VALUES('delete', old.rowid, old.id, old.category, old.content, old.tags);
            INSERT INTO memories_fts(rowid, id, category, content, tags)
            VALUES (new.rowid, new.id, new.category, new.content, new.tags);
        END
    """)

    # File hashes table for sync tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_hashes (
            filepath TEXT PRIMARY KEY,
            hash TEXT NOT NULL,
            last_sync TEXT
        )
    """)

    # Index for category filtering
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)
    """)

    # Index for date-based sorting (list, maintain commands)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC)
    """)

    # Composite index for category + date (optimizes list --category with ORDER BY)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_cat_date ON memories(category, created_at DESC)
    """)

    conn.commit()
    conn.close()


def compute_file_hash(filepath: Path) -> str:
    """Compute hash of a file's content."""
    if not filepath.exists():
        return ""
    content = filepath.read_bytes()
    return hashlib.md5(content).hexdigest()


def needs_sync(conn: sqlite3.Connection) -> bool:
    """Check if knowledge file has changed since last sync."""
    cursor = conn.cursor()

    current_hash = compute_file_hash(KNOWLEDGE_FILE)
    cursor.execute(
        "SELECT hash FROM file_hashes WHERE filepath = ?",
        (str(KNOWLEDGE_FILE),)
    )
    row = cursor.fetchone()

    if row is None or row["hash"] != current_hash:
        return True

    return False


def sync_from_file(conn: sqlite3.Connection, force: bool = False) -> None:
    """Sync database from knowledge file."""
    if not force and not needs_sync(conn):
        return

    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM memories")

    # Rebuild FTS index
    cursor.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")

    # Load all memories from file
    all_memories = parse_knowledge_file()

    # Insert into database
    for memory in all_memories:
        content_hash = hashlib.md5(memory.content.encode()).hexdigest()
        cursor.execute("""
            INSERT OR REPLACE INTO memories (id, category, content, tags, created_at, updated_at, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.id,
            memory.category,
            memory.content,
            ",".join(memory.tags),
            memory.created_at,
            memory.updated_at,
            content_hash,
        ))

    # Update file hash
    cursor.execute("DELETE FROM file_hashes")
    cursor.execute(
        "INSERT INTO file_hashes (filepath, hash, last_sync) VALUES (?, ?, ?)",
        (str(KNOWLEDGE_FILE), compute_file_hash(KNOWLEDGE_FILE), datetime.now().isoformat())
    )

    conn.commit()


def generate_memory_id(category: str, content: str) -> str:
    """Generate a unique ID for a memory."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    content_hash = hashlib.md5(content.encode()).hexdigest()[:6]
    return f"{category[:3]}-{timestamp}-{content_hash}"


# =============================================================================
# COMMANDS
# =============================================================================

def cmd_add(category: str, content: str, tags: list[str] | None = None) -> dict[str, Any]:
    """Add a new memory."""
    init_database()

    # Sanitize category
    safe_category = sanitize_category(category)

    # Generate ID and timestamps
    memory_id = generate_memory_id(safe_category, content)
    now = datetime.now().isoformat()

    memory = Memory(
        id=memory_id,
        category=safe_category,
        content=content,
        tags=tags or [],
        created_at=now,
        updated_at=now,
    )

    # Write to knowledge file
    add_memory_to_file(memory)

    # Sync to database
    conn = get_db_connection()
    sync_from_file(conn, force=True)
    conn.close()

    return {
        "status": "success",
        "message": f"Memory added with ID: {memory_id}",
        "memory": memory.to_dict(),
    }


def escape_fts5_query(query: str) -> str:
    """Escape special FTS5 characters and wrap in quotes for safe searching.

    This creates an EXACT PHRASE search. For keyword OR search, use build_or_query().
    """
    # FTS5 special characters: " * : ^ ( ) -
    # Escape by wrapping entire query in quotes after escaping internal quotes
    escaped = query.replace('"', '""')
    return f'"{escaped}"'


def build_or_query(query: str) -> str:
    """Build an FTS5 OR query from space-separated keywords.

    Tokenizes the input and creates a query that matches ANY of the keywords.
    Each keyword is quoted for safety and connected with OR.

    Example: "vespa-linux server docker" -> '"vespa-linux" OR "server" OR "docker"'

    Also supports prefix matching with * for partial matches.
    """
    # Handle explicit quoted phrases - pass through as-is
    if query.startswith('"') and query.endswith('"'):
        return escape_fts5_query(query[1:-1])

    # Tokenize: split on whitespace and filter empty tokens
    tokens = [t.strip() for t in query.split() if t.strip()]

    if not tokens:
        return '""'

    if len(tokens) == 1:
        # Single token: escape and add prefix wildcard for partial matches
        escaped = tokens[0].replace('"', '""')
        # Return both exact and prefix match
        return f'"{escaped}" OR "{escaped}"*'

    # Multiple tokens: create OR query with prefix matching
    or_parts = []
    for token in tokens:
        escaped = token.replace('"', '""')
        # Each token matches exact or prefix
        or_parts.append(f'"{escaped}"')
        or_parts.append(f'"{escaped}"*')

    return " OR ".join(or_parts)


def cmd_search(
    query: str,
    limit: int = 10,
    category: str | None = None,
    mode: str = "keywords",
) -> dict[str, Any]:
    """Search memories using FTS5 with recency boost.

    Args:
        query: Search query string
        limit: Maximum number of results
        category: Optional category filter
        mode: Search mode - "keywords" for OR search (default), "phrase" for exact phrase

    Ranking combines BM25 relevance with recency:
    - BM25 score measures text relevance (lower = better match)
    - Recency factor boosts newer memories
    - Formula: bm25_score - (recency_weight / age_days)
    """
    init_database()
    conn = get_db_connection()

    # Auto-sync before search
    sync_from_file(conn)

    cursor = conn.cursor()

    # Build FTS5 query based on mode
    if mode == "keywords":
        safe_query = build_or_query(query)
    else:
        safe_query = escape_fts5_query(query)

    # Recency-boosted ranking:
    # - bm25() returns negative scores (more negative = better match)
    # - We subtract a recency bonus (newer = larger bonus = more negative final score)
    # - age_days = days since creation (minimum 1 to avoid division issues)
    # - recency_bonus = 10.0 / age_days (newer memories get bigger bonus)
    recency_sql = """
        SELECT m.id, m.category, m.content, m.tags, m.created_at, m.updated_at,
               bm25(memories_fts) as bm25_score,
               MAX(1, julianday('now') - julianday(COALESCE(m.created_at, '2020-01-01'))) as age_days,
               bm25(memories_fts) - (10.0 / MAX(1, julianday('now') - julianday(COALESCE(m.created_at, '2020-01-01')))) as rank
        FROM memories m
        JOIN memories_fts ON m.rowid = memories_fts.rowid
        WHERE memories_fts MATCH ?
    """

    if category:
        safe_category = sanitize_category(category)
        cursor.execute(recency_sql + " AND m.category = ? ORDER BY rank LIMIT ?",
                      (safe_query, safe_category, limit))
    else:
        cursor.execute(recency_sql + " ORDER BY rank LIMIT ?",
                      (safe_query, limit))

    results = []
    for row in cursor.fetchall():
        results.append({
            "id": row["id"],
            "category": row["category"],
            "content": row["content"],
            "tags": [t for t in row["tags"].split(",") if t] if row["tags"] else [],
            "created_at": row["created_at"],
            "age_days": int(row["age_days"]),
            "relevance": -row["bm25_score"],  # BM25 returns negative scores
        })

    conn.close()

    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


def cmd_context(topic: str, limit: int = 5) -> dict[str, Any]:
    """Get a context block for a topic - formatted for injection into prompts.

    Uses keyword (OR) search mode to find memories matching ANY of the provided
    keywords. This is more flexible than exact phrase matching for context retrieval.
    """
    result = cmd_search(topic, limit=limit, mode="keywords")

    if not result["results"]:
        return {
            "topic": topic,
            "context": "",
            "count": 0,
        }

    # Format as a context block
    lines = [f"## Relevant Knowledge for: {topic}\n"]

    for mem in result["results"]:
        lines.append(f"### [{mem['category']}] {mem['id']}")
        if mem["tags"]:
            lines.append(f"Tags: {', '.join(mem['tags'])}")
        lines.append(mem["content"])
        lines.append("")

    return {
        "topic": topic,
        "context": "\n".join(lines),
        "count": len(result["results"]),
    }


def cmd_list(category: str | None = None, limit: int = 50) -> dict[str, Any]:
    """List all memories, optionally filtered by category."""
    init_database()
    conn = get_db_connection()

    # Auto-sync
    sync_from_file(conn)

    cursor = conn.cursor()

    if category:
        safe_category = sanitize_category(category)
        cursor.execute("""
            SELECT id, category, content, tags, created_at, updated_at
            FROM memories
            WHERE category = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (safe_category, limit))
    else:
        cursor.execute("""
            SELECT id, category, content, tags, created_at, updated_at
            FROM memories
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

    results = []
    for row in cursor.fetchall():
        results.append({
            "id": row["id"],
            "category": row["category"],
            "content": row["content"][:100] + "..." if len(row["content"]) > 100 else row["content"],
            "tags": [t for t in row["tags"].split(",") if t] if row["tags"] else [],
            "created_at": row["created_at"],
        })

    conn.close()

    return {
        "count": len(results),
        "category": category,
        "results": results,
    }


def cmd_rebuild() -> dict[str, Any]:
    """Force rebuild the index from markdown files."""
    init_database()
    conn = get_db_connection()
    sync_from_file(conn, force=True)

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM memories")
    count = cursor.fetchone()["count"]

    conn.close()

    return {
        "status": "success",
        "message": f"Index rebuilt with {count} memories",
        "count": count,
    }


def cmd_delete(memory_id: str) -> dict[str, Any]:
    """Delete a memory by ID."""
    init_database()
    conn = get_db_connection()

    # Delete from knowledge file
    if delete_memory_from_file(memory_id):
        # Sync database
        sync_from_file(conn, force=True)
        conn.close()
        return {
            "status": "success",
            "message": f"Memory deleted: {memory_id}",
        }

    conn.close()
    return {
        "status": "error",
        "message": f"Failed to delete memory: {memory_id}",
    }


def cmd_stats() -> dict[str, Any]:
    """Get statistics about the memory database."""
    init_database()
    conn = get_db_connection()
    sync_from_file(conn)

    cursor = conn.cursor()

    # Total count
    cursor.execute("SELECT COUNT(*) as count FROM memories")
    total = cursor.fetchone()["count"]

    # Count by category
    cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM memories
        GROUP BY category
        ORDER BY count DESC
    """)
    by_category = {row["category"]: row["count"] for row in cursor.fetchall()}

    conn.close()

    return {
        "total_memories": total,
        "by_category": by_category,
        "knowledge_file": str(KNOWLEDGE_FILE),
        "database": str(DB_PATH),
    }


def cmd_maintain(max_age_days: int | None = None, dry_run: bool = True) -> dict[str, Any]:
    """Maintain database: analyze age distribution and optionally clean old memories.

    Args:
        max_age_days: If set, identify/delete memories older than this
        dry_run: If True, only report what would be deleted (default)
    """
    init_database()
    conn = get_db_connection()
    sync_from_file(conn)

    cursor = conn.cursor()

    # Age distribution
    cursor.execute("""
        SELECT
            CASE
                WHEN julianday('now') - julianday(COALESCE(created_at, '2020-01-01')) <= 7 THEN 'last_week'
                WHEN julianday('now') - julianday(COALESCE(created_at, '2020-01-01')) <= 30 THEN 'last_month'
                WHEN julianday('now') - julianday(COALESCE(created_at, '2020-01-01')) <= 90 THEN 'last_quarter'
                WHEN julianday('now') - julianday(COALESCE(created_at, '2020-01-01')) <= 365 THEN 'last_year'
                ELSE 'older'
            END as age_bucket,
            COUNT(*) as count
        FROM memories
        GROUP BY age_bucket
    """)
    age_distribution = {row["age_bucket"]: row["count"] for row in cursor.fetchall()}

    # Ensure all buckets exist
    for bucket in ["last_week", "last_month", "last_quarter", "last_year", "older"]:
        if bucket not in age_distribution:
            age_distribution[bucket] = 0

    result = {
        "age_distribution": age_distribution,
        "total_memories": sum(age_distribution.values()),
    }

    # If max_age specified, find old memories
    if max_age_days is not None:
        cursor.execute("""
            SELECT id, category, content, created_at,
                   CAST(julianday('now') - julianday(COALESCE(created_at, '2020-01-01')) AS INTEGER) as age_days
            FROM memories
            WHERE julianday('now') - julianday(COALESCE(created_at, '2020-01-01')) > ?
            ORDER BY age_days DESC
        """, (max_age_days,))

        old_memories = []
        for row in cursor.fetchall():
            old_memories.append({
                "id": row["id"],
                "category": row["category"],
                "content": row["content"][:80] + "..." if len(row["content"]) > 80 else row["content"],
                "age_days": row["age_days"],
            })

        result["old_memories"] = old_memories
        result["old_count"] = len(old_memories)
        result["max_age_days"] = max_age_days
        result["dry_run"] = dry_run

        # Delete if not dry run
        if not dry_run and old_memories:
            deleted = 0
            for mem in old_memories:
                if delete_memory_from_file(mem["id"]):
                    deleted += 1
            sync_from_file(conn, force=True)
            result["deleted"] = deleted
            result["message"] = f"Deleted {deleted} memories older than {max_age_days} days"
        elif old_memories:
            result["message"] = f"Found {len(old_memories)} memories older than {max_age_days} days (dry run - use --execute to delete)"

    # Database integrity check
    cursor.execute("PRAGMA integrity_check")
    integrity = cursor.fetchone()[0]
    result["db_integrity"] = integrity

    # FTS index status
    cursor.execute("SELECT COUNT(*) FROM memories_fts")
    fts_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM memories")
    mem_count = cursor.fetchone()[0]
    result["index_synced"] = fts_count == mem_count

    conn.close()
    return result


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_output(data: dict[str, Any], output_format: str = "text") -> str:
    """Format output based on requested format."""
    if output_format == "json":
        return json.dumps(data, indent=2, ensure_ascii=False)

    # Text format
    # Handle context command (has "topic" and "context" keys)
    if "topic" in data and "context" in data:
        if data.get("context"):
            return data["context"]
        return f"No context found for: {data['topic']}"

    if "results" in data:
        lines = []
        if "query" in data:
            lines.append(f"Search: \"{data['query']}\" ({data['count']} results)\n")

        for result in data.get("results", []):
            lines.append(f"[{result['category']}] {result['id']}")
            if result.get("tags"):
                lines.append(f"  Tags: {', '.join(result['tags'])}")
            content = result.get("content", "")
            # Indent content
            for line in content.split("\n"):
                lines.append(f"  {line}")
            lines.append("")

        return "\n".join(lines)

    if "memory" in data:
        mem = data["memory"]
        return f"Added [{mem['category']}] {mem['id']}\n{mem['content']}"

    if "message" in data:
        return data["message"]

    if "total_memories" in data and "by_category" in data:
        lines = [f"Total memories: {data['total_memories']}"]
        lines.append("\nBy category:")
        for cat, count in data["by_category"].items():
            lines.append(f"  {cat}: {count}")
        return "\n".join(lines)

    if "age_distribution" in data:
        lines = ["Database Maintenance Report", "=" * 30]
        lines.append(f"\nTotal memories: {data['total_memories']}")
        lines.append("\nAge distribution:")
        order = ["last_week", "last_month", "last_quarter", "last_year", "older"]
        labels = {"last_week": "< 1 week", "last_month": "< 1 month", "last_quarter": "< 3 months",
                  "last_year": "< 1 year", "older": "> 1 year"}
        for bucket in order:
            count = data["age_distribution"].get(bucket, 0)
            lines.append(f"  {labels[bucket]}: {count}")

        lines.append(f"\nDatabase integrity: {data.get('db_integrity', 'unknown')}")
        lines.append(f"Index synced: {'Yes' if data.get('index_synced') else 'No'}")

        if "old_memories" in data:
            lines.append(f"\nMemories older than {data['max_age_days']} days: {data['old_count']}")
            if data["old_memories"]:
                for mem in data["old_memories"][:10]:  # Show max 10
                    lines.append(f"  [{mem['category']}] {mem['id']} ({mem['age_days']}d)")
                    lines.append(f"    {mem['content']}")
                if len(data["old_memories"]) > 10:
                    lines.append(f"  ... and {len(data['old_memories']) - 10} more")

        if "message" in data:
            lines.append(f"\n{data['message']}")

        return "\n".join(lines)

    return json.dumps(data, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def sanitize_error_message(error: Exception) -> str:
    """Sanitize error message to avoid leaking internal paths."""
    msg = str(error)
    # Remove potential file paths
    msg = re.sub(r'/[^\s]+/', '[path]/', msg)
    msg = re.sub(r'[A-Z]:\\[^\s]+\\', '[path]\\\\', msg)
    return msg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model Memory Tool - Persistent Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Commands:
  add <category> <content>    Add a new memory
  search <query>              Search memories (ranked by relevance + recency)
  context <topic>             Get context block for a topic (uses keyword OR search)
  list                        List all memories
  rebuild                     Force rebuild index
  delete <id>                 Delete a memory
  stats                       Show statistics
  maintain                    Check database health and clean old memories

Search Modes:
  --mode keywords   OR matching - finds memories with ANY keyword (default)
  --mode phrase     Exact phrase matching

Categories: {', '.join(CATEGORIES)}

Examples:
  memory.sh add discovery "The API uses OAuth2 with PKCE flow"
  memory.sh add gotcha "Redis connection pool must be closed explicitly" --tags redis,connection
  memory.sh search "authentication"
  memory.sh search "vespa server docker" --mode keywords   # Match ANY keyword
  memory.sh context "vespa-linux server services"          # Uses keyword mode by default
  memory.sh list --category gotcha
  memory.sh maintain --max-age 180
        """
    )

    parser.add_argument("command", nargs="?", choices=["add", "search", "context", "list", "rebuild", "delete", "stats", "maintain"],
                        help="Command to execute")
    parser.add_argument("args", nargs="*", help="Command arguments")
    parser.add_argument("--tags", "-t", help="Comma-separated tags (for add)")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Limit results")
    parser.add_argument("--category", "-c", help="Filter by category")
    parser.add_argument("--output", "-o", choices=["text", "json"], default="text",
                        help="Output format")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential output")
    parser.add_argument("--version", "-v", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--max-age", type=int, help="Max age in days for maintain command")
    parser.add_argument("--execute", action="store_true", help="Actually delete old memories (maintain command)")
    parser.add_argument("--mode", "-m", choices=["phrase", "keywords"], default="keywords",
                        help="Search mode: 'keywords' for OR search (default), 'phrase' for exact match")

    args = parser.parse_args()

    # Handle no command
    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "add":
            if len(args.args) < 2:
                print("Error: add requires <category> <content>", file=sys.stderr)
                sys.exit(1)
            category = args.args[0]
            content = " ".join(args.args[1:])
            tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
            result = cmd_add(category, content, tags)

        elif args.command == "search":
            if not args.args:
                print("Error: search requires <query>", file=sys.stderr)
                sys.exit(1)
            query = " ".join(args.args)
            result = cmd_search(query, limit=args.limit, category=args.category, mode=args.mode)

        elif args.command == "context":
            if not args.args:
                print("Error: context requires <topic>", file=sys.stderr)
                sys.exit(1)
            topic = " ".join(args.args)
            result = cmd_context(topic, limit=args.limit)

        elif args.command == "list":
            result = cmd_list(category=args.category, limit=args.limit)

        elif args.command == "rebuild":
            result = cmd_rebuild()

        elif args.command == "delete":
            if not args.args:
                print("Error: delete requires <id>", file=sys.stderr)
                sys.exit(1)
            result = cmd_delete(args.args[0])

        elif args.command == "stats":
            result = cmd_stats()

        elif args.command == "maintain":
            result = cmd_maintain(max_age_days=args.max_age, dry_run=not args.execute)

        # Check for error status and exit appropriately
        is_error = result.get("status") == "error"

        if not args.quiet:
            output = format_output(result, args.output)
            if is_error:
                print(output, file=sys.stderr)
            else:
                print(output)
        elif args.output == "json":
            # Always output JSON if requested, even in quiet mode
            print(format_output(result, "json"))

        if is_error:
            sys.exit(1)

    except Exception as e:
        print(f"Error: {sanitize_error_message(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
