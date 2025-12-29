#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# -*- coding: utf-8 -*-
"""
Model Memory Tool - Persistent Knowledge Base for Claude Code

A simple file-based memory system using only knowledge.md for storage.
No database required - just plain text with keyword search.

Usage:
    memory.py add <category> <content> [--tags tag1,tag2]
    memory.py search <query> [--limit N]
    memory.py context <topic>           # Get context block for a topic
    memory.py list [--category CAT]     # List all memories
    memory.py delete <id>               # Delete a memory by ID
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__version__ = "3.1.0"


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
KNOWLEDGE_FILE = PROJECT_ROOT / "knowledge.md"
SESSION_FILE = PROJECT_ROOT / "session.md"

# Categories for organizing long-term knowledge
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

# Categories for session-specific temporary storage
SESSION_CATEGORIES = [
    "plan",           # Implementation plans, task breakdowns
    "todo",           # Task items with status
    "progress",       # Log entries of what was done
    "note",           # General session information
    "context",        # Session-specific findings (not permanent)
    "decision",       # Tentative decisions (may become permanent)
    "blocker",        # Issues blocking progress
]

# Valid statuses for session todos
SESSION_STATUSES = ["pending", "in_progress", "completed", "blocked"]

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
    changed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "content": self.content,
            "tags": self.tags,
            "changed_at": self.changed_at,
        }


@dataclass
class SessionEntry:
    """A single session memory entry with optional status."""
    id: str
    category: str
    content: str
    status: str = ""  # For todos: pending, in_progress, completed, blocked
    changed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "category": self.category,
            "content": self.content,
            "changed_at": self.changed_at,
        }
        if self.status:
            result["status"] = self.status
        return result


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

        # Extract timestamp
        changed_at = ""
        changed_match = re.search(r'^Changed:\s*(.+)$', body, re.MULTILINE)
        if changed_match:
            changed_at = changed_match.group(1).strip()
            body = re.sub(r'^Changed:\s*.+\n?', '', body, flags=re.MULTILINE)

        # Clean up content
        content_text = body.strip()

        if content_text:
            memories.append(Memory(
                id=memory_id,
                category=category,
                content=content_text,
                tags=tags,
                changed_at=changed_at,
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
        if memory.changed_at:
            lines.append(f"Changed: {memory.changed_at}\n")
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
# SESSION FILE OPERATIONS
# =============================================================================

def parse_session_file() -> list[SessionEntry]:
    """Parse the session file into session entries."""
    entries = []
    if not SESSION_FILE.exists():
        return entries

    content = SESSION_FILE.read_text(encoding="utf-8")

    # Split by entry markers (## followed by ID)
    pattern = r'^## \[([^\]]+)\](.*?)(?=^## \[|\Z)'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

    for entry_id, body in matches:
        body = body.strip()
        if not body:
            continue

        # Extract category
        category = "note"
        cat_match = re.search(r'^Category:\s*(.+)$', body, re.MULTILINE)
        if cat_match:
            category = sanitize_category(cat_match.group(1).strip())
            body = re.sub(r'^Category:\s*.+\n?', '', body, flags=re.MULTILINE)

        # Extract status if present
        status = ""
        status_match = re.search(r'^Status:\s*(.+)$', body, re.MULTILINE)
        if status_match:
            status = status_match.group(1).strip().lower()
            if status not in SESSION_STATUSES:
                status = ""
            body = re.sub(r'^Status:\s*.+\n?', '', body, flags=re.MULTILINE)

        # Extract timestamp
        changed_at = ""
        changed_match = re.search(r'^Changed:\s*(.+)$', body, re.MULTILINE)
        if changed_match:
            changed_at = changed_match.group(1).strip()
            body = re.sub(r'^Changed:\s*.+\n?', '', body, flags=re.MULTILINE)

        content_text = body.strip()

        if content_text:
            entries.append(SessionEntry(
                id=entry_id,
                category=category,
                content=content_text,
                status=status,
                changed_at=changed_at,
            ))

    return entries


def write_session_file(entries: list[SessionEntry]) -> None:
    """Write all session entries to the session file."""
    lines = ["# Session Memory\n"]
    lines.append(f"Last updated: {datetime.now().isoformat()}\n\n")

    for entry in entries:
        lines.append(f"## [{entry.id}]\n")
        lines.append(f"Category: {entry.category}\n")
        if entry.status:
            lines.append(f"Status: {entry.status}\n")
        if entry.changed_at:
            lines.append(f"Changed: {entry.changed_at}\n")
        lines.append(f"\n{entry.content}\n\n")

    SESSION_FILE.write_text("".join(lines), encoding="utf-8")


def add_session_entry(entry: SessionEntry) -> None:
    """Add or update an entry in the session file."""
    existing_entries = parse_session_file()

    found = False
    for i, e in enumerate(existing_entries):
        if e.id == entry.id:
            existing_entries[i] = entry
            found = True
            break

    if not found:
        existing_entries.append(entry)

    write_session_file(existing_entries)


def delete_session_entry(entry_id: str) -> bool:
    """Delete an entry from the session file."""
    if not SESSION_FILE.exists():
        return False

    entries = parse_session_file()
    original_count = len(entries)
    entries = [e for e in entries if e.id != entry_id]

    if len(entries) < original_count:
        write_session_file(entries)
        return True
    return False


def clear_session_file() -> int:
    """Clear all entries from session file. Returns count of cleared entries."""
    if not SESSION_FILE.exists():
        return 0

    entries = parse_session_file()
    count = len(entries)

    if count > 0:
        SESSION_FILE.unlink()

    return count


def generate_session_id(category: str, content: str) -> str:
    """Generate a unique ID for a session entry."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    content_hash = hashlib.md5(content.encode()).hexdigest()[:4]
    return f"s-{category[:3]}-{timestamp}-{content_hash}"


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================

def generate_memory_id(category: str, content: str) -> str:
    """Generate a unique ID for a memory."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    content_hash = hashlib.md5(content.encode()).hexdigest()[:6]
    return f"{category[:3]}-{timestamp}-{content_hash}"


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words."""
    # Split on non-alphanumeric characters, keep hyphenated words
    words = re.findall(r'[a-zA-Z0-9][-a-zA-Z0-9]*', text.lower())
    return words


def calculate_match_score(memory: Memory, keywords: list[str]) -> int:
    """Calculate how many keywords match in a memory.

    Searches in: content, category, tags, and ID.
    Returns count of matching keywords (higher = better).
    """
    # Build searchable text
    searchable = f"{memory.content} {memory.category} {' '.join(memory.tags)} {memory.id}".lower()

    score = 0
    for keyword in keywords:
        keyword_lower = keyword.lower()
        # Exact word match or substring match
        if keyword_lower in searchable:
            score += 1
            # Bonus for multiple occurrences
            score += searchable.count(keyword_lower) - 1

    return score


def search_memories(
    query: str,
    memories: list[Memory],
    limit: int = 10,
    category: str | None = None,
) -> list[tuple[Memory, int]]:
    """Search memories using keyword matching.

    Returns list of (memory, score) tuples sorted by score desc, then recency.
    """
    keywords = tokenize(query)
    if not keywords:
        return []

    # Filter by category if specified
    if category:
        safe_category = sanitize_category(category)
        memories = [m for m in memories if m.category == safe_category]

    # Score each memory
    scored = []
    for memory in memories:
        score = calculate_match_score(memory, keywords)
        if score > 0:
            scored.append((memory, score))

    # Sort by score (desc), then by changed_at (desc for recency)
    scored.sort(key=lambda x: (x[1], x[0].changed_at or ""), reverse=True)

    return scored[:limit]


def calculate_age_days(changed_at: str) -> int:
    """Calculate age in days from changed_at timestamp."""
    if not changed_at:
        return 9999  # Very old if no timestamp
    try:
        dt = datetime.fromisoformat(changed_at)
        age = datetime.now() - dt
        return max(1, age.days)
    except (ValueError, TypeError):
        return 9999


# =============================================================================
# COMMANDS
# =============================================================================

def cmd_add(category: str, content: str, tags: list[str] | None = None) -> dict[str, Any]:
    """Add a new memory."""
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
        changed_at=now,
    )

    # Write to knowledge file
    add_memory_to_file(memory)

    return {
        "status": "success",
        "message": f"Memory added with ID: {memory_id}",
        "memory": memory.to_dict(),
    }


def cmd_search(
    query: str,
    limit: int = 10,
    category: str | None = None,
    mode: str = "keywords",
) -> dict[str, Any]:
    """Search memories using keyword matching.

    Args:
        query: Search query string
        limit: Maximum number of results
        category: Optional category filter
        mode: Search mode (kept for compatibility, always uses keyword matching)
    """
    memories = parse_knowledge_file()
    scored_results = search_memories(query, memories, limit=limit, category=category)

    results = []
    for memory, score in scored_results:
        age_days = calculate_age_days(memory.changed_at)
        results.append({
            "id": memory.id,
            "category": memory.category,
            "content": memory.content,
            "tags": memory.tags,
            "changed_at": memory.changed_at,
            "age_days": age_days,
            "relevance": score,
        })

    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


def cmd_context(topic: str, limit: int = 5) -> dict[str, Any]:
    """Get a context block for a topic - formatted for injection into prompts."""
    result = cmd_search(topic, limit=limit)

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
    memories = parse_knowledge_file()

    # Filter by category if specified
    if category:
        safe_category = sanitize_category(category)
        memories = [m for m in memories if m.category == safe_category]

    # Sort by changed_at descending (most recent first)
    memories.sort(key=lambda m: m.changed_at or "", reverse=True)

    # Apply limit
    memories = memories[:limit]

    results = []
    for memory in memories:
        results.append({
            "id": memory.id,
            "category": memory.category,
            "content": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
            "tags": memory.tags,
            "changed_at": memory.changed_at,
        })

    return {
        "count": len(results),
        "category": category,
        "results": results,
    }


def cmd_delete(memory_id: str) -> dict[str, Any]:
    """Delete a memory by ID."""
    if delete_memory_from_file(memory_id):
        return {
            "status": "success",
            "message": f"Memory deleted: {memory_id}",
        }

    return {
        "status": "error",
        "message": f"Failed to delete memory: {memory_id}",
    }


def cmd_stats() -> dict[str, Any]:
    """Get statistics about the memory storage."""
    memories = parse_knowledge_file()

    # Count by category
    by_category: dict[str, int] = {}
    for memory in memories:
        by_category[memory.category] = by_category.get(memory.category, 0) + 1

    # Sort by count descending
    by_category = dict(sorted(by_category.items(), key=lambda x: x[1], reverse=True))

    return {
        "total_memories": len(memories),
        "by_category": by_category,
        "knowledge_file": str(KNOWLEDGE_FILE),
    }


def cmd_maintain() -> dict[str, Any]:
    """Analyze memory age distribution."""
    memories = parse_knowledge_file()

    # Age distribution (based on last change)
    age_distribution = {
        "last_week": 0,
        "last_month": 0,
        "last_quarter": 0,
        "last_year": 0,
        "older": 0,
    }

    for memory in memories:
        age_days = calculate_age_days(memory.changed_at)
        if age_days <= 7:
            age_distribution["last_week"] += 1
        elif age_days <= 30:
            age_distribution["last_month"] += 1
        elif age_days <= 90:
            age_distribution["last_quarter"] += 1
        elif age_days <= 365:
            age_distribution["last_year"] += 1
        else:
            age_distribution["older"] += 1

    return {
        "age_distribution": age_distribution,
        "total_memories": len(memories),
    }


# =============================================================================
# SESSION COMMANDS
# =============================================================================

def cmd_session_add(
    category: str,
    content: str,
    status: str = "",
) -> dict[str, Any]:
    """Add a new session entry."""
    safe_category = sanitize_category(category)

    # Validate status if provided
    if status and status.lower() not in SESSION_STATUSES:
        return {
            "status": "error",
            "message": f"Invalid status: {status}. Valid: {', '.join(SESSION_STATUSES)}",
        }

    entry_id = generate_session_id(safe_category, content)
    now = datetime.now().isoformat()

    entry = SessionEntry(
        id=entry_id,
        category=safe_category,
        content=content,
        status=status.lower() if status else "",
        changed_at=now,
    )

    add_session_entry(entry)

    return {
        "status": "success",
        "message": f"Session entry added: {entry_id}",
        "entry": entry.to_dict(),
    }


def cmd_session_list(
    category: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List session entries, optionally filtered."""
    entries = parse_session_file()

    # Filter by category
    if category:
        safe_category = sanitize_category(category)
        entries = [e for e in entries if e.category == safe_category]

    # Filter by status
    if status:
        status_lower = status.lower()
        entries = [e for e in entries if e.status == status_lower]

    # Sort by changed_at descending
    entries.sort(key=lambda e: e.changed_at or "", reverse=True)

    # Apply limit
    entries = entries[:limit]

    results = []
    for entry in entries:
        result = {
            "id": entry.id,
            "category": entry.category,
            "content": entry.content[:100] + "..." if len(entry.content) > 100 else entry.content,
            "changed_at": entry.changed_at,
        }
        if entry.status:
            result["status"] = entry.status
        results.append(result)

    return {
        "count": len(results),
        "category": category,
        "status": status,
        "results": results,
    }


def cmd_session_update(
    entry_id: str,
    status: str | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    """Update a session entry's status or content."""
    entries = parse_session_file()

    for entry in entries:
        if entry.id == entry_id:
            if status:
                if status.lower() not in SESSION_STATUSES:
                    return {
                        "status": "error",
                        "message": f"Invalid status: {status}. Valid: {', '.join(SESSION_STATUSES)}",
                    }
                entry.status = status.lower()

            if content:
                entry.content = content

            entry.changed_at = datetime.now().isoformat()
            write_session_file(entries)

            return {
                "status": "success",
                "message": f"Session entry updated: {entry_id}",
                "entry": entry.to_dict(),
            }

    return {
        "status": "error",
        "message": f"Session entry not found: {entry_id}",
    }


def cmd_session_delete(entry_id: str) -> dict[str, Any]:
    """Delete a session entry by ID."""
    if delete_session_entry(entry_id):
        return {
            "status": "success",
            "message": f"Session entry deleted: {entry_id}",
        }

    return {
        "status": "error",
        "message": f"Failed to delete session entry: {entry_id}",
    }


def cmd_session_clear() -> dict[str, Any]:
    """Clear all session entries."""
    count = clear_session_file()

    return {
        "status": "success",
        "message": f"Cleared {count} session entries",
        "cleared_count": count,
    }


def cmd_session_show() -> dict[str, Any]:
    """Show full session context - all entries formatted for review."""
    entries = parse_session_file()

    if not entries:
        return {
            "count": 0,
            "context": "",
        }

    # Group by category
    by_category: dict[str, list[SessionEntry]] = {}
    for entry in entries:
        if entry.category not in by_category:
            by_category[entry.category] = []
        by_category[entry.category].append(entry)

    lines = ["# Current Session State\n"]

    # Order: plan first, then todo (by status), then progress, then others
    category_order = ["plan", "todo", "progress", "note", "context", "decision", "blocker"]
    for cat in category_order:
        if cat not in by_category:
            continue
        cat_entries = by_category[cat]
        lines.append(f"\n## {cat.title()}\n")

        # For todos, group by status
        if cat == "todo":
            status_order = ["in_progress", "pending", "blocked", "completed"]
            for status in status_order:
                status_entries = [e for e in cat_entries if e.status == status]
                if status_entries:
                    lines.append(f"\n### {status.replace('_', ' ').title()}\n")
                    for entry in status_entries:
                        lines.append(f"- [{entry.id}] {entry.content}")
        else:
            for entry in cat_entries:
                lines.append(f"### [{entry.id}]")
                lines.append(entry.content)
                lines.append("")

    return {
        "count": len(entries),
        "context": "\n".join(lines),
    }


def cmd_session_archive(entry_id: str, category: str | None = None) -> dict[str, Any]:
    """Archive a session entry to permanent knowledge.

    Copies the session entry to knowledge.md and deletes from session.md.
    """
    entries = parse_session_file()

    for entry in entries:
        if entry.id == entry_id:
            # Determine target category
            target_category = sanitize_category(category) if category else entry.category
            # Map session categories to knowledge categories
            category_mapping = {
                "plan": "architecture",
                "todo": "todo",
                "progress": "discovery",
                "note": "discovery",
                "context": "context",
                "decision": "decision",
                "blocker": "gotcha",
            }
            if target_category in category_mapping:
                target_category = category_mapping[target_category]

            # Create knowledge memory from session entry
            memory_id = generate_memory_id(target_category, entry.content)
            now = datetime.now().isoformat()

            memory = Memory(
                id=memory_id,
                category=target_category,
                content=entry.content,
                tags=[],
                changed_at=now,
            )

            # Add to knowledge
            add_memory_to_file(memory)

            # Remove from session
            delete_session_entry(entry_id)

            return {
                "status": "success",
                "message": f"Archived to knowledge: {memory_id}",
                "archived_from": entry_id,
                "archived_to": memory_id,
                "category": target_category,
            }

    return {
        "status": "error",
        "message": f"Session entry not found: {entry_id}",
    }


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

    # Handle session entry output
    if "entry" in data:
        entry = data["entry"]
        status_str = f" ({entry['status']})" if entry.get("status") else ""
        return f"[{entry['category']}] {entry['id']}{status_str}\n{entry['content']}"

    # Handle session show (context without topic)
    if "context" in data and "topic" not in data:
        if data.get("context"):
            return data["context"]
        return "No session entries"

    if "message" in data:
        return data["message"]

    if "total_memories" in data and "by_category" in data:
        lines = [f"Total memories: {data['total_memories']}"]
        lines.append("\nBy category:")
        for cat, count in data["by_category"].items():
            lines.append(f"  {cat}: {count}")
        return "\n".join(lines)

    if "age_distribution" in data:
        lines = ["Memory Age Report", "=" * 30]
        lines.append(f"\nTotal memories: {data['total_memories']}")
        lines.append("\nAge distribution (by last change):")
        order = ["last_week", "last_month", "last_quarter", "last_year", "older"]
        labels = {"last_week": "< 1 week", "last_month": "< 1 month", "last_quarter": "< 3 months",
                  "last_year": "< 1 year", "older": "> 1 year"}
        for bucket in order:
            count = data["age_distribution"].get(bucket, 0)
            lines.append(f"  {labels[bucket]}: {count}")

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
Commands (Long-term Knowledge):
  add <category> <content>    Add a new memory
  search <query>              Search memories (keyword matching)
  context <topic>             Get context block for a topic
  list                        List all memories
  delete <id>                 Delete a memory
  stats                       Show statistics
  maintain                    Show age distribution

Session Commands (Temporary):
  session add <cat> <content> Add session entry (plan/todo/progress/note/blocker)
  session list                List session entries
  session show                Show full session state
  session update <id>         Update entry status/content
  session delete <id>         Delete session entry
  session clear               Clear all session entries
  session archive <id>        Move entry to permanent knowledge

Knowledge Categories: {', '.join(CATEGORIES)}
Session Categories: {', '.join(SESSION_CATEGORIES)}
Session Statuses: {', '.join(SESSION_STATUSES)}

Examples:
  memory.sh add discovery "The API uses OAuth2 with PKCE flow"
  memory.sh add gotcha "Redis pool must be closed" --tags redis
  memory.sh search "authentication"
  memory.sh context "vespa-linux server"
  memory.sh session add plan "1. Add auth 2. Add tests 3. Deploy"
  memory.sh session add todo "Implement JWT middleware" --status pending
  memory.sh session add progress "Completed auth module"
  memory.sh session add note "User prefers ES modules over CommonJS"
  memory.sh session list --status pending
  memory.sh session update <id> --status completed
  memory.sh session show
  memory.sh session archive <id> --category gotcha
  memory.sh session clear
        """
    )

    parser.add_argument("command", nargs="?", choices=["add", "search", "context", "list", "delete", "stats", "maintain", "session"],
                        help="Command to execute")
    parser.add_argument("args", nargs="*", help="Command arguments")
    parser.add_argument("--tags", "-t", help="Comma-separated tags (for add)")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Limit results")
    parser.add_argument("--category", "-c", help="Filter by category")
    parser.add_argument("--status", "-s", help="Filter by status or set status (for session)")
    parser.add_argument("--content", help="New content (for session update)")
    parser.add_argument("--output", "-o", choices=["text", "json"], default="text",
                        help="Output format")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential output")
    parser.add_argument("--version", "-v", action="version", version=f"%(prog)s {__version__}")

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
            result = cmd_search(query, limit=args.limit, category=args.category)

        elif args.command == "context":
            if not args.args:
                print("Error: context requires <topic>", file=sys.stderr)
                sys.exit(1)
            topic = " ".join(args.args)
            result = cmd_context(topic, limit=args.limit)

        elif args.command == "list":
            result = cmd_list(category=args.category, limit=args.limit)

        elif args.command == "delete":
            if not args.args:
                print("Error: delete requires <id>", file=sys.stderr)
                sys.exit(1)
            result = cmd_delete(args.args[0])

        elif args.command == "stats":
            result = cmd_stats()

        elif args.command == "maintain":
            result = cmd_maintain()

        elif args.command == "session":
            if not args.args:
                print("Error: session requires a subcommand (add/list/show/update/delete/clear/archive)", file=sys.stderr)
                sys.exit(1)

            subcmd = args.args[0]

            if subcmd == "add":
                if len(args.args) < 3:
                    print("Error: session add requires <category> <content>", file=sys.stderr)
                    sys.exit(1)
                category = args.args[1]
                content = " ".join(args.args[2:])
                result = cmd_session_add(category, content, status=args.status or "")

            elif subcmd == "list":
                result = cmd_session_list(
                    category=args.category,
                    status=args.status,
                    limit=args.limit,
                )

            elif subcmd == "show":
                result = cmd_session_show()

            elif subcmd == "update":
                if len(args.args) < 2:
                    print("Error: session update requires <id>", file=sys.stderr)
                    sys.exit(1)
                entry_id = args.args[1]
                result = cmd_session_update(
                    entry_id,
                    status=args.status,
                    content=args.content,
                )

            elif subcmd == "delete":
                if len(args.args) < 2:
                    print("Error: session delete requires <id>", file=sys.stderr)
                    sys.exit(1)
                result = cmd_session_delete(args.args[1])

            elif subcmd == "clear":
                result = cmd_session_clear()

            elif subcmd == "archive":
                if len(args.args) < 2:
                    print("Error: session archive requires <id>", file=sys.stderr)
                    sys.exit(1)
                result = cmd_session_archive(args.args[1], category=args.category)

            else:
                print(f"Error: unknown session subcommand: {subcmd}", file=sys.stderr)
                sys.exit(1)

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
