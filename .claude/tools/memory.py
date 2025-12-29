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

__version__ = "3.0.0"


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
KNOWLEDGE_FILE = PROJECT_ROOT / "knowledge.md"

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
    changed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "content": self.content,
            "tags": self.tags,
            "changed_at": self.changed_at,
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
Commands:
  add <category> <content>    Add a new memory
  search <query>              Search memories (keyword matching)
  context <topic>             Get context block for a topic
  list                        List all memories
  delete <id>                 Delete a memory
  stats                       Show statistics
  maintain                    Show age distribution

Categories: {', '.join(CATEGORIES)}

Examples:
  memory.sh add discovery "The API uses OAuth2 with PKCE flow"
  memory.sh add gotcha "Redis connection pool must be closed explicitly" --tags redis,connection
  memory.sh search "authentication"
  memory.sh search "vespa server docker"
  memory.sh context "vespa-linux server services"
  memory.sh list --category gotcha
  memory.sh maintain
        """
    )

    parser.add_argument("command", nargs="?", choices=["add", "search", "context", "list", "delete", "stats", "maintain"],
                        help="Command to execute")
    parser.add_argument("args", nargs="*", help="Command arguments")
    parser.add_argument("--tags", "-t", help="Comma-separated tags (for add)")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Limit results")
    parser.add_argument("--category", "-c", help="Filter by category")
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
