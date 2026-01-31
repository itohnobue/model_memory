#!/usr/bin/env python3
"""
Model Memory Tool v5.0.0 - Lean Edition

A simple, focused persistent memory system for Claude Code.
Stripped down from v4.0 bloat to essential functionality only.

Features:
- Long-term knowledge storage in knowledge.md
- Session memory in session.md
- Simple keyword search with word boundaries
- 13 commands total (down from 26)

Usage:
    memory.sh add <category> <content> [--tags tag1,tag2]
    memory.sh search <query> [--category cat] [--limit n]
    memory.sh context <topic>
    memory.sh list [--category cat]
    memory.sh delete <id>
    memory.sh stats
    memory.sh session add <category> <content> [--status status]
    memory.sh session list [--status status]
    memory.sh session show
    memory.sh session update <id> --status <status>
    memory.sh session delete <id>
    memory.sh session clear
    memory.sh session archive <id> [--category cat]
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

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

__version__ = "5.0.0"


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
KNOWLEDGE_FILE = PROJECT_ROOT / "knowledge.md"
SESSION_FILE = PROJECT_ROOT / "session.md"

# Categories for knowledge
CATEGORIES = [
    "architecture", "discovery", "pattern", "gotcha", "config",
    "entity", "decision", "todo", "reference", "context",
]

# Session categories
SESSION_CATEGORIES = ["plan", "todo", "progress", "note", "context", "decision", "blocker"]

# Session statuses
SESSION_STATUSES = ["pending", "in_progress", "completed", "blocked"]

# Common stop words to filter from searches
STOP_WORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "it", "this", "that", "be", "are", "was",
])


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
    """A single session entry."""
    id: str
    category: str
    content: str
    status: str = ""
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
# FILE OPERATIONS
# =============================================================================

def parse_knowledge_file() -> list[Memory]:
    """Parse knowledge.md into memory entries."""
    if not KNOWLEDGE_FILE.exists():
        return []

    content = KNOWLEDGE_FILE.read_text(encoding="utf-8")
    pattern = r'^## \[([^\]]+)\](.*?)(?=^## \[|\Z)'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

    memories = []
    for memory_id, body in matches:
        body = body.strip()
        if not body:
            continue

        # Extract metadata
        category = "misc"
        tags = []
        changed_at = ""

        cat_match = re.search(r'^Category:\s*(.+)$', body, re.MULTILINE)
        if cat_match:
            category = cat_match.group(1).strip().lower()
            body = re.sub(r'^Category:\s*.+\n?', '', body, flags=re.MULTILINE)

        tag_match = re.search(r'^Tags:\s*(.+)$', body, re.MULTILINE)
        if tag_match:
            tags = [t.strip() for t in tag_match.group(1).split(',') if t.strip()]
            body = re.sub(r'^Tags:\s*.+\n?', '', body, flags=re.MULTILINE)

        changed_match = re.search(r'^Changed:\s*(.+)$', body, re.MULTILINE)
        if changed_match:
            changed_at = changed_match.group(1).strip()
            body = re.sub(r'^Changed:\s*.+\n?', '', body, flags=re.MULTILINE)

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
    """Write all memories to knowledge.md."""
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


def parse_session_file() -> list[SessionEntry]:
    """Parse session.md into session entries."""
    if not SESSION_FILE.exists():
        return []

    content = SESSION_FILE.read_text(encoding="utf-8")
    pattern = r'^## \[([^\]]+)\](.*?)(?=^## \[|\Z)'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

    entries = []
    for entry_id, body in matches:
        body = body.strip()
        if not body:
            continue

        category = "note"
        status = ""
        changed_at = ""

        cat_match = re.search(r'^Category:\s*(.+)$', body, re.MULTILINE)
        if cat_match:
            category = cat_match.group(1).strip().lower()
            body = re.sub(r'^Category:\s*.+\n?', '', body, flags=re.MULTILINE)

        status_match = re.search(r'^Status:\s*(.+)$', body, re.MULTILINE)
        if status_match:
            status = status_match.group(1).strip().lower()
            body = re.sub(r'^Status:\s*.+\n?', '', body, flags=re.MULTILINE)

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
    """Write all session entries to session.md."""
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


# =============================================================================
# SEARCH
# =============================================================================

def search_memories(
    query: str,
    memories: list[Memory],
    limit: int = 10,
    category: str | None = None,
) -> list[tuple[Memory, float]]:
    """Simple keyword search with word boundaries.

    Scoring: category match = 2x, tag match = 1.5x, content match = 1x
    """
    # Tokenize query, filter stop words
    keywords = [w.lower() for w in re.findall(r'\w+', query) if w.lower() not in STOP_WORDS]
    if not keywords:
        return []

    # Filter by category if specified
    if category:
        memories = [m for m in memories if m.category == category.lower()]

    scored = []
    for memory in memories:
        searchable = f"{memory.content} {memory.category} {' '.join(memory.tags)}".lower()
        score = 0.0

        for kw in keywords:
            pattern = rf'\b{re.escape(kw)}\b'
            if re.search(pattern, memory.category.lower()):
                score += 2.0
            if any(re.search(pattern, tag.lower()) for tag in memory.tags):
                score += 1.5
            if re.search(pattern, memory.content.lower()):
                score += 1.0

        if score > 0:
            scored.append((memory, score))

    # Sort by score desc, then recency desc
    scored.sort(key=lambda x: (x[1], x[0].changed_at or ""), reverse=True)
    return scored[:limit]


# =============================================================================
# ID GENERATION
# =============================================================================

def generate_memory_id(category: str, content: str) -> str:
    """Generate unique memory ID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    content_hash = hashlib.md5(content.encode()).hexdigest()[:6]
    return f"{category[:3]}-{timestamp}-{content_hash}"


def generate_session_id(category: str, content: str) -> str:
    """Generate unique session entry ID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    content_hash = hashlib.md5(content.encode()).hexdigest()[:4]
    return f"s-{category[:3]}-{timestamp}-{content_hash}"


# =============================================================================
# COMMANDS - KNOWLEDGE
# =============================================================================

def cmd_add(category: str, content: str, tags: list[str] | None = None) -> dict[str, Any]:
    """Add a new memory."""
    # Validate category
    cat_lower = category.lower()
    if cat_lower not in CATEGORIES:
        cat_lower = "misc"

    memory_id = generate_memory_id(cat_lower, content)
    now = datetime.now().isoformat()

    memory = Memory(
        id=memory_id,
        category=cat_lower,
        content=content,
        tags=tags or [],
        changed_at=now,
    )

    # Add to file
    memories = parse_knowledge_file()
    memories.append(memory)
    write_knowledge_file(memories)

    return {
        "status": "success",
        "message": f"Memory added: {memory_id}",
        "memory": memory.to_dict(),
    }


def cmd_search(query: str, limit: int = 10, category: str | None = None) -> dict[str, Any]:
    """Search memories."""
    memories = parse_knowledge_file()
    results = search_memories(query, memories, limit=limit, category=category)

    return {
        "query": query,
        "count": len(results),
        "results": [
            {
                "id": m.id,
                "category": m.category,
                "content": m.content,
                "tags": m.tags,
                "score": round(score, 2),
            }
            for m, score in results
        ],
    }


def cmd_context(topic: str, limit: int = 5) -> dict[str, Any]:
    """Get formatted context for a topic."""
    memories = parse_knowledge_file()
    results = search_memories(topic, memories, limit=limit)

    if not results:
        return {"topic": topic, "context": ""}

    lines = [f"## Relevant context for: {topic}\n"]
    for memory, _ in results:
        lines.append(f"### [{memory.category}] {memory.id}")
        if memory.tags:
            lines.append(f"Tags: {', '.join(memory.tags)}")
        lines.append(f"\n{memory.content}\n")

    return {"topic": topic, "context": "\n".join(lines)}


def cmd_list(category: str | None = None, limit: int = 50) -> dict[str, Any]:
    """List memories."""
    memories = parse_knowledge_file()

    if category:
        memories = [m for m in memories if m.category == category.lower()]

    # Sort by recency
    memories.sort(key=lambda m: m.changed_at or "", reverse=True)
    memories = memories[:limit]

    return {
        "count": len(memories),
        "category": category,
        "results": [m.to_dict() for m in memories],
    }


def cmd_delete(memory_id: str) -> dict[str, Any]:
    """Delete a memory."""
    memories = parse_knowledge_file()
    original_count = len(memories)
    memories = [m for m in memories if m.id != memory_id]

    if len(memories) < original_count:
        write_knowledge_file(memories)
        return {"status": "success", "message": f"Deleted: {memory_id}"}

    return {"status": "error", "message": f"Not found: {memory_id}"}


def cmd_stats() -> dict[str, Any]:
    """Show statistics."""
    memories = parse_knowledge_file()

    by_category: dict[str, int] = {}
    for m in memories:
        by_category[m.category] = by_category.get(m.category, 0) + 1

    return {
        "total_memories": len(memories),
        "by_category": dict(sorted(by_category.items(), key=lambda x: x[1], reverse=True)),
    }


# =============================================================================
# COMMANDS - SESSION
# =============================================================================

def cmd_session_add(category: str, content: str, status: str = "") -> dict[str, Any]:
    """Add a session entry."""
    cat_lower = category.lower()
    if cat_lower not in SESSION_CATEGORIES:
        cat_lower = "note"

    if status and status.lower() not in SESSION_STATUSES:
        return {"status": "error", "message": f"Invalid status. Valid: {', '.join(SESSION_STATUSES)}"}

    entry_id = generate_session_id(cat_lower, content)
    now = datetime.now().isoformat()

    entry = SessionEntry(
        id=entry_id,
        category=cat_lower,
        content=content,
        status=status.lower() if status else "",
        changed_at=now,
    )

    entries = parse_session_file()
    entries.append(entry)
    write_session_file(entries)

    return {
        "status": "success",
        "message": f"Session entry added: {entry_id}",
        "entry": entry.to_dict(),
    }


def cmd_session_list(status: str | None = None, limit: int = 50) -> dict[str, Any]:
    """List session entries."""
    entries = parse_session_file()

    if status:
        entries = [e for e in entries if e.status == status.lower()]

    entries.sort(key=lambda e: e.changed_at or "", reverse=True)
    entries = entries[:limit]

    return {
        "count": len(entries),
        "status_filter": status,
        "results": [e.to_dict() for e in entries],
    }


def cmd_session_show() -> dict[str, Any]:
    """Show full session state."""
    entries = parse_session_file()

    if not entries:
        return {"context": ""}

    lines = ["## Current Session\n"]

    # Group by category
    by_category: dict[str, list[SessionEntry]] = {}
    for entry in entries:
        by_category.setdefault(entry.category, []).append(entry)

    for cat, cat_entries in by_category.items():
        lines.append(f"### {cat.upper()}")
        for entry in cat_entries:
            status_str = f" [{entry.status}]" if entry.status else ""
            lines.append(f"- {entry.id}{status_str}: {entry.content[:80]}")
        lines.append("")

    return {"context": "\n".join(lines)}


def cmd_session_update(entry_id: str, status: str) -> dict[str, Any]:
    """Update session entry status."""
    if status.lower() not in SESSION_STATUSES:
        return {"status": "error", "message": f"Invalid status. Valid: {', '.join(SESSION_STATUSES)}"}

    entries = parse_session_file()

    for entry in entries:
        if entry.id == entry_id:
            entry.status = status.lower()
            entry.changed_at = datetime.now().isoformat()
            write_session_file(entries)
            return {
                "status": "success",
                "message": f"Updated: {entry_id}",
                "entry": entry.to_dict(),
            }

    return {"status": "error", "message": f"Not found: {entry_id}"}


def cmd_session_delete(entry_id: str) -> dict[str, Any]:
    """Delete a session entry."""
    entries = parse_session_file()
    original_count = len(entries)
    entries = [e for e in entries if e.id != entry_id]

    if len(entries) < original_count:
        write_session_file(entries)
        return {"status": "success", "message": f"Deleted: {entry_id}"}

    return {"status": "error", "message": f"Not found: {entry_id}"}


def cmd_session_clear() -> dict[str, Any]:
    """Clear all session entries."""
    entries = parse_session_file()
    count = len(entries)

    if SESSION_FILE.exists():
        SESSION_FILE.unlink()

    return {"status": "success", "message": f"Cleared {count} entries"}


def cmd_session_archive(entry_id: str, category: str | None = None) -> dict[str, Any]:
    """Archive a session entry to knowledge."""
    entries = parse_session_file()

    for entry in entries:
        if entry.id == entry_id:
            # Determine target category
            target_cat = category.lower() if category else entry.category
            if target_cat not in CATEGORIES:
                target_cat = "discovery"

            # Create memory
            memory_id = generate_memory_id(target_cat, entry.content)
            now = datetime.now().isoformat()

            memory = Memory(
                id=memory_id,
                category=target_cat,
                content=entry.content,
                tags=[],
                changed_at=now,
            )

            # Add to knowledge
            memories = parse_knowledge_file()
            memories.append(memory)
            write_knowledge_file(memories)

            # Remove from session
            entries = [e for e in entries if e.id != entry_id]
            write_session_file(entries)

            return {
                "status": "success",
                "message": f"Archived: {entry_id} -> {memory_id}",
                "memory": memory.to_dict(),
            }

    return {"status": "error", "message": f"Not found: {entry_id}"}


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_output(data: dict[str, Any], fmt: str = "text") -> str:
    """Format output for display."""
    if fmt == "json":
        return json.dumps(data, indent=2, ensure_ascii=False)

    # Context output
    if "context" in data:
        return data["context"] if data["context"] else "No context found"

    # Search results
    if "results" in data and "query" in data:
        lines = [f'Search: "{data["query"]}" ({data["count"]} results)\n']
        for r in data["results"]:
            lines.append(f"[{r['category']}] {r['id']}")
            if r.get("tags"):
                lines.append(f"  Tags: {', '.join(r['tags'])}")
            content = r.get("content", "")[:200]
            for line in content.split("\n"):
                lines.append(f"  {line}")
            lines.append("")
        return "\n".join(lines)

    # List results
    if "results" in data:
        lines = [f"Count: {data['count']}\n"]
        for r in data["results"]:
            status = f" [{r['status']}]" if r.get("status") else ""
            lines.append(f"[{r['category']}] {r['id']}{status}")
            content = r.get("content", "")[:100]
            lines.append(f"  {content}")
            lines.append("")
        return "\n".join(lines)

    # Stats
    if "total_memories" in data:
        lines = [f"Total memories: {data['total_memories']}\n"]
        lines.append("By category:")
        for cat, count in data["by_category"].items():
            lines.append(f"  {cat}: {count}")
        return "\n".join(lines)

    # Single memory/entry
    if "memory" in data:
        m = data["memory"]
        return f"Added [{m['category']}] {m['id']}\n{m['content']}"

    if "entry" in data:
        e = data["entry"]
        status = f" ({e['status']})" if e.get("status") else ""
        return f"[{e['category']}] {e['id']}{status}\n{e['content']}"

    # Message
    if "message" in data:
        return data["message"]

    return json.dumps(data, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model Memory Tool v5.0 - Lean Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Commands:
  add <category> <content>    Add a memory
  search <query>              Search memories
  context <topic>             Get context for topic
  list                        List memories
  delete <id>                 Delete a memory
  stats                       Show statistics

Session Commands:
  session add <cat> <content> Add session entry
  session list                List session entries
  session show                Show full session state
  session update <id>         Update entry status
  session delete <id>         Delete session entry
  session clear               Clear all session entries
  session archive <id>        Move to knowledge

Categories: {', '.join(CATEGORIES)}
Session Categories: {', '.join(SESSION_CATEGORIES)}
Session Statuses: {', '.join(SESSION_STATUSES)}
        """
    )

    parser.add_argument("command", nargs="?",
                        choices=["add", "search", "context", "list", "delete", "stats", "session"],
                        help="Command to execute")
    parser.add_argument("args", nargs="*", help="Command arguments")
    parser.add_argument("--tags", "-t", help="Comma-separated tags")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Limit results")
    parser.add_argument("--category", "-c", help="Filter by category")
    parser.add_argument("--status", "-s", help="Filter/set status")
    parser.add_argument("--output", "-o", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    parser.add_argument("--version", "-v", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        result = None

        if args.command == "add":
            if len(args.args) < 2:
                print("Error: add requires <category> <content>", file=sys.stderr)
                sys.exit(1)
            tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
            result = cmd_add(args.args[0], " ".join(args.args[1:]), tags)

        elif args.command == "search":
            if not args.args:
                print("Error: search requires <query>", file=sys.stderr)
                sys.exit(1)
            result = cmd_search(" ".join(args.args), limit=args.limit, category=args.category)

        elif args.command == "context":
            if not args.args:
                print("Error: context requires <topic>", file=sys.stderr)
                sys.exit(1)
            result = cmd_context(" ".join(args.args), limit=args.limit)

        elif args.command == "list":
            result = cmd_list(category=args.category, limit=args.limit)

        elif args.command == "delete":
            if not args.args:
                print("Error: delete requires <id>", file=sys.stderr)
                sys.exit(1)
            result = cmd_delete(args.args[0])

        elif args.command == "stats":
            result = cmd_stats()

        elif args.command == "session":
            if not args.args:
                print("Error: session requires subcommand", file=sys.stderr)
                sys.exit(1)

            subcmd = args.args[0]

            if subcmd == "add":
                if len(args.args) < 3:
                    print("Error: session add requires <category> <content>", file=sys.stderr)
                    sys.exit(1)
                result = cmd_session_add(args.args[1], " ".join(args.args[2:]), status=args.status or "")

            elif subcmd == "list":
                result = cmd_session_list(status=args.status, limit=args.limit)

            elif subcmd == "show":
                result = cmd_session_show()

            elif subcmd == "update":
                if len(args.args) < 2 or not args.status:
                    print("Error: session update requires <id> --status <status>", file=sys.stderr)
                    sys.exit(1)
                result = cmd_session_update(args.args[1], args.status)

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

        if result is None:
            print("Error: no result", file=sys.stderr)
            sys.exit(1)

        is_error = result.get("status") == "error"

        if not args.quiet:
            output = format_output(result, args.output)
            print(output, file=sys.stderr if is_error else sys.stdout)
        elif args.output == "json":
            print(format_output(result, "json"))

        if is_error:
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
