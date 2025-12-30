#!/usr/bin/env python3
"""Unit tests for memory.py v5.0.0 - Lean Edition"""

import sys
import tempfile
import shutil
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / ".claude" / "tools"))

import memory


@pytest.fixture
def temp_project():
    """Create a temporary project directory for testing."""
    temp_dir = tempfile.mkdtemp()

    # Override paths
    original_project_root = memory.PROJECT_ROOT
    original_knowledge_file = memory.KNOWLEDGE_FILE
    original_session_file = memory.SESSION_FILE

    memory.PROJECT_ROOT = Path(temp_dir)
    memory.KNOWLEDGE_FILE = Path(temp_dir) / "knowledge.md"
    memory.SESSION_FILE = Path(temp_dir) / "session.md"

    yield temp_dir

    # Restore original paths
    memory.PROJECT_ROOT = original_project_root
    memory.KNOWLEDGE_FILE = original_knowledge_file
    memory.SESSION_FILE = original_session_file

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestMemoryDataclass:
    """Tests for Memory dataclass."""

    def test_to_dict(self):
        mem = memory.Memory(
            id="test-123",
            category="discovery",
            content="Test content",
            tags=["tag1", "tag2"],
            changed_at="2024-01-01T00:00:00"
        )
        result = mem.to_dict()

        assert result["id"] == "test-123"
        assert result["category"] == "discovery"
        assert result["content"] == "Test content"
        assert result["tags"] == ["tag1", "tag2"]
        assert result["changed_at"] == "2024-01-01T00:00:00"

    def test_default_tags(self):
        mem = memory.Memory(id="test", category="test", content="test")
        assert mem.tags == []


class TestSessionEntryDataclass:
    """Tests for SessionEntry dataclass."""

    def test_to_dict_with_status(self):
        entry = memory.SessionEntry(
            id="s-pla-20241230-abc1",
            category="plan",
            content="Test plan",
            status="pending",
            changed_at="2024-12-30T10:00:00"
        )
        result = entry.to_dict()

        assert result["id"] == "s-pla-20241230-abc1"
        assert result["category"] == "plan"
        assert result["content"] == "Test plan"
        assert result["status"] == "pending"
        assert result["changed_at"] == "2024-12-30T10:00:00"

    def test_to_dict_without_status(self):
        entry = memory.SessionEntry(
            id="s-not-20241230-abc1",
            category="note",
            content="General note"
        )
        result = entry.to_dict()

        assert "status" not in result


class TestGenerateMemoryId:
    """Tests for generate_memory_id function."""

    def test_id_format(self):
        mem_id = memory.generate_memory_id("architecture", "test content")
        parts = mem_id.split("-")

        assert len(parts) == 3
        assert parts[0] == "arc"  # First 3 chars of category
        assert len(parts[1]) == 14  # Timestamp YYYYMMDDHHMMSS
        assert len(parts[2]) == 6  # 6-char hash

    def test_different_content_different_hash(self):
        id1 = memory.generate_memory_id("test", "content1")
        id2 = memory.generate_memory_id("test", "content2")

        # Hash portion should differ
        assert id1.split("-")[2] != id2.split("-")[2]


class TestGenerateSessionId:
    """Tests for generate_session_id function."""

    def test_id_format(self):
        entry_id = memory.generate_session_id("plan", "test content")
        parts = entry_id.split("-")

        assert len(parts) == 4
        assert parts[0] == "s"  # Session prefix
        assert parts[1] == "pla"  # First 3 chars of category
        assert len(parts[2]) == 14  # Timestamp
        assert len(parts[3]) == 4  # 4-char hash


class TestSearchMemories:
    """Tests for search_memories function."""

    def test_single_keyword_match(self):
        memories = [memory.Memory(id="test", category="discovery", content="Redis server")]
        results = memory.search_memories("redis", memories)
        assert len(results) == 1

    def test_multiple_keyword_matches(self):
        memories = [memory.Memory(id="test", category="discovery", content="Redis server on vespa-linux")]
        results = memory.search_memories("redis vespa", memories)
        assert len(results) == 1
        assert results[0][1] >= 2.0  # Score >= 2 for multiple matches

    def test_no_match(self):
        memories = [memory.Memory(id="test", category="discovery", content="Redis server")]
        results = memory.search_memories("postgres", memories)
        assert len(results) == 0

    def test_tag_match(self):
        memories = [memory.Memory(id="test", category="discovery", content="Server", tags=["redis", "production"])]
        results = memory.search_memories("redis", memories)
        assert len(results) == 1
        assert results[0][1] >= 1.5  # Tag match weight

    def test_category_match(self):
        memories = [memory.Memory(id="test", category="gotcha", content="Some issue")]
        results = memory.search_memories("gotcha", memories)
        assert len(results) == 1
        assert results[0][1] >= 2.0  # Category match weight

    def test_stop_words_filtered(self):
        memories = [memory.Memory(id="test", category="discovery", content="The server is running")]
        # "the" and "is" are stop words - should match "server" and "running"
        results = memory.search_memories("the server", memories)
        assert len(results) == 1

    def test_word_boundary_matching(self):
        memories = [memory.Memory(id="test", category="discovery", content="catalog service")]
        # "log" should NOT match "catalog"
        results = memory.search_memories("log", memories)
        assert len(results) == 0

    def test_category_filter(self):
        memories = [
            memory.Memory(id="1", category="discovery", content="test"),
            memory.Memory(id="2", category="gotcha", content="test"),
        ]
        results = memory.search_memories("test", memories, category="gotcha")
        assert len(results) == 1
        assert results[0][0].category == "gotcha"


class TestKnowledgeCommands:
    """Tests for knowledge commands."""

    def test_add_memory(self, temp_project):
        result = memory.cmd_add("discovery", "Test discovery", ["tag1"])

        assert result["status"] == "success"
        assert "memory" in result
        assert result["memory"]["category"] == "discovery"
        assert result["memory"]["content"] == "Test discovery"

    def test_add_invalid_category_becomes_misc(self, temp_project):
        result = memory.cmd_add("invalid_category", "Test content")
        assert result["memory"]["category"] == "misc"

    def test_search_memory(self, temp_project):
        memory.cmd_add("discovery", "The API uses JWT authentication")

        result = memory.cmd_search("JWT authentication")

        assert result["count"] >= 1
        assert any("JWT" in r["content"] for r in result["results"])

    def test_search_with_limit(self, temp_project):
        for i in range(5):
            memory.cmd_add("discovery", f"Test memory {i}")

        result = memory.cmd_search("test memory", limit=3)
        assert len(result["results"]) <= 3

    def test_search_with_category(self, temp_project):
        memory.cmd_add("discovery", "Discovery test")
        memory.cmd_add("gotcha", "Gotcha test")

        result = memory.cmd_search("test", category="gotcha")
        assert all(r["category"] == "gotcha" for r in result["results"])

    def test_list_memories(self, temp_project):
        memory.cmd_add("discovery", "Discovery 1")
        memory.cmd_add("gotcha", "Gotcha 1")

        # List all
        result = memory.cmd_list()
        assert result["count"] >= 2

        # List by category
        result = memory.cmd_list(category="gotcha")
        assert all(r["category"] == "gotcha" for r in result["results"])

    def test_delete_memory(self, temp_project):
        add_result = memory.cmd_add("discovery", "To be deleted")
        mem_id = add_result["memory"]["id"]

        delete_result = memory.cmd_delete(mem_id)
        assert delete_result["status"] == "success"

        # Verify deleted
        list_result = memory.cmd_list()
        assert not any(r["id"] == mem_id for r in list_result["results"])

    def test_delete_nonexistent(self, temp_project):
        result = memory.cmd_delete("nonexistent-id")
        assert result["status"] == "error"

    def test_stats(self, temp_project):
        memory.cmd_add("discovery", "Test 1")
        memory.cmd_add("gotcha", "Test 2")

        result = memory.cmd_stats()

        assert result["total_memories"] >= 2
        assert "discovery" in result["by_category"]
        assert "gotcha" in result["by_category"]

    def test_stats_empty(self, temp_project):
        result = memory.cmd_stats()
        assert result["total_memories"] == 0


class TestContextCommand:
    """Tests for context command."""

    def test_context_returns_formatted(self, temp_project):
        memory.cmd_add("discovery", "Redis runs on port 6379", ["redis"])

        result = memory.cmd_context("redis")

        assert "context" in result
        assert "Redis" in result["context"]
        assert "Relevant context for" in result["context"]

    def test_context_no_results(self, temp_project):
        result = memory.cmd_context("nonexistent topic")
        assert result["context"] == ""


class TestSessionCommands:
    """Tests for session commands."""

    def test_add_session_entry(self, temp_project):
        result = memory.cmd_session_add("plan", "Implementation plan", status="")

        assert result["status"] == "success"
        assert "entry" in result
        assert result["entry"]["category"] == "plan"
        assert result["entry"]["content"] == "Implementation plan"

    def test_add_session_todo_with_status(self, temp_project):
        result = memory.cmd_session_add("todo", "Write tests", status="pending")

        assert result["status"] == "success"
        assert result["entry"]["status"] == "pending"

    def test_add_session_invalid_status(self, temp_project):
        result = memory.cmd_session_add("todo", "Task", status="invalid_status")

        assert result["status"] == "error"
        assert "Invalid status" in result["message"]

    def test_add_session_invalid_category_becomes_note(self, temp_project):
        result = memory.cmd_session_add("invalid", "Test")
        assert result["entry"]["category"] == "note"

    def test_list_session_entries(self, temp_project):
        memory.cmd_session_add("plan", "Plan 1")
        memory.cmd_session_add("todo", "Todo 1", status="pending")

        result = memory.cmd_session_list()
        assert result["count"] >= 2

    def test_list_session_by_status(self, temp_project):
        memory.cmd_session_add("todo", "Todo 1", status="pending")
        memory.cmd_session_add("todo", "Todo 2", status="completed")

        result = memory.cmd_session_list(status="pending")
        assert all(r.get("status") == "pending" for r in result["results"])

    def test_session_show(self, temp_project):
        memory.cmd_session_add("plan", "Main implementation plan")
        memory.cmd_session_add("todo", "Task 1", status="in_progress")

        result = memory.cmd_session_show()
        assert "context" in result
        assert "Current Session" in result["context"]

    def test_session_show_empty(self, temp_project):
        result = memory.cmd_session_show()
        assert result["context"] == ""

    def test_update_session_status(self, temp_project):
        add_result = memory.cmd_session_add("todo", "Task", status="pending")
        entry_id = add_result["entry"]["id"]

        update_result = memory.cmd_session_update(entry_id, status="completed")
        assert update_result["status"] == "success"
        assert update_result["entry"]["status"] == "completed"

    def test_update_session_invalid_status(self, temp_project):
        add_result = memory.cmd_session_add("todo", "Task", status="pending")
        entry_id = add_result["entry"]["id"]

        result = memory.cmd_session_update(entry_id, status="invalid")
        assert result["status"] == "error"

    def test_update_nonexistent_entry(self, temp_project):
        result = memory.cmd_session_update("nonexistent-id", status="completed")
        assert result["status"] == "error"

    def test_delete_session_entry(self, temp_project):
        add_result = memory.cmd_session_add("note", "To delete")
        entry_id = add_result["entry"]["id"]

        delete_result = memory.cmd_session_delete(entry_id)
        assert delete_result["status"] == "success"

        # Verify deleted
        list_result = memory.cmd_session_list()
        assert not any(r["id"] == entry_id for r in list_result["results"])

    def test_delete_nonexistent_entry(self, temp_project):
        result = memory.cmd_session_delete("nonexistent-id")
        assert result["status"] == "error"

    def test_clear_session(self, temp_project):
        memory.cmd_session_add("plan", "Plan 1")
        memory.cmd_session_add("todo", "Todo 1")

        clear_result = memory.cmd_session_clear()
        assert clear_result["status"] == "success"

        # Verify all cleared
        list_result = memory.cmd_session_list()
        assert list_result["count"] == 0

    def test_session_archive(self, temp_project):
        add_result = memory.cmd_session_add("note", "Important discovery worth keeping")
        entry_id = add_result["entry"]["id"]

        archive_result = memory.cmd_session_archive(entry_id)
        assert archive_result["status"] == "success"
        assert "memory" in archive_result

        # Verify removed from session
        list_result = memory.cmd_session_list()
        assert not any(r["id"] == entry_id for r in list_result["results"])

        # Verify added to knowledge
        knowledge_result = memory.cmd_list()
        assert knowledge_result["count"] >= 1

    def test_session_archive_with_category(self, temp_project):
        add_result = memory.cmd_session_add("note", "This is actually a gotcha")
        entry_id = add_result["entry"]["id"]

        archive_result = memory.cmd_session_archive(entry_id, category="gotcha")
        assert archive_result["status"] == "success"
        assert archive_result["memory"]["category"] == "gotcha"

    def test_session_archive_nonexistent(self, temp_project):
        result = memory.cmd_session_archive("nonexistent-id")
        assert result["status"] == "error"


class TestFileParsing:
    """Tests for file parsing."""

    def test_parse_empty_knowledge(self, temp_project):
        memories = memory.parse_knowledge_file()
        assert memories == []

    def test_parse_empty_session(self, temp_project):
        entries = memory.parse_session_file()
        assert entries == []

    def test_parse_session_with_status(self, temp_project):
        memory.cmd_session_add("todo", "Test task", status="in_progress")

        entries = memory.parse_session_file()
        assert len(entries) >= 1
        assert any(e.status == "in_progress" for e in entries)

    def test_parse_knowledge_with_tags(self, temp_project):
        memory.cmd_add("discovery", "Test", ["tag1", "tag2"])

        memories = memory.parse_knowledge_file()
        assert len(memories) >= 1
        assert any(m.tags == ["tag1", "tag2"] for m in memories)


class TestFormatOutput:
    """Tests for format_output function."""

    def test_json_output(self):
        data = {"key": "value", "count": 42}
        result = memory.format_output(data, "json")

        assert '"key": "value"' in result
        assert '"count": 42' in result

    def test_text_output_message(self):
        data = {"message": "Operation successful"}
        result = memory.format_output(data, "text")

        assert result == "Operation successful"

    def test_text_output_context(self):
        data = {"topic": "test", "context": "Test context content"}
        result = memory.format_output(data, "text")

        assert result == "Test context content"

    def test_text_output_stats(self):
        data = {"total_memories": 5, "by_category": {"discovery": 3, "gotcha": 2}}
        result = memory.format_output(data, "text")

        assert "Total memories: 5" in result
        assert "discovery: 3" in result

    def test_text_output_memory(self):
        data = {
            "status": "success",
            "message": "Memory added",
            "memory": {"id": "test", "category": "discovery", "content": "Test content"}
        }
        result = memory.format_output(data, "text")

        assert "[discovery]" in result
        assert "Test content" in result


class TestConstants:
    """Tests for configuration constants."""

    def test_categories_defined(self):
        assert "architecture" in memory.CATEGORIES
        assert "discovery" in memory.CATEGORIES
        assert "gotcha" in memory.CATEGORIES
        assert len(memory.CATEGORIES) == 10

    def test_session_categories_defined(self):
        assert "plan" in memory.SESSION_CATEGORIES
        assert "todo" in memory.SESSION_CATEGORIES
        assert len(memory.SESSION_CATEGORIES) == 7

    def test_session_statuses_defined(self):
        assert "pending" in memory.SESSION_STATUSES
        assert "in_progress" in memory.SESSION_STATUSES
        assert "completed" in memory.SESSION_STATUSES
        assert "blocked" in memory.SESSION_STATUSES

    def test_stop_words_defined(self):
        assert "the" in memory.STOP_WORDS
        assert "and" in memory.STOP_WORDS
        assert len(memory.STOP_WORDS) == 21  # Actual count in v5.0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
