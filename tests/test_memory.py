#!/usr/bin/env python3
"""Unit tests for memory.py"""

import os
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


class TestSanitizeCategory:
    """Tests for sanitize_category function."""

    def test_valid_category(self):
        assert memory.sanitize_category("architecture") == "architecture"
        assert memory.sanitize_category("Discovery") == "discovery"

    def test_path_traversal_attack(self):
        assert memory.sanitize_category("../../../etc") == "etc"
        assert memory.sanitize_category("..\\..\\windows") == "windows"
        assert memory.sanitize_category("./hidden") == "hidden"

    def test_invalid_characters(self):
        assert memory.sanitize_category("cat@egory!") == "category"
        assert memory.sanitize_category("123category") == "misc"

    def test_empty_becomes_misc(self):
        assert memory.sanitize_category("...") == "misc"
        assert memory.sanitize_category("///") == "misc"


class TestTokenize:
    """Tests for tokenize function."""

    def test_simple_words(self):
        assert memory.tokenize("hello world") == ["hello", "world"]

    def test_hyphenated_words(self):
        assert memory.tokenize("vespa-linux server") == ["vespa-linux", "server"]

    def test_mixed_case(self):
        assert memory.tokenize("Hello WORLD") == ["hello", "world"]

    def test_special_characters(self):
        assert memory.tokenize("test@example.com") == ["test", "example", "com"]

    def test_numbers(self):
        assert memory.tokenize("port 8080") == ["port", "8080"]

    def test_empty_string(self):
        assert memory.tokenize("") == []


class TestCalculateMatchScore:
    """Tests for calculate_match_score function."""

    def test_single_keyword_match(self):
        mem = memory.Memory(id="test", category="discovery", content="Redis server")
        assert memory.calculate_match_score(mem, ["redis"]) == 1

    def test_multiple_keyword_matches(self):
        mem = memory.Memory(id="test", category="discovery", content="Redis server on vespa-linux")
        assert memory.calculate_match_score(mem, ["redis", "vespa"]) == 2

    def test_no_match(self):
        mem = memory.Memory(id="test", category="discovery", content="Redis server")
        assert memory.calculate_match_score(mem, ["postgres"]) == 0

    def test_tag_match(self):
        mem = memory.Memory(id="test", category="discovery", content="Server", tags=["redis", "production"])
        assert memory.calculate_match_score(mem, ["redis"]) == 1

    def test_category_match(self):
        mem = memory.Memory(id="test", category="gotcha", content="Some issue")
        assert memory.calculate_match_score(mem, ["gotcha"]) == 1

    def test_multiple_occurrences(self):
        mem = memory.Memory(id="test", category="discovery", content="Redis redis REDIS")
        score = memory.calculate_match_score(mem, ["redis"])
        assert score >= 3  # Base 1 + 2 extra occurrences


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


class TestFileOperations:
    """Tests for file operations."""

    def test_add_memory(self, temp_project):
        result = memory.cmd_add("discovery", "Test discovery", ["tag1"])

        assert result["status"] == "success"
        assert "memory" in result
        assert result["memory"]["category"] == "discovery"
        assert result["memory"]["content"] == "Test discovery"

    def test_search_memory(self, temp_project):
        # Add a memory first
        memory.cmd_add("discovery", "The API uses JWT authentication")

        # Search for it
        result = memory.cmd_search("JWT authentication")

        assert result["count"] >= 1
        assert any("JWT" in r["content"] for r in result["results"])

    def test_search_partial_match(self, temp_project):
        memory.cmd_add("discovery", "vespa-linux server configuration")

        result = memory.cmd_search("vespa")
        assert result["count"] >= 1

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
        # Add then delete
        add_result = memory.cmd_add("discovery", "To be deleted")
        mem_id = add_result["memory"]["id"]

        delete_result = memory.cmd_delete(mem_id)
        assert delete_result["status"] == "success"

        # Verify deleted
        list_result = memory.cmd_list()
        assert not any(r["id"] == mem_id for r in list_result["results"])

    def test_stats(self, temp_project):
        memory.cmd_add("discovery", "Test 1")
        memory.cmd_add("gotcha", "Test 2")

        result = memory.cmd_stats()

        assert result["total_memories"] >= 2
        assert "discovery" in result["by_category"]
        assert "gotcha" in result["by_category"]


class TestMaintain:
    """Tests for maintain command."""

    def test_maintain_basic(self, temp_project):
        memory.cmd_add("discovery", "Test memory")

        result = memory.cmd_maintain()

        assert "age_distribution" in result
        assert result["total_memories"] >= 1


class TestContext:
    """Tests for context command."""

    def test_context_returns_formatted(self, temp_project):
        memory.cmd_add("discovery", "Redis runs on port 6379", ["redis"])

        result = memory.cmd_context("redis")

        assert result["count"] >= 1
        assert "Redis" in result["context"]
        assert "Relevant Knowledge" in result["context"]

    def test_context_no_results(self, temp_project):
        result = memory.cmd_context("nonexistent topic")

        assert result["count"] == 0
        assert result["context"] == ""


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
        data = {"topic": "test", "context": "Test context content", "count": 1}
        result = memory.format_output(data, "text")

        assert result == "Test context content"


class TestSanitizeErrorMessage:
    """Tests for sanitize_error_message function."""

    def test_unix_path_sanitization(self):
        error = Exception("File not found: /home/user/secret/file.txt")
        result = memory.sanitize_error_message(error)

        assert "/home/user/secret/" not in result
        assert "[path]" in result

    def test_windows_path_sanitization(self):
        error = Exception("File not found: C:\\Users\\secret\\file.txt")
        result = memory.sanitize_error_message(error)

        assert "C:\\Users\\secret\\" not in result


class TestCalculateAgeDays:
    """Tests for calculate_age_days function."""

    def test_recent_date(self):
        from datetime import datetime
        recent = datetime.now().isoformat()
        assert memory.calculate_age_days(recent) <= 1

    def test_empty_date(self):
        assert memory.calculate_age_days("") == 9999

    def test_invalid_date(self):
        assert memory.calculate_age_days("not-a-date") == 9999


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


class TestSessionFileOperations:
    """Tests for session file operations."""

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

    def test_list_session_entries(self, temp_project):
        memory.cmd_session_add("plan", "Plan 1")
        memory.cmd_session_add("todo", "Todo 1", status="pending")

        result = memory.cmd_session_list()
        assert result["count"] >= 2

    def test_list_session_by_category(self, temp_project):
        memory.cmd_session_add("plan", "Plan 1")
        memory.cmd_session_add("todo", "Todo 1", status="pending")

        result = memory.cmd_session_list(category="todo")
        assert all(r["category"] == "todo" for r in result["results"])

    def test_list_session_by_status(self, temp_project):
        memory.cmd_session_add("todo", "Todo 1", status="pending")
        memory.cmd_session_add("todo", "Todo 2", status="completed")

        result = memory.cmd_session_list(status="pending")
        assert all(r.get("status") == "pending" for r in result["results"])

    def test_update_session_status(self, temp_project):
        add_result = memory.cmd_session_add("todo", "Task", status="pending")
        entry_id = add_result["entry"]["id"]

        update_result = memory.cmd_session_update(entry_id, status="completed")
        assert update_result["status"] == "success"
        assert update_result["entry"]["status"] == "completed"

    def test_update_session_content(self, temp_project):
        add_result = memory.cmd_session_add("note", "Original content")
        entry_id = add_result["entry"]["id"]

        update_result = memory.cmd_session_update(entry_id, content="Updated content")
        assert update_result["status"] == "success"
        assert update_result["entry"]["content"] == "Updated content"

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

    def test_clear_session(self, temp_project):
        memory.cmd_session_add("plan", "Plan 1")
        memory.cmd_session_add("todo", "Todo 1")

        clear_result = memory.cmd_session_clear()
        assert clear_result["status"] == "success"
        assert clear_result["cleared_count"] >= 2

        # Verify all cleared
        list_result = memory.cmd_session_list()
        assert list_result["count"] == 0

    def test_session_show(self, temp_project):
        memory.cmd_session_add("plan", "Main implementation plan")
        memory.cmd_session_add("todo", "Task 1", status="in_progress")

        result = memory.cmd_session_show()
        assert result["count"] >= 2
        assert "Current Session State" in result["context"]

    def test_session_archive(self, temp_project):
        add_result = memory.cmd_session_add("note", "Important discovery worth keeping")
        entry_id = add_result["entry"]["id"]

        archive_result = memory.cmd_session_archive(entry_id)
        assert archive_result["status"] == "success"
        assert "archived_to" in archive_result

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
        assert archive_result["category"] == "gotcha"


class TestSessionParsing:
    """Tests for session file parsing."""

    def test_parse_empty_session(self, temp_project):
        entries = memory.parse_session_file()
        assert entries == []

    def test_parse_session_with_status(self, temp_project):
        memory.cmd_session_add("todo", "Test task", status="in_progress")

        entries = memory.parse_session_file()
        assert len(entries) >= 1
        assert any(e.status == "in_progress" for e in entries)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
