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
    original_db_path = memory.DB_PATH

    memory.PROJECT_ROOT = Path(temp_dir)
    memory.KNOWLEDGE_FILE = Path(temp_dir) / "knowledge.md"
    memory.DB_PATH = Path(temp_dir) / "memory.db"

    yield temp_dir

    # Restore original paths
    memory.PROJECT_ROOT = original_project_root
    memory.KNOWLEDGE_FILE = original_knowledge_file
    memory.DB_PATH = original_db_path

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


class TestEscapeFts5Query:
    """Tests for escape_fts5_query function."""

    def test_simple_query(self):
        assert memory.escape_fts5_query("hello") == '"hello"'

    def test_query_with_quotes(self):
        assert memory.escape_fts5_query('say "hello"') == '"say ""hello"""'

    def test_query_with_special_chars(self):
        assert memory.escape_fts5_query("test*query") == '"test*query"'
        assert memory.escape_fts5_query("foo:bar") == '"foo:bar"'


class TestBuildOrQuery:
    """Tests for build_or_query function."""

    def test_single_keyword(self):
        result = memory.build_or_query("hello")
        assert '"hello"' in result
        assert '"hello"*' in result
        assert "OR" in result

    def test_multiple_keywords(self):
        result = memory.build_or_query("vespa linux docker")
        assert '"vespa"' in result
        assert '"linux"' in result
        assert '"docker"' in result
        assert result.count("OR") >= 5  # Each keyword has exact + prefix

    def test_quoted_phrase_passthrough(self):
        result = memory.build_or_query('"exact phrase"')
        assert result == '"exact phrase"'

    def test_empty_query(self):
        result = memory.build_or_query("")
        assert result == '""'

    def test_whitespace_handling(self):
        result = memory.build_or_query("  foo   bar  ")
        assert '"foo"' in result
        assert '"bar"' in result

    def test_special_chars_escaped(self):
        result = memory.build_or_query('say "hello"')
        assert '""hello""' in result  # Quotes escaped


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


class TestDatabaseOperations:
    """Tests for database operations."""

    def test_init_database(self, temp_project):
        memory.init_database()
        assert memory.DB_PATH.exists()

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
        assert "db_integrity" in result
        assert result["db_integrity"] == "ok"
        assert result["index_synced"] == True


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
