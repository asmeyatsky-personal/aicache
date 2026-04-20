"""
Comprehensive test suite for aicache project.

Tests cover: security, config, core/cache, domain/models, domain/services,
domain/ports, cache_factory, and mcp_server modules.
"""

import asyncio
import json
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aicache.cache_factory import CacheFactory, create_cache, get_cache
from aicache.config import ConfigManager
from aicache.core.cache import CoreCache
from aicache.domain.models import (
    CacheEntry,
    CacheInvalidationEvent,
    CacheMetadata,
    CacheMetrics,
    CachePolicy,
    CacheResult,
    EvictionPolicy,
    InvalidationStrategy,
    SemanticMatch,
    TokenUsageMetrics,
)
from aicache.domain.ports import (
    CacheMetricsPort,
    EmbeddingGeneratorPort,
    EventPublisherPort,
    QueryNormalizerPort,
    RepositoryPort,
    SemanticIndexPort,
    StoragePort,
    TokenCounterPort,
    TOONRepositoryPort,
)
from aicache.domain.services import (
    CacheEvictionService,
    CacheTTLService,
)
from aicache.mcp_server import MCPConnection

# ============================================================
# Module imports
# ============================================================
from aicache.security import (
    SecurityUtils,
    detect_pii,
    is_safe_prompt,
    mask_pii,
    sanitize_input,
    validate_context,
)


# ============================================================
# Helper: run async in sync context
# ============================================================
def run_async(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ############################################################
# 1. SECURITY MODULE TESTS
# ############################################################


class TestSecuritySanitizeInput:
    def test_sanitize_removes_null_bytes(self):
        su = SecurityUtils()
        result = su.sanitize_input("hello\x00world")
        assert "\x00" not in result
        assert "hello" in result

    def test_sanitize_collapses_whitespace(self):
        su = SecurityUtils()
        result = su.sanitize_input("hello    world")
        assert result == "hello world"

    def test_sanitize_removes_control_characters(self):
        su = SecurityUtils()
        result = su.sanitize_input("hello\x01\x02world")
        assert "\x01" not in result
        assert "\x02" not in result

    def test_sanitize_preserves_newlines(self):
        su = SecurityUtils()
        result = su.sanitize_input("line1\nline2")
        assert "\n" in result

    def test_sanitize_strips_leading_trailing(self):
        su = SecurityUtils()
        result = su.sanitize_input("  hello  ")
        assert result == "hello"

    def test_sanitize_empty_string(self):
        su = SecurityUtils()
        result = su.sanitize_input("")
        assert result == ""

    def test_convenience_sanitize_input(self):
        result = sanitize_input("hello\x00world")
        assert "\x00" not in result


class TestSecurityDetectPII:
    def test_detect_openai_api_key(self):
        su = SecurityUtils()
        text = "My key is sk-abcdefghijklmnopqrstuvwxyz1234"
        findings = su.detect_pii(text)
        api_findings = [f for f in findings if f["type"] == "api_key"]
        assert len(api_findings) >= 1

    def test_detect_github_token(self):
        su = SecurityUtils()
        text = "token: ghp_abcdefghijklmnopqrstuvwxyz1234567890"
        findings = su.detect_pii(text)
        api_findings = [f for f in findings if f["type"] == "api_key"]
        assert len(api_findings) >= 1

    def test_detect_sensitive_keyword_password(self):
        su = SecurityUtils()
        text = "my password is secret123"
        findings = su.detect_pii(text)
        password_findings = [f for f in findings if f["type"] == "password"]
        assert len(password_findings) >= 1

    def test_detect_empty_text(self):
        su = SecurityUtils()
        findings = su.detect_pii("")
        assert findings == []

    def test_detect_no_pii(self):
        su = SecurityUtils()
        text = "The sky is blue and the grass is green."
        findings = su.detect_pii(text)
        # No API keys detected
        api_findings = [f for f in findings if f["type"] == "api_key"]
        assert len(api_findings) == 0

    def test_convenience_detect_pii(self):
        findings = detect_pii("my password is xyz")
        password_findings = [f for f in findings if f["type"] == "password"]
        assert len(password_findings) >= 1


class TestSecurityMaskPII:
    def test_mask_api_key(self):
        su = SecurityUtils()
        text = "Key: sk-abcdefghijklmnopqrstuvwxyz1234"
        masked = su.mask_pii(text)
        assert "sk-a" in masked  # API keys keep first 4 chars
        assert "abcdefghij" not in masked

    def test_mask_sensitive_keyword(self):
        su = SecurityUtils()
        text = "my password"
        masked = su.mask_pii(text)
        assert "password" not in masked
        assert "***" in masked or "****" in masked

    def test_mask_no_pii(self):
        su = SecurityUtils()
        text = "just a normal sentence"
        masked = su.mask_pii(text)
        assert masked == text

    def test_convenience_mask_pii(self):
        masked = mask_pii("my password is secret")
        assert "password" not in masked


class TestSecurityIsSafePrompt:
    def test_safe_normal_prompt(self):
        su = SecurityUtils()
        assert su.is_safe_prompt("What is the capital of France?") is True

    def test_unsafe_xss_script(self):
        su = SecurityUtils()
        assert su.is_safe_prompt('<script>alert("xss")</script>') is False

    def test_unsafe_eval(self):
        su = SecurityUtils()
        assert su.is_safe_prompt('eval("malicious code")') is False

    def test_unsafe_import_os(self):
        su = SecurityUtils()
        assert su.is_safe_prompt("import os; os.system('rm -rf /')") is False

    def test_unsafe_dunder_import(self):
        su = SecurityUtils()
        assert su.is_safe_prompt("__import__('os')") is False

    def test_empty_prompt_unsafe(self):
        su = SecurityUtils()
        assert su.is_safe_prompt("") is False

    def test_very_long_prompt_unsafe(self):
        su = SecurityUtils()
        assert su.is_safe_prompt("a" * 200000) is False

    def test_convenience_is_safe_prompt(self):
        assert is_safe_prompt("Hello world") is True
        assert is_safe_prompt("eval('bad')") is False


class TestSecurityValidateContext:
    def test_validate_none_returns_empty(self):
        su = SecurityUtils()
        result = su.validate_context(None)
        assert result == {}

    def test_validate_non_dict_returns_empty(self):
        su = SecurityUtils()
        result = su.validate_context("not a dict")
        assert result == {}

    def test_validate_redacts_sensitive_keys(self):
        su = SecurityUtils()
        # The code checks if the literal pattern string is a substring of key.
        # Pattern "password" is a substring of key "password_db".
        context = {"password_db": "secret123", "name": "test"}
        result = su.validate_context(context)
        assert result["password_db"] == "[REDACTED]"
        assert result["name"] == "test"

    def test_validate_passes_safe_values(self):
        su = SecurityUtils()
        context = {"model": "gpt-4", "count": 42, "active": True, "nothing": None}
        result = su.validate_context(context)
        assert result["model"] == "gpt-4"
        assert result["count"] == 42
        assert result["active"] is True
        assert result["nothing"] is None

    def test_validate_nested_dict(self):
        su = SecurityUtils()
        context = {"settings": {"name": "test"}}
        result = su.validate_context(context)
        assert result["settings"]["name"] == "test"

    def test_validate_list_filters_bad_types(self):
        su = SecurityUtils()
        context = {"items": ["a", 1, True, {"nested": "bad"}]}
        result = su.validate_context(context)
        # dicts within lists should be filtered out
        assert {"nested": "bad"} not in result["items"]
        assert "a" in result["items"]

    def test_convenience_validate_context(self):
        # Pattern "password" is a literal substring match against key names
        result = validate_context({"password": "secret"})
        assert result["password"] == "[REDACTED]"


class TestSecurityUtilsCustomPatterns:
    def test_custom_patterns_from_config(self):
        su = SecurityUtils(config={"sensitive_patterns": [("my_custom_secret", "CUSTOM_SECRET")]})
        findings = su.detect_pii("this has my_custom_secret inside")
        custom_findings = [f for f in findings if f["type"] == "custom_secret"]
        assert len(custom_findings) >= 1

    def test_custom_string_pattern(self):
        su = SecurityUtils(config={"sensitive_patterns": ["special_value"]})
        findings = su.detect_pii("this has special_value in it")
        custom_findings = [f for f in findings if f["type"] == "custom"]
        assert len(custom_findings) >= 1


# ############################################################
# 2. CONFIG MODULE TESTS
# ############################################################


class TestConfigManager:
    def _make_manager(self, yaml_content=None):
        """Create a ConfigManager with a temp config directory."""
        tmpdir = tempfile.mkdtemp()
        config_path = Path(tmpdir) / "config.yaml"
        if yaml_content is not None:
            config_path.write_text(yaml_content)
        mgr = ConfigManager(config_path=config_path)
        return mgr, tmpdir

    def test_default_config_values(self):
        mgr, tmpdir = self._make_manager()
        try:
            assert mgr.get("ttl") == 0
            assert mgr.get("cache_size_limit") == 0
            assert mgr.get("semantic_cache.enabled") is True
            assert mgr.get("semantic_cache.backend") == "chromadb"
        finally:
            shutil.rmtree(tmpdir)

    def test_get_returns_full_config_when_no_key(self):
        mgr, tmpdir = self._make_manager()
        try:
            config = mgr.get()
            assert isinstance(config, dict)
            assert "cache_dir" in config
        finally:
            shutil.rmtree(tmpdir)

    def test_get_dot_notation(self):
        mgr, tmpdir = self._make_manager()
        try:
            result = mgr.get("semantic_cache.similarity_threshold")
            assert result == 0.85
        finally:
            shutil.rmtree(tmpdir)

    def test_get_default_for_missing_key(self):
        mgr, tmpdir = self._make_manager()
        try:
            result = mgr.get("nonexistent.key", default="fallback")
            assert result == "fallback"
        finally:
            shutil.rmtree(tmpdir)

    def test_set_and_retrieve(self):
        mgr, tmpdir = self._make_manager()
        try:
            mgr.set("ttl", 3600, persist=False)
            assert mgr.get("ttl") == 3600
        finally:
            shutil.rmtree(tmpdir)

    def test_set_nested_key(self):
        mgr, tmpdir = self._make_manager()
        try:
            mgr.set("semantic_cache.similarity_threshold", 0.95, persist=False)
            assert mgr.get("semantic_cache.similarity_threshold") == 0.95
        finally:
            shutil.rmtree(tmpdir)

    def test_load_yaml_config(self):
        yaml_content = "ttl: 7200\ncache_dir: /tmp/test\n"
        mgr, tmpdir = self._make_manager(yaml_content=yaml_content)
        try:
            assert mgr.get("ttl") == 7200
            assert mgr.get("cache_dir") == "/tmp/test"
            # Defaults are still accessible for unset keys
            assert mgr.get("semantic_cache.enabled") is True
        finally:
            shutil.rmtree(tmpdir)

    def test_deep_merge(self):
        yaml_content = "semantic_cache:\n  similarity_threshold: 0.99\n"
        mgr, tmpdir = self._make_manager(yaml_content=yaml_content)
        try:
            # Overridden value
            assert mgr.get("semantic_cache.similarity_threshold") == 0.99
            # Default value preserved
            assert mgr.get("semantic_cache.backend") == "chromadb"
        finally:
            shutil.rmtree(tmpdir)

    def test_validate_config_valid(self):
        mgr, tmpdir = self._make_manager()
        try:
            result = mgr.validate_config()
            assert result["valid"] is True
            assert len(result["errors"]) == 0
        finally:
            shutil.rmtree(tmpdir)

    def test_validate_config_invalid_backend(self):
        mgr, tmpdir = self._make_manager()
        try:
            mgr.set("semantic_cache.backend", "invalid_backend", persist=False)
            result = mgr.validate_config()
            assert result["valid"] is False
            assert any("backend" in e for e in result["errors"])
        finally:
            shutil.rmtree(tmpdir)

    def test_get_feature_flags(self):
        mgr, tmpdir = self._make_manager()
        try:
            flags = mgr.get_feature_flags()
            assert isinstance(flags, dict)
            assert "semantic_cache" in flags
            assert flags["semantic_cache"] is True
            assert flags["analytics"] is False
        finally:
            shutil.rmtree(tmpdir)

    def test_save_and_reload(self):
        mgr, tmpdir = self._make_manager()
        try:
            mgr.set("ttl", 999, persist=True)
            # Reload
            mgr2 = ConfigManager(config_path=mgr.config_path)
            assert mgr2.get("ttl") == 999
        finally:
            shutil.rmtree(tmpdir)

    def test_export_config(self):
        mgr, tmpdir = self._make_manager()
        try:
            export_path = Path(tmpdir) / "exported.yaml"
            mgr.export_config(export_path, include_defaults=True)
            assert export_path.exists()
            import yaml

            with open(export_path) as f:
                data = yaml.safe_load(f)
            assert "cache_dir" in data
        finally:
            shutil.rmtree(tmpdir)


# ############################################################
# 3. CORE CACHE TESTS
# ############################################################


class TestCoreCachePathTraversal:
    def test_path_traversal_attack_rejected(self):
        tmpdir = tempfile.mkdtemp()
        try:
            cache = CoreCache(cache_dir=tmpdir)
            with pytest.raises(ValueError, match="path traversal"):
                cache._get_cache_file("../../etc/passwd")
        finally:
            shutil.rmtree(tmpdir)

    def test_path_traversal_with_dotdot_prefix(self):
        tmpdir = tempfile.mkdtemp()
        try:
            cache = CoreCache(cache_dir=tmpdir)
            with pytest.raises(ValueError, match="path traversal"):
                cache._get_cache_file("../../../some/file")
        finally:
            shutil.rmtree(tmpdir)

    def test_valid_cache_key_ok(self):
        tmpdir = tempfile.mkdtemp()
        try:
            cache = CoreCache(cache_dir=tmpdir)
            path = cache._get_cache_file("abc123def456")
            # Resolve tmpdir to handle macOS /var -> /private/var symlink
            resolved_tmpdir = str(Path(tmpdir).resolve())
            assert str(path).startswith(resolved_tmpdir)
        finally:
            shutil.rmtree(tmpdir)


class TestCoreCacheGetValue:
    def test_get_value_returns_string(self):
        tmpdir = tempfile.mkdtemp()
        try:
            cache = CoreCache(cache_dir=tmpdir)
            cache.set("What is 2+2?", "4")
            result = cache.get_value("What is 2+2?")
            assert isinstance(result, str)
            assert result == "4"
        finally:
            shutil.rmtree(tmpdir)

    def test_get_value_returns_none_for_missing(self):
        tmpdir = tempfile.mkdtemp()
        try:
            cache = CoreCache(cache_dir=tmpdir)
            result = cache.get_value("nonexistent prompt")
            assert result is None
        finally:
            shutil.rmtree(tmpdir)


class TestCoreCacheTTL:
    def test_expired_entry_returns_none(self):
        tmpdir = tempfile.mkdtemp()
        try:
            cache = CoreCache(cache_dir=tmpdir)
            cache.set("ttl test", "value", ttl_seconds=1)
            # Force expiration by modifying the file timestamp
            cache_key = cache._get_cache_key("ttl test")
            cache_file = cache._get_cache_file(cache_key)
            with open(cache_file, "r") as f:
                data = json.load(f)
            data["timestamp"] = time.time() - 100  # 100 seconds ago
            with open(cache_file, "w") as f:
                json.dump(data, f)
            result = cache.get_value("ttl test")
            assert result is None
        finally:
            shutil.rmtree(tmpdir)

    def test_non_expired_entry_returns_value(self):
        tmpdir = tempfile.mkdtemp()
        try:
            cache = CoreCache(cache_dir=tmpdir)
            cache.set("ttl test", "value", ttl_seconds=3600)
            result = cache.get_value("ttl test")
            assert result == "value"
        finally:
            shutil.rmtree(tmpdir)


class TestCoreCacheIndex:
    def test_index_file_created(self):
        tmpdir = tempfile.mkdtemp()
        try:
            cache = CoreCache(cache_dir=tmpdir)
            cache.set("idx test", "val")
            index_path = Path(tmpdir) / ".index.json"
            assert index_path.exists()
            with open(index_path) as f:
                index = json.load(f)
            assert len(index) == 1
        finally:
            shutil.rmtree(tmpdir)

    def test_index_updated_on_delete(self):
        tmpdir = tempfile.mkdtemp()
        try:
            cache = CoreCache(cache_dir=tmpdir)
            cache.set("idx test", "val")
            cache_key = cache._get_cache_key("idx test")
            cache.delete(cache_key)
            index_path = Path(tmpdir) / ".index.json"
            with open(index_path) as f:
                index = json.load(f)
            assert len(index) == 0
        finally:
            shutil.rmtree(tmpdir)

    def test_index_stores_prompt_preview(self):
        tmpdir = tempfile.mkdtemp()
        try:
            cache = CoreCache(cache_dir=tmpdir)
            prompt = "short prompt"
            cache.set(prompt, "val")
            cache_key = cache._get_cache_key(prompt)
            assert cache._index[cache_key]["prompt_preview"] == prompt
        finally:
            shutil.rmtree(tmpdir)


# ############################################################
# 4. DOMAIN MODELS TESTS
# ############################################################


class TestCacheResult:
    def test_create_hit(self):
        result = CacheResult.create_hit(value=b"hello", entry_key="k1", response_time_ms=1.5)
        assert result.hit is True
        assert result.value == b"hello"
        assert result.entry_key == "k1"
        assert result.response_time_ms == 1.5

    def test_create_miss(self):
        result = CacheResult.create_miss(response_time_ms=2.0)
        assert result.hit is False
        assert result.value is None
        assert result.response_time_ms == 2.0

    def test_create_semantic_hit(self):
        result = CacheResult.create_semantic_hit(
            value=b"data",
            entry_key="k2",
            similarity_score=0.92,
            confidence=0.88,
            response_time_ms=5.0,
        )
        assert result.hit is True
        assert result.similarity_score == 0.92
        assert result.confidence == 0.88
        assert result.value == b"data"

    def test_create_miss_defaults(self):
        result = CacheResult.create_miss()
        assert result.hit is False
        assert result.response_time_ms == 0.0


class TestDomainCacheEntry:
    def test_empty_key_raises_value_error(self):
        with pytest.raises(ValueError, match="key cannot be empty"):
            CacheEntry(key="", value=b"data", created_at=datetime.now())

    def test_empty_value_raises_value_error(self):
        with pytest.raises(ValueError, match="value cannot be empty"):
            CacheEntry(key="k1", value=b"", created_at=datetime.now())

    def test_expires_before_created_raises(self):
        now = datetime.now()
        with pytest.raises(ValueError, match="Expiration time must be after"):
            CacheEntry(
                key="k1",
                value=b"data",
                created_at=now,
                expires_at=now - timedelta(hours=1),
            )

    def test_is_expired_true(self):
        entry = CacheEntry(
            key="k1",
            value=b"data",
            created_at=datetime.now() - timedelta(hours=2),
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert entry.is_expired() is True

    def test_is_expired_false_no_expiry(self):
        entry = CacheEntry(
            key="k1",
            value=b"data",
            created_at=datetime.now(),
        )
        assert entry.is_expired() is False

    def test_touch_returns_new_instance(self):
        entry = CacheEntry(key="k1", value=b"data", created_at=datetime.now())
        touched = entry.touch()
        assert touched is not entry
        assert touched.metadata.accessed_count == entry.metadata.accessed_count + 1

    def test_refresh_ttl(self):
        entry = CacheEntry(
            key="k1",
            value=b"data",
            created_at=datetime.now(),
            ttl_seconds=3600,
            expires_at=datetime.now() + timedelta(seconds=100),
        )
        refreshed = entry.refresh_ttl()
        assert refreshed.expires_at > entry.expires_at

    def test_get_size_bytes(self):
        entry = CacheEntry(key="abc", value=b"1234", created_at=datetime.now())
        # size = len("abc".encode()) + len(b"1234") = 3 + 4 = 7
        assert entry.get_size_bytes() == 7

    def test_immutable_frozen(self):
        entry = CacheEntry(key="k1", value=b"data", created_at=datetime.now())
        with pytest.raises(AttributeError):
            entry.key = "new_key"


class TestCacheMetadata:
    def test_touch_returns_new_instance(self):
        meta = CacheMetadata(created_at=datetime.now(), accessed_count=0)
        touched = meta.touch()
        assert touched is not meta
        assert touched.accessed_count == 1
        assert touched.last_accessed_at is not None

    def test_defaults_initialized(self):
        meta = CacheMetadata(created_at=datetime.now())
        assert meta.semantic_tags == []
        assert meta.metadata == {}

    def test_frozen(self):
        meta = CacheMetadata(created_at=datetime.now())
        with pytest.raises(AttributeError):
            meta.accessed_count = 5


class TestTokenUsageMetrics:
    def test_negative_prompt_tokens_raises(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            TokenUsageMetrics(
                prompt_tokens=-1,
                completion_tokens=10,
                total_tokens=9,
                estimated_cost=0.01,
            )

    def test_negative_completion_tokens_raises(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            TokenUsageMetrics(
                prompt_tokens=10,
                completion_tokens=-5,
                total_tokens=5,
                estimated_cost=0.01,
            )

    def test_valid_metrics(self):
        m = TokenUsageMetrics(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost=0.005,
        )
        assert m.prompt_tokens == 100
        assert m.total_tokens == 150


class TestSemanticMatch:
    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Similarity score must be between"):
            SemanticMatch(similarity_score=1.5, matched_entry_key="k", confidence=0.8)

    def test_negative_score_raises(self):
        with pytest.raises(ValueError, match="Similarity score must be between"):
            SemanticMatch(similarity_score=-0.1, matched_entry_key="k", confidence=0.8)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Confidence must be between"):
            SemanticMatch(similarity_score=0.9, matched_entry_key="k", confidence=1.5)

    def test_valid_semantic_match(self):
        m = SemanticMatch(similarity_score=0.95, matched_entry_key="key1", confidence=0.9)
        assert m.similarity_score == 0.95
        assert m.matched_entry_key == "key1"


class TestCacheMetricsROI:
    def test_calculate_roi_no_operations(self):
        m = CacheMetrics(
            total_hits=0,
            total_misses=0,
            total_evictions=0,
            average_response_time_ms=0,
            total_tokens_saved=0,
            total_cost_saved=0,
            hit_rate=0,
            memory_usage_bytes=0,
            semantic_matches=0,
            false_positives=0,
        )
        assert m.calculate_roi() == 0.0

    def test_calculate_roi_with_data(self):
        m = CacheMetrics(
            total_hits=80,
            total_misses=20,
            total_evictions=0,
            average_response_time_ms=5,
            total_tokens_saved=1000,
            total_cost_saved=10.0,
            hit_rate=0.8,
            memory_usage_bytes=1024,
            semantic_matches=50,
            false_positives=2,
        )
        roi = m.calculate_roi()
        assert roi == pytest.approx(10.0 / 100)


class TestCachePolicy:
    def test_valid_policy(self):
        p = CachePolicy(
            max_size_bytes=1024,
            default_ttl_seconds=3600,
            eviction_policy=EvictionPolicy.LRU,
        )
        assert p.validate() is True

    def test_invalid_zero_size(self):
        p = CachePolicy(
            max_size_bytes=0,
            default_ttl_seconds=3600,
            eviction_policy=EvictionPolicy.LRU,
        )
        assert p.validate() is False

    def test_invalid_threshold(self):
        p = CachePolicy(
            max_size_bytes=1024,
            default_ttl_seconds=3600,
            eviction_policy=EvictionPolicy.LRU,
            semantic_match_threshold=0.0,
        )
        assert p.validate() is False


class TestCacheInvalidationEvent:
    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="Cache key required"):
            CacheInvalidationEvent(
                cache_key="",
                reason="test",
                triggered_by="user",
                timestamp=datetime.now(),
                strategy=InvalidationStrategy.IMMEDIATE,
            )


# ############################################################
# 5. DOMAIN SERVICES TESTS (async)
# ############################################################


def _make_entry(key, value, created_at=None, last_accessed=None, accessed_count=0):
    """Helper to create a domain CacheEntry with metadata."""
    now = datetime.now()
    if created_at is None:
        created_at = now
    meta = CacheMetadata(
        created_at=created_at,
        accessed_count=accessed_count,
        last_accessed_at=last_accessed,
    )
    return CacheEntry(
        key=key,
        value=value.encode() if isinstance(value, str) else value,
        created_at=created_at,
        metadata=meta,
    )


class TestCacheEvictionServiceLRU:
    def test_lru_evicts_least_recently_accessed(self):
        old_time = datetime.now() - timedelta(hours=3)
        mid_time = datetime.now() - timedelta(hours=2)
        new_time = datetime.now() - timedelta(hours=1)

        e1 = _make_entry("k1", "aaaa", created_at=old_time, last_accessed=old_time)
        e2 = _make_entry("k2", "bbbb", created_at=old_time, last_accessed=new_time)
        e3 = _make_entry("k3", "cccc", created_at=old_time, last_accessed=mid_time)

        storage = AsyncMock(spec=StoragePort)
        storage.get_all_keys = AsyncMock(return_value=["k1", "k2", "k3"])
        storage.get = AsyncMock(side_effect=lambda k: {"k1": e1, "k2": e2, "k3": e3}[k])
        storage.delete = AsyncMock(return_value=True)

        policy = CachePolicy(
            max_size_bytes=10,
            default_ttl_seconds=None,
            eviction_policy=EvictionPolicy.LRU,
        )
        service = CacheEvictionService(policy=policy, storage=storage)

        evicted = run_async(service.evict_if_necessary(current_size=12, new_entry_size=5))
        # Should evict k1 first (least recently accessed)
        assert "k1" in evicted


class TestCacheEvictionServiceLFU:
    def test_lfu_evicts_least_frequently_accessed(self):
        now = datetime.now()

        e1 = _make_entry("k1", "aaaa", created_at=now, accessed_count=10)
        e2 = _make_entry("k2", "bbbb", created_at=now, accessed_count=1)
        e3 = _make_entry("k3", "cccc", created_at=now, accessed_count=5)

        storage = AsyncMock(spec=StoragePort)
        storage.get_all_keys = AsyncMock(return_value=["k1", "k2", "k3"])
        storage.get = AsyncMock(side_effect=lambda k: {"k1": e1, "k2": e2, "k3": e3}[k])
        storage.delete = AsyncMock(return_value=True)

        policy = CachePolicy(
            max_size_bytes=10,
            default_ttl_seconds=None,
            eviction_policy=EvictionPolicy.LFU,
        )
        service = CacheEvictionService(policy=policy, storage=storage)

        evicted = run_async(service.evict_if_necessary(current_size=12, new_entry_size=5))
        # Should evict k2 first (lowest access count)
        assert evicted[0] == "k2"


class TestCacheEvictionServiceFIFO:
    def test_fifo_evicts_oldest_created(self):
        oldest = datetime.now() - timedelta(hours=3)
        middle = datetime.now() - timedelta(hours=2)
        newest = datetime.now() - timedelta(hours=1)

        e1 = _make_entry("k1", "aaaa", created_at=newest)
        e2 = _make_entry("k2", "bbbb", created_at=oldest)
        e3 = _make_entry("k3", "cccc", created_at=middle)

        storage = AsyncMock(spec=StoragePort)
        storage.get_all_keys = AsyncMock(return_value=["k1", "k2", "k3"])
        storage.get = AsyncMock(side_effect=lambda k: {"k1": e1, "k2": e2, "k3": e3}[k])
        storage.delete = AsyncMock(return_value=True)

        policy = CachePolicy(
            max_size_bytes=10,
            default_ttl_seconds=None,
            eviction_policy=EvictionPolicy.FIFO,
        )
        service = CacheEvictionService(policy=policy, storage=storage)

        evicted = run_async(service.evict_if_necessary(current_size=12, new_entry_size=5))
        # Should evict k2 first (oldest creation time)
        assert evicted[0] == "k2"


class TestCacheEvictionServiceNoEviction:
    def test_no_eviction_when_size_ok(self):
        storage = AsyncMock(spec=StoragePort)
        policy = CachePolicy(
            max_size_bytes=100,
            default_ttl_seconds=None,
            eviction_policy=EvictionPolicy.LRU,
        )
        service = CacheEvictionService(policy=policy, storage=storage)
        evicted = run_async(service.evict_if_necessary(current_size=10, new_entry_size=5))
        assert evicted == []


class TestCacheTTLService:
    def test_get_expiration_time_none(self):
        result = CacheTTLService.get_expiration_time(None)
        assert result is None

    def test_get_expiration_time_positive(self):
        result = CacheTTLService.get_expiration_time(3600)
        assert result is not None
        assert result > datetime.now()

    def test_should_refresh_ttl_no_ttl(self):
        entry = _make_entry("k1", "data")
        assert CacheTTLService.should_refresh_ttl(entry) is False

    def test_should_refresh_ttl_still_fresh(self):
        now = datetime.now()
        entry = CacheEntry(
            key="k1",
            value=b"data",
            created_at=now,
            ttl_seconds=3600,
            expires_at=now + timedelta(seconds=3600),
        )
        assert CacheTTLService.should_refresh_ttl(entry) is False


# ############################################################
# 6. DOMAIN PORTS TESTS
# ############################################################


class TestPortsAreAbstract:
    def test_storage_port_not_instantiable(self):
        with pytest.raises(TypeError):
            StoragePort()

    def test_semantic_index_port_not_instantiable(self):
        with pytest.raises(TypeError):
            SemanticIndexPort()

    def test_token_counter_port_not_instantiable(self):
        with pytest.raises(TypeError):
            TokenCounterPort()

    def test_event_publisher_port_not_instantiable(self):
        with pytest.raises(TypeError):
            EventPublisherPort()

    def test_query_normalizer_port_not_instantiable(self):
        with pytest.raises(TypeError):
            QueryNormalizerPort()

    def test_cache_metrics_port_not_instantiable(self):
        with pytest.raises(TypeError):
            CacheMetricsPort()

    def test_embedding_generator_port_not_instantiable(self):
        with pytest.raises(TypeError):
            EmbeddingGeneratorPort()

    def test_repository_port_not_instantiable(self):
        with pytest.raises(TypeError):
            RepositoryPort()

    def test_toon_repository_port_not_instantiable(self):
        with pytest.raises(TypeError):
            TOONRepositoryPort()


# ############################################################
# 7. CACHE FACTORY TESTS
# ############################################################


class TestCacheFactory:
    def test_get_cache_returns_core_cache(self):
        CacheFactory.reset()
        tmpdir = tempfile.mkdtemp()
        try:
            with patch("aicache.cache_factory.get_config") as mock_config:
                mock_config.return_value = {"cache_dir": tmpdir}
                cache = CacheFactory.get_cache()
                assert isinstance(cache, CoreCache)
        finally:
            CacheFactory.reset()
            shutil.rmtree(tmpdir)

    def test_get_cache_singleton(self):
        CacheFactory.reset()
        tmpdir = tempfile.mkdtemp()
        try:
            with patch("aicache.cache_factory.get_config") as mock_config:
                mock_config.return_value = {"cache_dir": tmpdir}
                c1 = CacheFactory.get_cache()
                c2 = CacheFactory.get_cache()
                assert c1 is c2
        finally:
            CacheFactory.reset()
            shutil.rmtree(tmpdir)

    def test_reset_clears_instances(self):
        CacheFactory.reset()
        tmpdir = tempfile.mkdtemp()
        try:
            with patch("aicache.cache_factory.get_config") as mock_config:
                mock_config.return_value = {"cache_dir": tmpdir}
                CacheFactory.get_cache()
                CacheFactory.reset()
                assert CacheFactory._instance is None
        finally:
            CacheFactory.reset()
            shutil.rmtree(tmpdir)

    def test_create_cache_function(self):
        tmpdir = tempfile.mkdtemp()
        try:
            with patch("aicache.cache_factory.get_config") as mock_config:
                mock_config.return_value = {
                    "cache_dir": tmpdir,
                    "ttl": 0,
                    "max_size_mb": 1000,
                    "security.encrypt_sensitive": True,
                }
                cache = create_cache(cache_dir=tmpdir, ttl=300)
                assert isinstance(cache, CoreCache)
                assert cache._config["ttl"] == 300
        finally:
            shutil.rmtree(tmpdir)

    def test_get_cache_convenience_function(self):
        CacheFactory.reset()
        tmpdir = tempfile.mkdtemp()
        try:
            with patch("aicache.cache_factory.get_config") as mock_config:
                mock_config.return_value = {"cache_dir": tmpdir}
                cache = get_cache()
                assert isinstance(cache, CoreCache)
        finally:
            CacheFactory.reset()
            shutil.rmtree(tmpdir)


# ############################################################
# 8. MCP SERVER TESTS
# ############################################################


class TestMCPToolHandler:
    def _make_connection(self):
        """Create an MCPConnection with temp cache dir."""
        tmpdir = tempfile.mkdtemp()
        with patch("aicache.mcp_server.get_config") as mock_config:
            mock_config.return_value = {"cache_dir": tmpdir}
            conn = MCPConnection.__new__(MCPConnection)
            conn.cache = CoreCache(cache_dir=tmpdir)
            conn.config = {"cache_dir": tmpdir}
        return conn, tmpdir

    def test_handle_tool_call_get(self):
        conn, tmpdir = self._make_connection()
        try:
            conn.cache.set("hello", "world")
            result = conn.handle_tool_call("aicache_get", {"prompt": "hello"})
            assert "content" in result
            content = json.loads(result["content"][0]["text"])
            assert content["found"] is True
        finally:
            shutil.rmtree(tmpdir)

    def test_handle_tool_call_set(self):
        conn, tmpdir = self._make_connection()
        try:
            result = conn.handle_tool_call("aicache_set", {"prompt": "test", "response": "answer"})
            assert "content" in result
            content = json.loads(result["content"][0]["text"])
            assert content["success"] is True
        finally:
            shutil.rmtree(tmpdir)

    def test_handle_tool_call_stats(self):
        conn, tmpdir = self._make_connection()
        try:
            result = conn.handle_tool_call("aicache_stats", {})
            assert "content" in result
            content = json.loads(result["content"][0]["text"])
            assert "total_entries" in content
        finally:
            shutil.rmtree(tmpdir)

    def test_handle_tool_call_unknown_tool(self):
        conn, tmpdir = self._make_connection()
        try:
            result = conn.handle_tool_call("nonexistent_tool", {})
            assert "error" in result
            assert "Unknown tool" in result["error"]
        finally:
            shutil.rmtree(tmpdir)

    def test_handle_tool_call_filters_unknown_kwargs(self):
        conn, tmpdir = self._make_connection()
        try:
            # aicache_stats takes no arguments; extra kwargs should be filtered out
            result = conn.handle_tool_call(
                "aicache_stats", {"bogus_arg": "value", "another_unknown": 42}
            )
            assert "content" in result
            # Should not have errored
            assert "error" not in result
        finally:
            shutil.rmtree(tmpdir)

    def test_handle_tool_call_clear_requires_confirm(self):
        conn, tmpdir = self._make_connection()
        try:
            result = conn.handle_tool_call("aicache_clear", {"confirm": False})
            content = json.loads(result["content"][0]["text"])
            assert content["success"] is False
        finally:
            shutil.rmtree(tmpdir)

    def test_handle_tool_call_clear_with_confirm(self):
        conn, tmpdir = self._make_connection()
        try:
            conn.cache.set("p1", "r1")
            result = conn.handle_tool_call("aicache_clear", {"confirm": True})
            content = json.loads(result["content"][0]["text"])
            assert content["success"] is True
        finally:
            shutil.rmtree(tmpdir)

    def test_handle_tool_call_delete(self):
        conn, tmpdir = self._make_connection()
        try:
            conn.cache.set("p1", "r1")
            cache_key = conn.cache._get_cache_key("p1")
            result = conn.handle_tool_call("aicache_delete", {"cache_key": cache_key})
            content = json.loads(result["content"][0]["text"])
            assert content["success"] is True
        finally:
            shutil.rmtree(tmpdir)

    def test_handle_tool_call_list_verbose(self):
        conn, tmpdir = self._make_connection()
        try:
            conn.cache.set("p1", "r1")
            result = conn.handle_tool_call("aicache_list", {"limit": 5, "verbose": True})
            content = json.loads(result["content"][0]["text"])
            assert "entries" in content
            assert "total" in content
        finally:
            shutil.rmtree(tmpdir)

    def test_handle_tool_call_list_non_verbose(self):
        """Non-verbose list returns simplified entries."""
        conn, tmpdir = self._make_connection()
        try:
            conn.cache.set("p1", "r1")
            result = conn.handle_tool_call("aicache_list", {"limit": 5})
            content = json.loads(result["content"][0]["text"])
            assert "entries" in content
            assert "total" in content
        finally:
            shutil.rmtree(tmpdir)

    def test_tools_list_returns_all_tools(self):
        conn, tmpdir = self._make_connection()
        try:
            result = conn.handle_tools_list()
            tool_names = [t["name"] for t in result["tools"]]
            assert "aicache_get" in tool_names
            assert "aicache_set" in tool_names
            assert "aicache_stats" in tool_names
            assert "aicache_clear" in tool_names
            assert "aicache_delete" in tool_names
            assert "aicache_list" in tool_names
            assert "aicache_prune" in tool_names
        finally:
            shutil.rmtree(tmpdir)

    def test_handle_initialize(self):
        conn, tmpdir = self._make_connection()
        try:
            result = conn.handle_initialize({})
            assert "protocolVersion" in result
            assert result["serverInfo"]["name"] == "aicache"
        finally:
            shutil.rmtree(tmpdir)
