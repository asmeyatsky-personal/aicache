"""
Security utilities for AI Cache.

This module provides:
- Input sanitization
- PII detection and masking
- Basic encryption for sensitive data
"""

import hashlib
import re
from typing import Any

# Default sensitive patterns
DEFAULT_SENSITIVE_PATTERNS = [
    (r"api[_-]?key", "API_KEY"),
    (r"secret[_-]?key", "SECRET_KEY"),
    (r"password", "PASSWORD"),
    (r"token", "TOKEN"),
    (r"auth", "AUTH"),
    (r"bearer", "BEARER"),
    (r"private[_-]?key", "PRIVATE_KEY"),
    (r"access[_-]?key", "ACCESS_KEY"),
    (r"session[_-]?id", "SESSION_ID"),
    (r"credential", "CREDENTIAL"),
]

# Common API key patterns
API_KEY_PATTERNS = [
    r"sk-[a-zA-Z0-9]{20,}",  # OpenAI
    r"xox[baprs]-[a-zA-Z0-9]{10,}",  # Slack
    r"ghp_[a-zA-Z0-9]{36}",  # GitHub
    r"ghp_[a-zA-Z0-9]{10,}",  # GitHub (short)
    r"AIza[0-9A-Za-z_-]{35}",  # Google API
    r"ya29\.[0-9A-Za-z_-]+",  # Google OAuth
]


class SecurityUtils:
    """Security utilities for input validation and PII handling."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._sensitive_patterns = DEFAULT_SENSITIVE_PATTERNS.copy()
        self._api_patterns = API_KEY_PATTERNS.copy()
        self._init_custom_patterns()

    def _init_custom_patterns(self):
        """Initialize custom patterns from config"""
        custom = self.config.get("sensitive_patterns", [])
        if custom:
            for pattern in custom:
                if isinstance(pattern, tuple):
                    self._sensitive_patterns.append(pattern)
                elif isinstance(pattern, str):
                    self._sensitive_patterns.append((pattern, "CUSTOM"))

    def sanitize_input(self, text: str) -> str:
        """
        Sanitize user input by removing potentially harmful content.

        Args:
            text: Input text to sanitize

        Returns:
            Sanitized text
        """
        if not text:
            return text

        # Remove null bytes
        text = text.replace("\x00", "")

        # Remove excessive whitespace (but preserve newlines)
        text = re.sub(r"[ \t]+", " ", text)

        # Remove control characters except newlines and tabs
        text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")

        return text.strip()

    def detect_pii(self, text: str) -> list[dict[str, Any]]:
        """
        Detect potential PII in text.

        Args:
            text: Text to analyze

        Returns:
            List of detected PII with positions
        """
        findings = []

        if not text:
            return findings

        # Check for API key patterns
        for pattern in self._api_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                findings.append(
                    {
                        "type": "api_key",
                        "value": match.group()[:10] + "...",  # Truncate for safety
                        "start": match.start(),
                        "end": match.end(),
                        "pattern": pattern,
                    }
                )

        # Check for sensitive key patterns
        for pattern, label in self._sensitive_patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            matches = regex.finditer(text)
            for match in matches:
                findings.append(
                    {
                        "type": label.lower(),
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "pattern": pattern,
                    }
                )

        return findings

    def mask_pii(self, text: str, mask_char: str = "*") -> str:
        """
        Mask detected PII in text.

        Args:
            text: Text to mask
            mask_char: Character to use for masking

        Returns:
            Text with PII masked
        """
        findings = self.detect_pii(text)

        if not findings:
            return text

        # Sort by position descending to avoid index shifting
        findings.sort(key=lambda x: x["start"], reverse=True)

        result = text
        for finding in findings:
            start, end = finding["start"], finding["end"]
            # Replace with asterisks, keeping first and last chars for some types
            length = end - start
            if finding["type"] == "api_key":
                # Keep first 4 chars for API keys
                masked = result[start : start + 4] + mask_char * (length - 4)
            else:
                masked = mask_char * length

            result = result[:start] + masked + result[end:]

        return result

    def hash_sensitive(self, value: str) -> str:
        """
        Create a secure hash of sensitive value.

        Args:
            value: Value to hash

        Returns:
            SHA-256 hash (hex encoded)
        """
        return hashlib.sha256(value.encode()).hexdigest()

    def is_safe_prompt(self, prompt: str) -> bool:
        """
        Check if prompt appears safe (basic heuristics).

        Args:
            prompt: Prompt to check

        Returns:
            True if prompt appears safe
        """
        if not prompt:
            return False

        # Check for suspiciously long prompts (potential DoS)
        if len(prompt) > 100000:
            return False

        # Check for suspicious patterns
        dangerous_patterns = [
            r"<script[^>]*>.*?</script>",  # XSS
            r"eval\s*\(",  # Code injection
            r"exec\s*\(",  # Code injection
            r"import\s+os",  # Import injection
            r"__import__\(",  # Import injection
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return False

        return True

    def validate_context(self, context: dict[str, Any] | None) -> dict[str, Any]:
        """
        Validate and sanitize context dictionary.

        Args:
            context: Context dictionary to validate

        Returns:
            Sanitized context (or empty dict if invalid)
        """
        if not context:
            return {}

        if not isinstance(context, dict):
            return {}

        sanitized = {}
        for key, value in context.items():
            # Validate key
            if not isinstance(key, str):
                continue

            # Skip sensitive keys (but don't fail, just skip)
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive, _ in self._sensitive_patterns):
                sanitized[key] = "[REDACTED]"
                continue

            # Validate value
            if isinstance(value, (str, int, float, bool, type(None))):
                sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = self.validate_context(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    v for v in value if isinstance(v, (str, int, float, bool, type(None)))
                ]

        return sanitized


# Global instance
_security_utils = None


def get_security_utils(config: dict[str, Any] | None = None) -> SecurityUtils:
    """Get global security utils instance."""
    global _security_utils
    if _security_utils is None:
        _security_utils = SecurityUtils(config)
    return _security_utils


def sanitize_input(text: str) -> str:
    """Convenience function for input sanitization."""
    return get_security_utils().sanitize_input(text)


def detect_pii(text: str) -> list[dict[str, Any]]:
    """Convenience function for PII detection."""
    return get_security_utils().detect_pii(text)


def mask_pii(text: str) -> str:
    """Convenience function for PII masking."""
    return get_security_utils().mask_pii(text)


def is_safe_prompt(prompt: str) -> bool:
    """Convenience function for prompt safety check."""
    return get_security_utils().is_safe_prompt(prompt)


def validate_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """Convenience function for context validation."""
    return get_security_utils().validate_context(context)
