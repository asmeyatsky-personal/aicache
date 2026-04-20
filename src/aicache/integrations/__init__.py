"""Drop-in caching wrappers for Anthropic / OpenAI SDKs.

Presentation-layer entry point for AI developers. Every wrapper routes
through the application use cases via ``infrastructure.Container``.

Example:

    from anthropic import Anthropic
    from aicache.integrations.anthropic import cached_client

    client = cached_client(Anthropic())
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from ._common import build_request_fingerprint

__all__ = ["build_request_fingerprint"]
