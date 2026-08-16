"""Provider adapters.

The only place in the package permitted to touch a vendor's message format.
Everything else speaks the canonical IR — the moment that rule breaks, this is
a wrapper for one vendor instead of an engine.

* ``anthropic`` — Messages API, plus the Tier 0 client wrap
* ``openai``    — Chat Completions; also covers vLLM, Ollama, LM Studio, TGI
                  and any other OpenAI-compatible server
* ``generic``   — plain chat dicts or a single prompt string, for anything else
                  including self-hosted completion endpoints
"""

from . import anthropic, generic, openai

__all__ = ["anthropic", "openai", "generic"]
