"""Backward-compatible shim (Phase 10 reviewer cleanup).

The reviewer is now the provider-agnostic :class:`agent.reviewers.reviewer.Reviewer`
built via ``build_client("reviewer")``. ``ClaudeReviewer`` is kept ONLY as an import
alias so existing call sites keep working; it no longer imports ``claude_client``
(the Claude/arbitration coupling is gone from the run path). Anthropic remains one
optional provider choice via the factory.
"""

from agent.reviewers.reviewer import Reviewer


class ClaudeReviewer(Reviewer):
    """Deprecated name for :class:`Reviewer`. Use ``Reviewer`` in new code."""
    pass
