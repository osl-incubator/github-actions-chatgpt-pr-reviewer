"""Test empty output."""

from __future__ import annotations

import logging

from typing import Dict, List

import pytest

from main import GitHubChatGPTPullRequestReviewer


@pytest.mark.integration
def test_openai_responses_completed_real(
    ensure_required_env: Dict[str, str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Run test_openai_responses_completed_real.

    Real Responses API call returns non-empty output and logs usage/status.
    """
    # Small but valid; must be >= 16 for reasoning models.
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-5')
    monkeypatch.setenv('OPENAI_REASONING', 'on')
    monkeypatch.setenv('OPENAI_MAX_COMPLETION_TOKENS', '64')

    caplog.set_level(logging.DEBUG)
    r = GitHubChatGPTPullRequestReviewer()

    sys = 'You are a concise reviewer.'
    user = (
        'Summarize this diff in one short sentence. Focus on risk only.\n'
        'Diff:\n--- a/file\n+++ b/file\n@@ -1 +1 @@\n- old\n+ new\n'
    )

    out = r._call_openai_responses(sys, user)

    assert isinstance(out, str)
    assert out.strip() != ''

    logs = '\n'.join(m.message for m in caplog.records)
    assert 'Resp status=' in logs
    assert 'Resp usage:' in logs
    assert 'Resp output item types:' in logs


@pytest.mark.integration
def test_pr_review_handles_truncation_and_logs_real(
    ensure_required_env: Dict[str, str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Run test_pr_review_handles_truncation_and_logs_real.

    Use minimal valid output tokens (16) to likely force truncation but still
    get a successful response. Ensure the diagnostic logs are present and the
    review produces some content rather than crashing.
    """
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-5')
    monkeypatch.setenv('OPENAI_REASONING', 'on')
    # IMPORTANT: 16 is the minimum allowed for reasoning models.
    monkeypatch.setenv('OPENAI_MAX_COMPLETION_TOKENS', '16')

    caplog.set_level(logging.DEBUG)
    r = GitHubChatGPTPullRequestReviewer()

    pr_diff: Dict[str, str] = {
        'tiny.txt': (
            ' --git a/tiny.txt b/tiny.txt\n'
            '--- a/tiny.txt\n'
            '+++ b/tiny.txt\n'
            '@@ -1,1 +1,1 @@\n'
            '-hello\n'
            '+hello world\n'
        )
    }

    results: List[str] = r.pr_review(pr_diff)

    # We got a single section for tiny.txt and it isn't empty.
    assert len(results) == 1
    assert '### tiny.txt' in results[0]
    assert results[0].strip() != ''

    logs = '\n'.join(m.message for m in caplog.records)
    # These only appear on successful Responses calls.
    assert 'Resp status=' in logs
    assert 'Resp usage:' in logs
