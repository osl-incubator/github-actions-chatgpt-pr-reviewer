"""
Integration tests for main.py using real GitHub and OpenAI clients.

Requirements:
- A .env file with at least:
    GITHUB_TOKEN=...
    GITHUB_REPOSITORY=owner/repo
    GITHUB_PR_ID=123
    OPENAI_API_KEY=...
  (Optionally)
    OPENAI_MODEL=gpt-4o-mini
    OPENAI_TEMPERATURE=0.5
    OPENAI_MAX_TOKENS=2048
    OPENAI_EXTRA_CRITERIA=security; add tests
    TEST_POST_COMMENT=1        # only if you want to actually post a PR comment

Notes
-----
- By default we DO NOT comment on the PR (side-effect) unless
    TEST_POST_COMMENT=1.
- These are networked integration tests; they will be skipped if required env
    vars are missing.
"""

import os
import sys

from pathlib import Path

import pytest

root_path = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, root_path)

import main  # noqa: E402


@pytest.fixture(scope='session')
def reviewer(ensure_required_env):
    """Create a real reviewer instance using env-backed configuration."""
    return main.GitHubChatGPTPullRequestReviewer()


@pytest.mark.integration
def test_get_diff_returns_mapping(reviewer):
    """Test get_diff_returns_mapping."""
    files_diff = reviewer.get_diff()
    assert isinstance(files_diff, dict)
    # If the PR has changes, each entry should be non-empty strings.
    for fname, diff in files_diff.items():
        assert isinstance(fname, str) and fname.strip()
        assert isinstance(diff, str) and diff.strip()


@pytest.mark.integration
def test_pr_review_generates_output(reviewer):
    """Test pr_review_generates_output."""
    files_diff = reviewer.get_diff()
    review = reviewer.pr_review(files_diff)
    assert isinstance(review, list)
    assert all(isinstance(x, str) for x in review)

    text = '\n'.join(review)
    if not files_diff:
        # Fallback path when PR diff is empty
        assert 'LGTM! (No changes detected in diff)' in text
    else:
        # When there are diffs, we expect at least one section header
        assert '### ' in text


@pytest.mark.integration
def test_prompt_includes_extra_criteria_if_set():
    """Test prompt_includes_extra_criteria_if_set."""
    # Only run this assertion when OPENAI_EXTRA_CRITERIA is provided in .env
    extra = os.getenv('OPENAI_EXTRA_CRITERIA')
    if not extra:
        pytest.skip(
            'OPENAI_EXTRA_CRITERIA not set; skipping prompt content check.'
        )

    r = main.GitHubChatGPTPullRequestReviewer()
    sys_prompt = r.chatgpt_initial_instruction
    # Ensure each non-empty criterion shows up as a bullet in the system prompt
    for item in (s.strip() for s in extra.split(';')):
        if item:
            assert f'- {item}' in sys_prompt
