"""Integration tests for main.py using real GitHub and OpenAI clients."""

import os
import sys

from pathlib import Path
from typing import Dict

import pytest

root_path = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, root_path)

import main  # noqa: E402


@pytest.fixture(scope='session')
def reviewer(
    ensure_required_env: Dict[str, str],
) -> main.GitHubChatGPTPullRequestReviewer:
    """Create a real reviewer instance using env-backed configuration."""
    return main.GitHubChatGPTPullRequestReviewer()


@pytest.mark.integration
def test_get_diff_returns_mapping(
    reviewer: main.GitHubChatGPTPullRequestReviewer,
) -> None:
    """Test get_diff_returns_mapping."""
    files_diff = reviewer.get_diff()
    assert isinstance(files_diff, dict)
    for fname, diff in files_diff.items():
        assert isinstance(fname, str) and fname.strip()
        assert isinstance(diff, str) and diff.strip()


@pytest.mark.integration
def test_pr_review_generates_output(
    reviewer: main.GitHubChatGPTPullRequestReviewer,
) -> None:
    """Test pr_review_generates_output."""
    files_diff = reviewer.get_diff()
    review = reviewer.pr_review(files_diff)
    assert isinstance(review, list)
    assert all(isinstance(x, str) for x in review)

    text = '\n'.join(review)
    if not files_diff:
        assert 'LGTM! (No changes detected in diff)' in text
    else:
        assert '### ' in text


@pytest.mark.integration
def test_prompt_includes_extra_criteria_if_set() -> None:
    """Test prompt_includes_extra_criteria_if_set."""
    extra = os.getenv('OPENAI_EXTRA_CRITERIA')
    if not extra:
        pytest.skip(
            'OPENAI_EXTRA_CRITERIA not set; skipping prompt content check.'
        )

    r = main.GitHubChatGPTPullRequestReviewer()
    sys_prompt = r.chatgpt_initial_instruction
    for item in (s.strip() for s in extra.split(';')):
        if item:
            assert f'- {item}' in sys_prompt
