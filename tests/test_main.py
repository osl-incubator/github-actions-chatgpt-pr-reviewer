"""Integration tests for main.py using real GitHub and OpenAI clients."""

import os
import sys

from pathlib import Path
from typing import Dict

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
import main


def _build_diff(body_chars: int) -> str:
    """Return a synthetic unified diff with ~body_chars of added content."""
    header = (
        'diff --git a/file.py b/file.py\n'
        'index 0000000..1111111 100644\n'
        '--- a/file.py\n'
        '+++ b/file.py\n'
        '@@ -1,3 +1,3 @@\n'
    )
    line = '+' + ('x' * 79) + '\n'
    need = max(1, (body_chars // len(line)) + 1)
    return header + (line * need)


@pytest.fixture(scope='module')
def require_openai() -> None:
    """Skip if no OpenAI key is present."""
    if not os.getenv('OPENAI_API_KEY'):
        pytest.skip('OPENAI_API_KEY not set')


def _base_env() -> Dict[str, str]:
    """Return a minimal env mapping for the reviewer object."""
    return {
        'GITHUB_PR_ID': '1',
        'GITHUB_TOKEN': 'ghs_dummy',
        'GITHUB_REPOSITORY': 'owner/repo',
        'OPENAI_MODEL': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
        'OPENAI_REASONING': os.getenv('OPENAI_REASONING', 'off'),
    }


@pytest.fixture(scope='session')
def reviewer(
    ensure_required_env: Dict[str, str],
) -> main.GitHubChatGPTPullRequestReviewer:
    """Create a real reviewer instance using env-backed configuration."""
    return main.GitHubChatGPTPullRequestReviewer()


def test_get_diff_returns_mapping(
    reviewer: main.GitHubChatGPTPullRequestReviewer,
) -> None:
    """Test get_diff_returns_mapping."""
    files_diff = reviewer.get_diff()
    assert isinstance(files_diff, dict)
    for fname, diff in files_diff.items():
        assert isinstance(fname, str) and fname.strip()
        assert isinstance(diff, str) and diff.strip()


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
    assert 'No content returned by model' not in text


def test_prompt_includes_extra_criteria_if_set() -> None:
    """Test prompt_includes_extra_criteria_if_set."""
    extra = os.getenv('PROMPT_EXTRA_CRITERIA')
    if not extra:
        pytest.skip(
            'PROMPT_EXTRA_CRITERIA not set; skipping prompt content check.'
        )

    r = main.GitHubChatGPTPullRequestReviewer()
    sys_prompt = r.chatgpt_initial_instruction
    for item in (s.strip() for s in extra.split(';')):
        if item:
            assert f'- {item}' in sys_prompt


def test_pr_review_adds_note_when_chunked(
    require_openai: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Large file yields chunking note and per-part headings."""
    env = _base_env()
    env['OPENAI_MAX_INPUT_TOKENS'] = '1024'
    env['OPENAI_MAX_TOKENS'] = '128'
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    reviewer = main.GitHubChatGPTPullRequestReviewer()

    big_diff = _build_diff(body_chars=1100)
    out = reviewer.pr_review({'pkg/file.py': big_diff})
    blob = '\n'.join(out)

    assert 'Too many changes in this file' in blob
    assert '**Part 1/' in blob
    assert '_No content returned by model._' not in blob


def test_pr_review_no_note_when_small(
    require_openai: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Small file stays in a single request and has no chunking note."""
    env = _base_env()
    env['OPENAI_MAX_INPUT_TOKENS'] = '4096'
    env['OPENAI_MAX_TOKENS'] = '256'
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    reviewer = main.GitHubChatGPTPullRequestReviewer()

    small_diff = _build_diff(body_chars=200)
    out = reviewer.pr_review({'pkg/small.py': small_diff})
    blob = '\n'.join(out)

    assert 'Too many changes in this file' not in blob
    assert '**Part 1/' not in blob
    assert '### pkg/small.py' in blob


def test_pr_review_marks_deleted_file(
    require_openai: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleted files are recognized and skipped."""
    env = _base_env()
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    reviewer = main.GitHubChatGPTPullRequestReviewer()

    deleted = (
        'diff --git a/file.py b/file.py\n'
        'deleted file mode 100644\n'
        '--- a/file.py\n'
        '+++ /dev/null\n'
    )
    out = reviewer.pr_review({'file.py': deleted})
    blob = '\n'.join(out)

    assert '_File deleted; no review._' in blob
