"""Behavior tests for logging when the model returns empty output."""

from __future__ import annotations

import io
import logging

from typing import Dict, List, Tuple

import main


def _buffer_logger() -> Tuple[logging.Logger, io.StringIO]:
    """Return a logger wired to an in-memory buffer."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(f'prdiag.{id(buf)}')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(handler)
    return logger, buf


def _bare_reviewer() -> main.GitHubChatGPTPullRequestReviewer:
    """Create a reviewer without running __init__."""
    return object.__new__(main.GitHubChatGPTPullRequestReviewer)


def _wire_minimal_config(
    r: main.GitHubChatGPTPullRequestReviewer, logger: logging.Logger
) -> None:
    """Attach minimal fields so pr_review can run without env."""
    r._log = logger
    r.exclude_globs = []
    r.chatgpt_initial_instruction = 'x'
    r.openai_model = 'gpt-4o-mini'
    r.openai_reasoning_mode = 'off'
    r.openai_max_tokens = 4096
    r.openai_max_completion_tokens = 4096
    r.openai_max_input_tokens = 0
    r._ctx_default = 128_000
    r._out_default = 16_384


def test_pr_review_logs_warning_and_placeholder_on_empty_output() -> None:
    """When model returns empty, log a warning and add placeholder text."""
    logger, buf = _buffer_logger()
    r = _bare_reviewer()
    _wire_minimal_config(r, logger)

    def _empty_review(_: str) -> str:
        return ''

    r._review_one = _empty_review  # type: ignore

    pr_diff: Dict[str, str] = {'file.py': "@@ -0,0 +1 @@\n+print('x')\n"}

    out: List[str] = r.pr_review(pr_diff)
    joined = '\n'.join(out)
    logs = buf.getvalue()

    assert 'Empty model output for "file.py"' in logs
    assert '### file.py' in joined
    assert '_No content returned by model._' in joined


def test_pr_review_logs_exception_and_includes_error_text() -> None:
    """If review fails with an exception, log and include the error."""
    logger, buf = _buffer_logger()
    r = _bare_reviewer()
    _wire_minimal_config(r, logger)

    def _boom(_: str, __: str) -> Tuple[str, bool]:
        raise ValueError('boom')

    r._review_file_in_chunks = _boom  # type: ignore

    pr_diff: Dict[str, str] = {'f.txt': '@@ -1 +1 @@\n- a\n+ b\n'}
    out: List[str] = r.pr_review(pr_diff)
    joined = '\n'.join(out)
    logs = buf.getvalue()

    assert 'Review failed for "f.txt"' in logs
    assert 'ChatGPT was not able to review the file.' in joined
    assert 'Error: boom' in joined
