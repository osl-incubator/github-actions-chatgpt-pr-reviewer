"""Diagnostic logging tests for meta loggers."""

from __future__ import annotations

import io
import logging
import sys

from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
import main


def _make_buffer_logger() -> Tuple[logging.Logger, io.StringIO]:
    """Return a logger wired to a string buffer."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(f'diag.{id(buf)}')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(handler)
    return logger, buf


def _make_reviewer_with_logger() -> Tuple[
    main.GitHubChatGPTPullRequestReviewer, io.StringIO
]:
    """Create a reviewer instance with only a logger attached."""
    reviewer = object.__new__(main.GitHubChatGPTPullRequestReviewer)
    logger, buf = _make_buffer_logger()
    reviewer._log = logger
    return reviewer, buf


def test_log_chat_meta_emits_usage_and_finish() -> None:
    """_log_chat_meta should emit usage and finish reasons."""

    class _Usage:
        prompt_tokens = 123
        completion_tokens = 456
        total_tokens = 579

    class _Msg:
        tool_calls = [{'id': 't1'}]

    class _Choice:
        finish_reason = 'stop'
        message = _Msg()

    class _ChatObj:
        usage = _Usage()
        choices = [_Choice()]

    reviewer, buf = _make_reviewer_with_logger()
    reviewer._log_chat_meta(_ChatObj())
    out = buf.getvalue()

    assert 'Chat usage: prompt=123 completion=456 total=579' in out
    assert "Chat finish_reasons: ['stop']" in out
    assert 'Chat first choice has_tools=True' in out


def test_log_responses_meta_logs_usage_status_types() -> None:
    """_log_responses_meta should log usage, status, id and item types."""

    class _Usage:
        input_tokens = 1000
        output_tokens = 2000
        total_tokens = 3000

    class _Block:
        text = 'hello'

    class _Item:
        type = 'message'

        @property
        def content(self) -> list[_Block]:
            return [_Block()]

    class _Rsp:
        usage = _Usage()
        status = 'completed'
        id = 'resp_abc'
        output = [_Item()]

    reviewer, buf = _make_reviewer_with_logger()
    reviewer._log_responses_meta(_Rsp())
    out = buf.getvalue()

    assert 'Resp usage: input=1000 output=2000 total=3000' in out
    assert 'Resp status=completed id=resp_abc' in out
    assert "Resp output item types: ['message']" in out


def test_log_responses_meta_handles_none_output() -> None:
    """_log_responses_meta should log when output is None."""

    class _Rsp:
        usage = None
        status = 'completed'
        id = 'resp_empty'
        output = None

    reviewer, buf = _make_reviewer_with_logger()
    reviewer._log_responses_meta(_Rsp())
    out = buf.getvalue()

    assert 'Resp status=completed id=resp_empty' in out
    assert 'Resp output is None' in out
