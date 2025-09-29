"""Redaction unit tests."""

from __future__ import annotations

import io
import logging
import sys

from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
import main


def _make_logger() -> Tuple[logging.Logger, io.StringIO]:
    """Return a logger wired with the redacting formatter and a buffer."""
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.DEBUG)
    fmt = main.RedactingFormatter('%(message)s', main._redaction_patterns())
    h.setFormatter(fmt)

    logger = logging.getLogger(f'test.redaction.{id(buf)}')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(h)
    return logger, buf


def test_redaction_request_options_and_headers() -> None:
    """Ensure SDK request payloads and headers are redacted."""
    logger, buf = _make_logger()
    try:
        msg = (
            "Request options: {'method': 'post', 'url': '/responses', "
            "'idempotency_key': 'stainless-python-retry-abc123', "
            "'json_data': {'input': [{'role': 'system', 'content': 'secret'}],"
            " 'messages': [{'role': 'user', 'content': '```diff\\n@@\\n```'}]"
            '}}\n'
            'Set-Cookie: __cf_bm=xyz; path=/; HttpOnly\n'
            "('openai-organization', 'my_org'), "
            "('openai-project', 'proj_123')\n"
            'Here is fenced: ```TOPSECRET```'
        )
        logger.debug(msg)
        out = buf.getvalue()

        assert 'Request options: [REDACTED]' in out
        assert 'Set-Cookie: [REDACTED]' in out
        assert "('openai-organization', '[REDACTED]')" in out
        assert "('openai-project', '[REDACTED]')" in out
        assert '```[REDACTED]```' in out

        assert '__cf_bm' not in out
        assert 'my_org' not in out
        assert 'proj_123' not in out
        assert 'TOPSECRET' not in out
    finally:
        for h in list(logger.handlers):
            logger.removeHandler(h)


def test_redaction_generic_credentials() -> None:
    """Ensure generic credential shapes are redacted."""
    logger, buf = _make_logger()
    try:
        msg = (
            'authorization: Bearer sk-live-abc.def\n'
            'api_key=sk-123456\n'
            "API-KEY: 'sk-abcdef'\n"
            "idempotency_key: 'abc-xyz-123'"
        )
        logger.debug(msg)
        out = buf.getvalue()

        assert 'authorization: [REDACTED]' in out
        assert 'api_key=[REDACTED]' in out
        assert "API-KEY: '[REDACTED]'" in out
        assert "idempotency_key: '[REDACTED]'" in out

        assert 'sk-live' not in out
        assert 'sk-123456' not in out
        assert 'abc-xyz-123' not in out
    finally:
        for h in list(logger.handlers):
            logger.removeHandler(h)
