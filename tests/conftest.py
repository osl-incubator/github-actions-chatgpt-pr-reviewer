"""Test configuration."""

import os

from pathlib import Path

import pytest

from dotenv import load_dotenv

REQUIRED_ENV = (
    'GITHUB_TOKEN',
    'GITHUB_REPOSITORY',
    'GITHUB_PR_ID',
    'OPENAI_API_KEY',
)


@pytest.fixture(scope='session', autouse=True)
def load_env_from_dotenv():
    """
    Load variables from the first .env we can find (cwd, repo root, tests dir).

    Does not override already-set environment variables.
    """
    candidates = [
        Path.cwd() / '.env',
        Path(__file__).resolve().parent.parent / '.env',
        Path(__file__).resolve().parent / '.env',
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(path.as_posix(), override=False)
            break


@pytest.fixture(scope='session')
def ensure_required_env():
    """Ensure required environment."""
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        pytest.skip(
            f'Missing env vars for integration test: {", ".join(missing)}'
        )
    return {k: os.getenv(k) for k in REQUIRED_ENV}
