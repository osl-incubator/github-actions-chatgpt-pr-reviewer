"""Test configuration."""

import os

from pathlib import Path
from typing import Dict

import pytest

from dotenv import load_dotenv

REQUIRED_ENV = (
    'GITHUB_TOKEN',
    'GITHUB_REPOSITORY',
    'GITHUB_PR_ID',
    'OPENAI_API_KEY',
)


@pytest.fixture(scope='session', autouse=True)
def load_env_from_dotenv() -> None:
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
    os.environ['LOG_LEVEL'] = 'DEBUG'


@pytest.fixture(scope='session')
def ensure_required_env() -> Dict[str, str]:
    """Ensure required environment."""
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        pytest.skip(
            f'Missing env vars for integration test: {", ".join(missing)}'
        )
    env: Dict[str, str] = {k: os.environ[k] for k in REQUIRED_ENV}
    return env
