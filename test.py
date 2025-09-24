import os
from types import SimpleNamespace

import pytest
import main

# --- Fakes ---------------------------------------------------------------

_REGISTRY = {}  # (repo, pr_id) -> FakePR


class FakePR:
    def __init__(self, pr_id: int):
        self.pr_id = pr_id
        self.comments = []

    def create_issue_comment(self, body: str):
        self.comments.append(body)


class FakeRepo:
    def __init__(self, name: str):
        self.name = name

    def get_pull(self, pr_id: int):
        key = (self.name, int(pr_id))
        if key not in _REGISTRY:
            _REGISTRY[key] = FakePR(pr_id)
        return _REGISTRY[key]


class FakeGithub:
    def __init__(self, auth=None):
        self.auth = auth

    def get_repo(self, name: str):
        return FakeRepo(name)


class FakeAuth:
    class Token:
        def __init__(self, token: str):
            self.token = token


class FakeOpenAI:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        # Mimic OpenAI SDK v1 response structure
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="LGTM!"))]
        )


class FakeResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text


# --- Fixtures ------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registry():
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()


@pytest.fixture
def set_env(monkeypatch):
    monkeypatch.setenv("GITHUB_PR_ID", "123")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_testtoken")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    # Use our fakes
    monkeypatch.setattr(main, "Github", FakeGithub, raising=True)
    monkeypatch.setattr(main, "Auth", FakeAuth, raising=True)
    monkeypatch.setattr(main, "OpenAI", FakeOpenAI, raising=True)


# --- Helpers -------------------------------------------------------------

DIFF_TWO_FILES = (
    "diff --git a/foo.py b/foo.py\n"
    "index 111..222 100644\n"
    "--- a/foo.py\n"
    "+++ b/foo.py\n"
    "@@ -1 +1 @@\n"
    "-print('Hello')\n"
    "+print('Hello, world')\n"
    "diff --git a/bar.txt b/bar.txt\n"
    "index 333..444 100644\n"
    "--- a/bar.txt\n"
    "+++ b/bar.txt\n"
    "@@ -1 +1 @@\n"
    "-old\n"
    "+new\n"
)

def fake_requests_get_with_text(text: str):
    def _fake(url, headers=None, timeout=None):
        return FakeResponse(200, text)
    return _fake


# --- Tests ---------------------------------------------------------------

def test_run_posts_review_for_each_file(monkeypatch, set_env):
    monkeypatch.setattr(main.requests, "get", fake_requests_get_with_text(DIFF_TWO_FILES), raising=True)

    reviewer = main.GitHubChatGPTPullRequestReviewer()
    openai_client = reviewer._openai  # FakeOpenAI
    reviewer.run()

    pr = _REGISTRY[("owner/repo", 123)]
    assert len(pr.comments) == 1
    body = pr.comments[0]
    assert body.startswith("# OSL ChatGPT Reviewer")
    assert "### foo.py" in body
    assert "### bar.txt" in body
    assert "LGTM!" in body

    # Called once per file
    assert len(openai_client.calls) == 2
    # First call contains a system + user message
    msgs0 = openai_client.calls[0]["messages"]
    assert msgs0[0]["role"] == "system"
    assert msgs0[1]["role"] == "user"
    assert "foo.py" in msgs0[1]["content"]


def test_empty_diff_yields_lgtm(monkeypatch, set_env):
    monkeypatch.setattr(main.requests, "get", fake_requests_get_with_text(""), raising=True)

    reviewer = main.GitHubChatGPTPullRequestReviewer()
    reviewer.run()

    pr = _REGISTRY[("owner/repo", 123)]
    assert len(pr.comments) == 1
    body = pr.comments[0]
    assert "LGTM! (No changes detected in diff)" in body


def test_extra_criteria_are_in_system_prompt(monkeypatch, set_env):
    monkeypatch.setenv("OPENAI_EXTRA_CRITERIA", "security; add tests ")
    monkeypatch.setattr(main.requests, "get", fake_requests_get_with_text(DIFF_TWO_FILES), raising=True)

    reviewer = main.GitHubChatGPTPullRequestReviewer()
    openai_client = reviewer._openai
    reviewer.run()

    msgs0 = openai_client.calls[0]["messages"]
    system_text = msgs0[0]["content"]
    assert "- security" in system_text
    assert "- add tests" in system_text
