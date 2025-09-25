"""GitHub Actions PR Reviewer empowered by AI."""

import html
import os

from typing import Dict, List

import requests

from github import Auth, Github
from openai import OpenAI


def _as_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {'1', 'true', 'yes', 'on'}


def _is_reasoning_model(model: str) -> bool:
    """
    Check if the model has reasoning feature.

    Heuristic: o*-series and GPT-5* are reasoning-capable and typically require
    max_completion_tokens (Chat) or max_output_tokens (Responses).
    """
    m = model.lower()
    return m.startswith(('o1', 'o2', 'o3', 'o4', 'o-')) or m.startswith(
        'gpt-5'
    )


class GitHubChatGPTPullRequestReviewer:
    """GitHubChatGPTPullRequestReviewer class."""

    def __init__(self) -> None:
        """Initialize the class."""
        self._config_gh()
        self._config_openai()

    def _config_gh(self) -> None:
        self.gh_pr_id = os.environ.get('GITHUB_PR_ID')
        if not self.gh_pr_id:
            raise RuntimeError('GITHUB_PR_ID is required')

        gh_token = os.environ.get('GITHUB_TOKEN')
        if not gh_token:
            raise RuntimeError('GITHUB_TOKEN is required')

        self.gh_repo_name = os.environ.get('GITHUB_REPOSITORY')
        if not self.gh_repo_name:
            raise RuntimeError('GITHUB_REPOSITORY is required')

        gh_api_url = os.environ.get('GITHUB_API_URL', 'https://api.github.com')
        self.gh_pr_url = (
            f'{gh_api_url}/repos/{self.gh_repo_name}/pulls/{self.gh_pr_id}'
        )
        self.gh_headers = {
            'Authorization': f'token {gh_token}',
            'Accept': 'application/vnd.github.v3.diff',
        }

        self.gh_api = Github(auth=Auth.Token(gh_token))

    def _config_openai(self) -> None:
        self.openai_model = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
        self.openai_temperature = float(
            os.environ.get('OPENAI_TEMPERATURE', '0.5')
        )
        # legacy + new names
        self.openai_max_tokens = int(
            os.environ.get('OPENAI_MAX_TOKENS', '2048')
        )
        self.openai_max_completion_tokens = int(
            os.environ.get(
                'OPENAI_MAX_COMPLETION_TOKENS', str(self.openai_max_tokens)
            )
        )

        # Reasoning toggle:
        #   auto (default): detect based on model name
        #   on/off: force behavior
        reasoning_mode_env = (
            os.environ.get('OPENAI_REASONING', 'auto').strip().lower()
        )
        if reasoning_mode_env not in {'auto', 'on', 'off'}:
            reasoning_mode_env = 'auto'
        self.openai_reasoning_mode = reasoning_mode_env
        self.openai_reasoning_effort = os.environ.get(
            'OPENAI_REASONING_EFFORT', 'medium'
        )

        self._openai = OpenAI()

        extra_criteria = self._prepare_extra_criteria(
            os.environ.get('OPENAI_EXTRA_CRITERIA', '').strip()
        )

        self.chatgpt_initial_instruction = (
            'You are a GitHub PR reviewer bot. You will receive a text '
            'containing the diff from a PR with all proposed changes. '
            'Analyze it and return suggestions '
            'to improve or fix issues, using the following criteria:\n'
            '- best practices that would improve the changes\n'
            '- code style formatting\n'
            '- recommendations specific to the programming language\n'
            '- performance improvements\n'
            '- improvements from the software engineering perspective\n'
            '- docstrings, when applicable\n'
            '- prefer explicit over implicit (e.g., in Python, avoid '
            '`from x import *`)\n'
            f'{extra_criteria}\n'
            'Return your response in Markdown. '
            'If everything looks good, just say: "LGTM!"'
        )

    def _want_reasoning(self) -> bool:
        if self.openai_reasoning_mode == 'on':
            return True
        if self.openai_reasoning_mode == 'off':
            return False
        return _is_reasoning_model(self.openai_model)  # auto

    def _prepare_extra_criteria(self, extra_criteria: str) -> str:
        if not extra_criteria:
            return ''
        lines = []
        for item in extra_criteria.split(';'):
            _item = item.strip()
            if _item:
                if not _item.startswith('-'):
                    _item = '- ' + _item
                lines.append(_item)
        return '\n'.join(lines)

    def get_pr_content(self) -> str:
        """Get the PR content."""
        resp = requests.get(
            self.gh_pr_url, headers=self.gh_headers, timeout=60
        )

        OK_STATUS = 200
        if resp.status_code != OK_STATUS:
            raise RuntimeError(
                f'GitHub API error {resp.status_code}: {resp.text}'
            )
        return resp.text or ''

    def get_diff(self) -> Dict[str, str]:
        """Get Diff content."""
        _ = self.gh_api.get_repo(self.gh_repo_name).get_pull(
            int(self.gh_pr_id)
        )  # ensure PR exists

        content = self.get_pr_content()
        if not content.strip():
            return {}

        parts = content.split('diff')
        files_diff: Dict[str, str] = {}
        bucket: List[str] = []
        file_name = ''

        for part in parts:
            if not part:
                continue
            if not part.startswith(' --git a/'):
                bucket.append(part)
                continue
            if file_name and bucket:
                files_diff[file_name] = '\n'.join(bucket)
            file_name = part.split('b/')[1].splitlines()[0]
            bucket = [part]

        if file_name and bucket:
            files_diff[file_name] = '\n'.join(bucket)

        return files_diff

    def _call_openai_chat(
        self, system_text: str, user_text: str, use_completion_tokens: bool
    ) -> str:
        """
        Call OpenAI Chat Completion.

        Chat Completions path.
        Some models require max_completion_tokens (reasoning).
        """
        kwargs = {
            'model': self.openai_model,
            'messages': [
                {'role': 'system', 'content': system_text},
                {'role': 'user', 'content': user_text},
            ],
        }
        if use_completion_tokens:
            kwargs['max_completion_tokens'] = (
                self.openai_max_completion_tokens
            )  # reasoning models
        else:
            kwargs['max_tokens'] = self.openai_max_tokens  # classic models

        # temperature is not supported by some reasoning models;
        # pass it only for non-reasoning
        if not use_completion_tokens:
            kwargs['temperature'] = self.openai_temperature

        completion = self._openai.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ''

    def _call_openai_responses(self, system_text: str, user_text: str) -> str:
        """Responses API path (preferred for reasoning models)."""
        rsp = self._openai.responses.create(
            model=self.openai_model,
            reasoning={
                'effort': self.openai_reasoning_effort
            },  # supported by o-series
            max_output_tokens=self.openai_max_completion_tokens,
            input=[
                {'role': 'system', 'content': system_text},
                {'role': 'user', 'content': user_text},
            ],
        )
        # SDK provides .output_text; fallback to string if absent
        text = getattr(rsp, 'output_text', None)
        if not text:
            # defensive fallback
            try:
                text = ''.join(
                    (
                        block.text
                        for item in getattr(rsp, 'output', [])
                        for block in getattr(item, 'content', [])
                    )
                )
            except Exception:
                text = ''
        return text

    def _review_one(self, message_diff: str) -> str:
        sys = self.chatgpt_initial_instruction
        want_reason = self._want_reasoning()

        try:
            if want_reason:
                # Prefer Responses API for reasoning models
                # (use max_output_tokens).
                return self._call_openai_responses(sys, message_diff)
            else:
                # Classic models: Chat Completions with max_tokens
                return self._call_openai_chat(
                    sys, message_diff, use_completion_tokens=False
                )
        except Exception as e:
            msg = str(e)
            # If the model complains about max_tokens,
            # retry with max_completion_tokens via Chat
            if 'max_tokens' in msg and 'max_completion_tokens' in msg:
                try:
                    return self._call_openai_chat(
                        sys, message_diff, use_completion_tokens=True
                    )
                except Exception as e_openai:
                    print('WARNING:\n', e_openai)
            # As a last resort, try the other API surface too
            try:
                if want_reason:
                    # we already tried Responses;
                    # try Chat with completion tokens
                    return self._call_openai_chat(
                        sys, message_diff, use_completion_tokens=True
                    )
                else:
                    # we already tried Chat; try Responses
                    return self._call_openai_responses(sys, message_diff)
            except Exception as e_openai:
                print('WARNING:\n', e_openai)
                raise  # bubble up so caller can render a per-file error

    def pr_review(self, pr_diff: Dict[str, str]) -> List[str]:
        """Call the PR Review."""
        if not pr_diff:
            return ['LGTM! (No changes detected in diff)']

        results: List[str] = []
        for filename, diff in pr_diff.items():
            message_diff = f'file:\n```{filename}```\ndiff:\n```{diff}```'
            print(
                'Estimated tokens:',
                int(len(self.chatgpt_initial_instruction + message_diff) / 4),
            )
            try:
                content = self._review_one(message_diff).strip()
                if not content:
                    content = '_No content returned by model._'
                results.append(f'### {filename}\n\n{content}\n\n---')
            except Exception as e:
                results.append(
                    f'### {filename}\nChatGPT was not able to review the file.'
                    f' Error: {html.escape(str(e))}'
                )
        return results

    def comment_review(self, review: List[str]) -> None:
        """Create a comment with the review content."""
        repo = self.gh_api.get_repo(self.gh_repo_name)
        pull_request = repo.get_pull(int(self.gh_pr_id))
        comment = (
            '# OSL ChatGPT Reviewer\n\n'
            '*NOTE: This is generated by an AI program, so some '
            'comments may not make sense.*\n\n'
        ) + '\n'.join(review)
        try:
            pull_request.create_issue_comment(comment)
        except Exception as e:
            print(f'[WARN] Could not post PR comment: {e}')

    def run(self) -> None:
        """Run the PR Reviewer."""
        pr_diff = self.get_diff()
        review = self.pr_review(pr_diff)
        self.comment_review(review)


if __name__ == '__main__':
    GitHubChatGPTPullRequestReviewer().run()
