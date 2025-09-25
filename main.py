"""GitHub Actions PR Reviewer empowered by AI."""

import fnmatch
import html
import os

from typing import Dict, List

import requests

from github import Auth, Github
from openai import OpenAI


def _as_bool(val: str | None, default: bool = False) -> bool:
    """Parse a boolean-like string."""
    if val is None:
        return default
    return str(val).strip().lower() in {'1', 'true', 'yes', 'on'}


def _is_reasoning_model(model: str) -> bool:
    """
    Check if the model has reasoning feature.

    Heuristic: o*-series and GPT-5* are reasoning-capable and typically
    require max_completion_tokens (Chat) or max_output_tokens (Responses).
    """
    m = model.lower()
    return m.startswith(('o1', 'o2', 'o3', 'o4', 'o-')) or m.startswith(
        'gpt-5'
    )


def _split_globs(raw: str) -> List[str]:
    """Split a raw patterns string into a clean list of glob patterns."""
    if not raw:
        return []
    parts: List[str] = []
    for chunk in raw.replace('\r', '\n').replace(';', '\n').split('\n'):
        for sub in chunk.split(','):
            pat = sub.strip()
            if pat:
                parts.append(pat)
    return parts


class GitHubChatGPTPullRequestReviewer:
    """GitHubChatGPTPullRequestReviewer class."""

    def __init__(self) -> None:
        """Initialize the class."""
        self.exclude_globs: List[str] = []
        self._config_gh()
        self._config_openai()

    def _config_gh(self) -> None:
        """Configure GitHub context."""
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

        self.exclude_globs = _split_globs(
            os.environ.get('EXCLUDE_PATH', '').strip()
        )

    def _config_openai(self) -> None:
        """Configure OpenAI client and prompts."""
        self.openai_model = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
        self.openai_temperature = float(
            os.environ.get('OPENAI_TEMPERATURE', '0.5')
        )
        self.openai_max_tokens = int(
            os.environ.get('OPENAI_MAX_TOKENS', '2048')
        )
        self.openai_max_completion_tokens = int(
            os.environ.get(
                'OPENAI_MAX_COMPLETION_TOKENS', str(self.openai_max_tokens)
            )
        )

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
            'You are a GitHub PR reviewer bot. You will receive a PR diff. '
            'Write a short, high-signal review that focuses only on material '
            'risks.\n\n'
            'Prioritize, in order:\n'
            '- correctness / logic bugs\n'
            '- security and unsafe patterns\n'
            '- performance regressions with real impact\n'
            '- breaking API / behavior changes\n'
            '- maintainability that affects future changes\n'
            f'{extra_criteria}\n'
            'Ignore minor style or preference nits unless they hide a bug or '
            'the fix is a one-liner.\n\n'
            'Constraints:\n'
            '- Do not restate the diff or comment on formatting-only '
            'changes.\n'
            '- Keep the whole review under ~250 words per file.\n'
            '- If no high-impact issues, reply exactly: LGTM!\n\n'
            'Return Markdown only.'
        )

    def _want_reasoning(self) -> bool:
        """Return True if reasoning mode should be used."""
        if self.openai_reasoning_mode == 'on':
            return True
        if self.openai_reasoning_mode == 'off':
            return False
        return _is_reasoning_model(self.openai_model)

    def _prepare_extra_criteria(self, extra_criteria: str) -> str:
        """Format extra criteria lines as markdown bullets."""
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

    def _is_excluded(self, filename: str) -> bool:
        """Return True if filename matches any exclude pattern."""
        if not self.exclude_globs:
            return False
        return any(
            fnmatch.fnmatch(filename, pat) for pat in self.exclude_globs
        )

    def get_pr_content(self) -> str:
        """Get the PR content."""
        resp = requests.get(
            self.gh_pr_url, headers=self.gh_headers, timeout=60
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f'GitHub API error {resp.status_code}: {resp.text}'
            )
        return resp.text or ''

    def get_diff(self) -> Dict[str, str]:
        """Get Diff content as a mapping of filename to diff text."""
        _ = self.gh_api.get_repo(self.gh_repo_name).get_pull(
            int(self.gh_pr_id)
        )

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
            if file_name and bucket and not self._is_excluded(file_name):
                files_diff[file_name] = '\n'.join(bucket)
            file_name = part.split('b/')[1].splitlines()[0]
            bucket = [part]

        if file_name and bucket and not self._is_excluded(file_name):
            files_diff[file_name] = '\n'.join(bucket)

        return files_diff

    def _call_openai_chat(
        self, system_text: str, user_text: str, use_completion_tokens: bool
    ) -> str:
        """
        Call OpenAI Chat Completion.

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
            kwargs['max_completion_tokens'] = self.openai_max_completion_tokens
        else:
            kwargs['max_tokens'] = self.openai_max_tokens

        if not use_completion_tokens:
            kwargs['temperature'] = self.openai_temperature

        completion = self._openai.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ''

    def _call_openai_responses(self, system_text: str, user_text: str) -> str:
        """Call OpenAI Responses API for reasoning models."""
        rsp = self._openai.responses.create(
            model=self.openai_model,
            reasoning={'effort': self.openai_reasoning_effort},
            max_output_tokens=self.openai_max_completion_tokens,
            input=[
                {'role': 'system', 'content': system_text},
                {'role': 'user', 'content': user_text},
            ],
        )
        text = getattr(rsp, 'output_text', None)
        if not text:
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
        """Review a single file diff."""
        sys = self.chatgpt_initial_instruction
        want_reason = self._want_reasoning()

        try:
            if want_reason:
                return self._call_openai_responses(sys, message_diff)
            return self._call_openai_chat(
                sys, message_diff, use_completion_tokens=False
            )
        except Exception as e:
            msg = str(e)
            if 'max_tokens' in msg and 'max_completion_tokens' in msg:
                try:
                    return self._call_openai_chat(
                        sys, message_diff, use_completion_tokens=True
                    )
                except Exception as e_openai:
                    print('WARNING:\n', e_openai)
            try:
                if want_reason:
                    return self._call_openai_chat(
                        sys, message_diff, use_completion_tokens=True
                    )
                return self._call_openai_responses(sys, message_diff)
            except Exception as e_openai:
                print('WARNING:\n', e_openai)
                raise

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
        pr = repo.get_pull(int(self.gh_pr_id))
        comment = (
            '# OSL ChatGPT Reviewer\n\n'
            '*NOTE: This is generated by an AI program, so some '
            'comments may not make sense.*\n\n'
        ) + '\n'.join(review)
        try:
            pr.create_issue_comment(comment)
        except Exception as e:
            print(f'[WARN] Could not post PR comment: {e}')

    def run(self) -> None:
        """Run the PR Reviewer."""
        pr_diff = self.get_diff()
        review = self.pr_review(pr_diff)
        self.comment_review(review)


if __name__ == '__main__':
    GitHubChatGPTPullRequestReviewer().run()
