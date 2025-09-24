"""GitHub Actions PR Reviewer empowered by AI."""

import html
import os

from typing import Dict, List

import requests

from github import Auth, Github
from openai import OpenAI


class GitHubChatGPTPullRequestReviewer:
    """GitHub Actions PR Reviewer empowered by AI."""

    def __init__(self) -> None:
        """Initialize class."""
        self._config_gh()
        self._config_openai()

    def _config_gh(self) -> None:
        # Inputs from GitHub Actions env
        self.gh_pr_id = os.environ.get('GITHUB_PR_ID')
        if not self.gh_pr_id:
            raise RuntimeError('GITHUB_PR_ID is required')

        gh_token = os.environ.get('GITHUB_TOKEN')
        if not gh_token:
            raise RuntimeError('GITHUB_TOKEN is required')

        self.gh_repo_name = os.environ.get('GITHUB_REPOSITORY')
        if not self.gh_repo_name:
            raise RuntimeError('GITHUB_REPOSITORY is required')

        # Build diff URL (accept header returns raw diff from the PR resource)
        gh_api_url = os.environ.get('GITHUB_API_URL', 'https://api.github.com')
        self.gh_pr_url = (
            f'{gh_api_url}/repos/{self.gh_repo_name}/pulls/{self.gh_pr_id}'
        )
        self.gh_headers = {
            'Authorization': f'token {gh_token}',
            'Accept': 'application/vnd.github.v3.diff',
        }

        # PyGithub v2 uses Auth.Token
        auth = Auth.Token(gh_token)
        self.gh_api = Github(auth=auth)

    def _config_openai(self) -> None:
        # Defaults can be overridden via env vars
        self.openai_model = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
        self.openai_temperature = float(
            os.environ.get('OPENAI_TEMPERATURE', '0.5')
        )
        self.openai_max_tokens = int(
            os.environ.get('OPENAI_MAX_TOKENS', '2048')
        )
        self.openai_extra_criteria = os.environ.get(
            'OPENAI_EXTRA_CRITERIA', ''
        ).strip()

        # SDK v1 client (uses OPENAI_API_KEY from env)
        self._openai = OpenAI()

        self.chatgpt_initial_instruction = (
            'You are a GitHub PR reviewer bot. You will receive a text '
            'containing the diff from a PR with all proposed changes. '
            'Analyze it and return suggestions '
            'to improve or fix issues, using the following criteria:\n'
            '- best practices that would improve the changes\n'
            '- code style formatting\n'
            '- recommendations specific to the programming language\n'
            '- performance improvements\n'
            '- security improvements\n'
            '- improvements from the software engineering perspective\n'
            '- docstrings, when applicable\n'
            '- variable with meaningful names, when applicable\n'
            '- avoid unnecessary code comments\n'
            '- performance improvements\n'
            '- prefer explicit over implicit (e.g., in Python, avoid '
            '`from x import *`)\n'
            f'{self._prepare_extra_criteria(self.openai_extra_criteria)}\n'
            'Return your response in Markdown. If everything looks good, '
            'just say: "LGTM!"'
        )

    def _prepare_extra_criteria(self, extra_criteria: str) -> str:
        if not extra_criteria:
            return ''
        lines = []
        for item in extra_criteria.split(';'):
            _item = item.strip()
            if not _item:
                continue
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
        """Return a mapping {filename: diff_text_for_that_file}."""
        # Touch PR to ensure it exists (also used later to comment)
        _ = self.gh_api.get_repo(self.gh_repo_name).get_pull(
            int(self.gh_pr_id)
        )

        content = self.get_pr_content()
        if not content.strip():
            return {}

        # The diff is a sequence of "diff --git a/... b/..." sections.
        parts = content.split('diff')
        files_diff: Dict[str, str] = {}
        bucket: List[str] = []
        file_name = ''

        for part in parts:
            if not part:
                continue

            if not part.startswith(' --git a/'):
                # Continuation of current file's diff
                if bucket is not None:
                    bucket.append(part)
                continue

            # New file section starts — flush previous
            if file_name and bucket:
                files_diff[file_name] = '\n'.join(bucket)

            # Extract file path after "b/"
            file_name = part.split('b/')[1].splitlines()[0]
            bucket = [part]

        if file_name and bucket:
            files_diff[file_name] = '\n'.join(bucket)

        return files_diff

    def pr_review(self, pr_diff: Dict[str, str]) -> List[str]:
        """Generate review from PR content."""
        if not pr_diff:
            return ['LGTM! (No changes detected in diff)']

        system_message = {
            'role': 'system',
            'content': self.chatgpt_initial_instruction,
        }
        results: List[str] = []

        for filename, diff in pr_diff.items():
            message_diff = f'file:\n```{filename}```\ndiff:\n```{diff}```'
            messages = [
                system_message,
                {'role': 'user', 'content': message_diff},
            ]

            # crude token estimate for logging
            print(
                'Estimated tokens:',
                int(len(self.chatgpt_initial_instruction + message_diff) / 4),
            )

            try:
                completion = self._openai.chat.completions.create(
                    model=self.openai_model,
                    temperature=self.openai_temperature,
                    max_tokens=self.openai_max_tokens,
                    messages=messages,
                )
                content = completion.choices[0].message.content or ''
                results.append(f'### {filename}\n\n{content}\n\n---')
            except Exception as e:
                results.append(
                    f'### {filename}\nChatGPT was not able to review the file.'
                    f' Error: {html.escape(str(e))}'
                )

        return results

    def comment_review(self, review: List[str]) -> None:
        """Add a comment to the PR with the review content."""
        repo = self.gh_api.get_repo(self.gh_repo_name)
        pull_request = repo.get_pull(int(self.gh_pr_id))
        comment = '# OSL ChatGPT Reviewer\n\n*NOTE...*\n\n' + '\n'.join(review)
        try:
            pull_request.create_issue_comment(comment)
        except Exception as e:
            print(f'[WARN] Could not post PR comment: {e}')
            print(comment)

    def run(self) -> None:
        """Run the PR reviewer function."""
        pr_diff = self.get_diff()
        review = self.pr_review(pr_diff)
        self.comment_review(review)


if __name__ == '__main__':
    reviewer = GitHubChatGPTPullRequestReviewer()
    reviewer.run()
