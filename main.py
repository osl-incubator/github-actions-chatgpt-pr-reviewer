"""GitHub Actions PR Reviewer empowered by AI."""

import fnmatch
import html
import os

from typing import Any, Dict, List, Tuple, cast

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


def _is_binary_diff(diff_text: str) -> bool:
    """Return True if a diff chunk represents a binary change."""
    t = diff_text.lower()
    if 'git binary patch' in t:
        return True
    if 'binary files ' in t:
        return True
    return False


def _is_deleted_file(diff_text: str) -> bool:
    """Return True if the diff indicates a full file deletion."""
    if '\ndeleted file mode ' in diff_text:
        return True
    if '\n+++ /dev/null' in diff_text:
        return True
    return False


def _estimate_tokens(text: str) -> int:
    """Return a rough token estimate for a string."""
    return max(1, len(text) // 4)


def _model_limits(model: str) -> Tuple[int, int]:  # noqa: PLR0911, PLR0912
    """
    Return default (context_max, max_output_tokens) for known models.

    Values are based on OpenAI model specs. Unknown models fall back to
    (128_000, 16_384).
    """
    m = model.lower()

    if m.startswith('gpt-5-chat-latest'):
        return 128_000, 16_384
    if m.startswith('gpt-5'):
        return 400_000, 128_000

    if m.startswith('gpt-4.1-mini'):
        return 1_047_576, 32_768
    if m.startswith('gpt-4.1'):
        return 1_047_576, 32_768

    if m.startswith('chatgpt-4o-latest'):
        return 128_000, 16_384
    if m.startswith('gpt-4o-mini'):
        return 128_000, 16_384
    if m.startswith('gpt-4o'):
        return 400_000, 128_000

    if m.startswith('o3-mini'):
        return 200_000, 100_000
    if m.startswith('o3'):
        return 200_000, 100_000
    if m.startswith('o1-pro'):
        return 200_000, 100_000
    if m.startswith('o1'):
        return 200_000, 100_000

    if m.startswith('gpt-3.5-turbo-16k'):
        return 16_000, 4_096
    if m.startswith('gpt-3.5-turbo'):
        return 4_096, 2_048

    return 128_000, 16_384


def _model_context_tokens(model: str) -> int:
    """
    Return an approximate context window for a model.

    Values come from _model_limits and can be overridden by env.
    """
    ctx, _ = _model_limits(model)
    return ctx


def _chunk_by_lines(text: str, max_tokens: int) -> List[str]:
    """Split text into chunks that fit under max_tokens by line groups."""
    lines = text.splitlines(keepends=True)
    chunks: List[str] = []
    buf: List[str] = []
    count = 0
    for ln in lines:
        t = _estimate_tokens(ln)
        if t > max_tokens:
            if buf:
                chunks.append(''.join(buf))
                buf, count = [], 0
            max_chars = max_tokens * 4
            for i in range(0, len(ln), max_chars):
                piece = ln[i : i + max_chars]
                chunks.append(piece)
            continue
        if count + t > max_tokens and buf:
            chunks.append(''.join(buf))
            buf, count = [ln], t
        else:
            buf.append(ln)
            count += t
    if buf:
        chunks.append(''.join(buf))
    return chunks


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

        ctx_default, out_default = _model_limits(self.openai_model)

        self.openai_temperature = float(
            os.environ.get('OPENAI_TEMPERATURE', '0.5')
        )
        self.openai_max_tokens = int(
            os.environ.get('OPENAI_MAX_TOKENS', str(out_default))
        )
        self.openai_max_completion_tokens = int(
            os.environ.get(
                'OPENAI_MAX_COMPLETION_TOKENS', str(self.openai_max_tokens)
            )
        )
        self.openai_max_input_tokens = int(
            os.environ.get('OPENAI_MAX_INPUT_TOKENS', '0')
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

        self._openai: Any = OpenAI()

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
            '- If no high-impact issues, reply exactly: LGTM!\n'
            '- If you want to suggest any code to be added or changed, '
            'remember to use docstrings (just the title is required) '
            'and type annotation, '
            'when possible.\n'
            '- Point the line number where the author should apply the '
            'change inside parenthesis, e.g. (L.123).\n'
            '- Do not summarize all the changes, just focus on your '
            "suggestions to the PR's author. It should be short and "
            'concise.\n\n'
        )

        self._ctx_default = ctx_default
        self._out_default = out_default

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

    def _token_budgets(self) -> Tuple[int, int, int]:
        """
        Return (context_max, system_tokens, reply_tokens).

        context_max is derived from model or OPENAI_MAX_INPUT_TOKENS.
        """
        context = (
            self.openai_max_input_tokens
            if self.openai_max_input_tokens > 0
            else self._ctx_default
        )
        system_tokens = _estimate_tokens(self.chatgpt_initial_instruction)
        want_reason = self._want_reasoning()
        reply = (
            self.openai_max_completion_tokens
            if want_reason
            else self.openai_max_tokens
        )
        return context, system_tokens, reply

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
        repo = self.gh_api.get_repo(cast(str, self.gh_repo_name))
        _ = repo.get_pull(int(cast(str, self.gh_pr_id)))

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
                chunk = '\n'.join(bucket)
                if not _is_binary_diff(chunk):
                    files_diff[file_name] = chunk
            file_name = part.split('b/')[1].splitlines()[0]
            bucket = [part]

        if file_name and bucket and not self._is_excluded(file_name):
            chunk = '\n'.join(bucket)
            if not _is_binary_diff(chunk):
                files_diff[file_name] = chunk

        return files_diff

    def _call_openai_chat(
        self, system_text: str, user_text: str, use_completion_tokens: bool
    ) -> str:
        """
        Call OpenAI Chat Completion.

        Some models require max_completion_tokens (reasoning).
        """
        gpt_args: Dict[str, Any] = {
            'model': self.openai_model,
        }
        if use_completion_tokens:
            gpt_args['max_completion_tokens'] = (
                self.openai_max_completion_tokens
            )
        else:
            gpt_args['max_tokens'] = self.openai_max_tokens

        if not use_completion_tokens:
            gpt_args['temperature'] = self.openai_temperature

        print('GPT params:', gpt_args)

        gpt_args['messages'] = [
            {'role': 'system', 'content': system_text},
            {'role': 'user', 'content': user_text},
        ]

        completion = self._openai.chat.completions.create(**gpt_args)
        return completion.choices[0].message.content or ''

    def _call_openai_responses(self, system_text: str, user_text: str) -> str:
        """Call OpenAI Responses API for reasoning models."""
        gpt_args = dict(
            model=self.openai_model,
            reasoning={'effort': self.openai_reasoning_effort},
            max_output_tokens=self.openai_max_completion_tokens,
        )
        print('GPT params:', gpt_args)

        rsp = self._openai.responses.create(
            input=[
                {'role': 'system', 'content': system_text},
                {'role': 'user', 'content': user_text},
            ],
            **gpt_args,
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
        """Review a single file diff chunk."""
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
                except Exception:
                    return ''
            try:
                if want_reason:
                    return self._call_openai_chat(
                        sys, message_diff, use_completion_tokens=True
                    )
                return self._call_openai_responses(sys, message_diff)
            except Exception:
                raise

    def _review_file_in_chunks(
        self, filename: str, diff: str
    ) -> Tuple[str, bool]:
        """
        Review a file diff with token-aware chunking.

        Returns (combined_review, was_chunked).
        """
        ctx_max, sys_tokens, reply_tokens = self._token_budgets()
        buffer_tokens = 512
        wrapper = f'file:\n```{filename}```\ndiff:\n```'
        wrapper_end = '```'
        overhead = _estimate_tokens(wrapper) + _estimate_tokens(wrapper_end)
        budget = max(1, ctx_max - sys_tokens - reply_tokens - buffer_tokens)
        budget = max(1, budget - overhead)

        diff_tokens = _estimate_tokens(diff)
        if diff_tokens <= budget:
            msg = f'file:\n```{filename}```\ndiff:\n```{diff}```'
            return self._review_one(msg).strip(), False

        parts = _chunk_by_lines(diff, budget)
        out: List[str] = []
        total = len(parts)
        for i, part in enumerate(parts, start=1):
            hdr = f'Part {i}/{total}'
            msg = f'file:\n```{filename}```\ndiff ({hdr}):\n```{part}```'
            chunk_review = self._review_one(msg).strip()
            if chunk_review:
                out.append(f'**{hdr}**\n\n{chunk_review}')
        return '\n\n'.join(out) if out else '', True

    def pr_review(self, pr_diff: Dict[str, str]) -> List[str]:
        """Call the PR Review."""
        if not pr_diff:
            return ['LGTM! (No changes detected in diff)']

        results: List[str] = []
        for filename, diff in pr_diff.items():
            if _is_deleted_file(diff):
                results.append(
                    f'### {filename}\n\n_File deleted; no review._\n\n---'
                )
                continue

            message_diff = f'file:\n```{filename}```\ndiff:\n```{diff}```'
            print(
                'Estimated tokens:',
                int(
                    (
                        _estimate_tokens(self.chatgpt_initial_instruction)
                        + _estimate_tokens(message_diff)
                    )
                ),
            )
            try:
                content, was_chunked = self._review_file_in_chunks(
                    filename, diff
                )
                if not content:
                    content = '_No content returned by model._'

                if was_chunked:
                    note = (
                        '> Note: Too many changes in this file; the diff was '
                        'split into parts due to model limits. Consider '
                        'smaller PRs to make review easier.'
                    )
                    content = f'{note}\n\n{content}'

                results.append(f'### {filename}\n\n{content}\n\n---')
            except Exception as e:
                results.append(
                    f'### {filename}\nChatGPT was not able to review the file.'
                    f' Error: {html.escape(str(e))}'
                )
        return results

    def comment_review(self, review: List[str]) -> None:
        """Create a comment with the review content."""
        repo = self.gh_api.get_repo(cast(str, self.gh_repo_name))
        pr = repo.get_pull(int(cast(str, self.gh_pr_id)))
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
