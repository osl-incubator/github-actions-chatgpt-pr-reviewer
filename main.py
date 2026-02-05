"""GitHub Actions PR Reviewer empowered by AI."""

from __future__ import annotations

import fnmatch
import html
import logging
import os
import re

from typing import Any, cast

import requests

from github import Auth, Github
from openai import OpenAI

# Minimum openai SDK version that supports Responses API
_MIN_OPENAI_VERSION = '1.0.0'


def _check_openai_sdk() -> tuple[bool, str]:
    """
    Check if the openai SDK version supports required features.

    Returns
    -------
    tuple[bool, str]
        (has_responses_api, version_string).
    """
    try:
        import openai  # noqa: PLC0415

        version = getattr(openai, '__version__', '0.0.0')
        client = OpenAI()
        has_responses = hasattr(client, 'responses')
        return has_responses, version
    except Exception:
        return False, 'unknown'


def _is_reasoning_model(model: str) -> bool:
    """
    Return True if the model name suggests reasoning capability.

    Parameters
    ----------
    model:
        Model name.

    Returns
    -------
    bool
        True if the model is likely reasoning-capable.
    """
    m = model.lower()
    return m.startswith(('o1', 'o2', 'o3', 'o4', 'o-')) or m.startswith(
        'gpt-5'
    )


def _split_globs(raw: str) -> list[str]:
    """
    Split a raw patterns string into a clean list of glob patterns.

    Parameters
    ----------
    raw:
        Raw glob patterns separated by commas/semicolons/newlines.

    Returns
    -------
    list[str]
        Cleaned patterns.
    """
    if not raw:
        return []
    parts: list[str] = []
    for chunk in raw.replace('\r', '\n').replace(';', '\n').split('\n'):
        for sub in chunk.split(','):
            pat = sub.strip()
            if pat:
                parts.append(pat)
    return parts


def _is_binary_diff(diff_text: str) -> bool:
    """
    Return True if a diff chunk represents a binary change.

    Parameters
    ----------
    diff_text:
        Unified diff text.

    Returns
    -------
    bool
        True if it looks like a binary diff.
    """
    t = diff_text.lower()
    return 'git binary patch' in t or 'binary files ' in t


def _is_deleted_file(diff_text: str) -> bool:
    """
    Return True if the diff indicates a full file deletion.

    Parameters
    ----------
    diff_text:
        Unified diff text.

    Returns
    -------
    bool
        True if the file was deleted.
    """
    return (
        '\ndeleted file mode ' in diff_text or '\n+++ /dev/null' in diff_text
    )


def _estimate_tokens(text: str, chars_per_token: float = 3.5) -> int:
    """
    Return a rough token estimate for a string.

    Uses a slightly more conservative estimate for code (3.5 chars/token)
    compared to prose (4 chars/token).

    Parameters
    ----------
    text:
        Input text.
    chars_per_token:
        Average characters per token (default 3.5 for code).

    Returns
    -------
    int
        Approximate token count.
    """
    return max(1, int(len(text) / chars_per_token))


def _model_limits(model: str) -> tuple[int, int]:
    """
    Return default (context_max, max_output_tokens) for known models.

    Parameters
    ----------
    model:
        Model name.

    Returns
    -------
    tuple[int, int]
        (context_max, max_output_tokens). Unknown models fall back to
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


def _chunk_by_lines(text: str, max_tokens: int) -> list[str]:
    """
    Split text into chunks that fit under max_tokens by line groups.

    Parameters
    ----------
    text:
        Input text to chunk.
    max_tokens:
        Per-chunk approximate token budget.

    Returns
    -------
    list[str]
        Chunked text pieces.
    """
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    buf: list[str] = []
    count = 0

    for ln in lines:
        t = _estimate_tokens(ln)
        if t > max_tokens:
            if buf:
                chunks.append(''.join(buf))
                buf, count = [], 0
            max_chars = int(max_tokens * 3.5)
            for i in range(0, len(ln), max_chars):
                chunks.append(ln[i : i + max_chars])
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


class RedactingFormatter(logging.Formatter):
    """
    Formatter that redacts sensitive data via regex substitutions.

    Parameters
    ----------
    fmt:
        Log format string.
    patterns:
        (pattern, replacement) pairs applied sequentially.
    """

    def __init__(
        self,
        fmt: str,
        patterns: list[tuple[re.Pattern[str], str]],
    ) -> None:
        super().__init__(fmt)
        self._patterns = patterns

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log message and apply redactions.

        Parameters
        ----------
        record:
            Log record.

        Returns
        -------
        str
            Redacted message.
        """
        msg = super().format(record)
        for pat, repl in self._patterns:
            msg = pat.sub(repl, msg)
        return msg


def _redaction_patterns() -> list[tuple[re.Pattern[str], str]]:
    """
    Return regex patterns used to redact sensitive info in logs.

    Returns
    -------
    list[tuple[re.Pattern[str], str]]
        Redaction patterns.
    """
    return [
        (
            re.compile(r'(?m)^(.*?\bRequest options:\s*)(.*)$'),
            r'\1[REDACTED]',
        ),
        (
            re.compile(
                r"(['\"]json_data['\"]\s*:\s*)\{(?:.|\n)*?\}",
                re.I | re.S,
            ),
            r'\1[REDACTED]',
        ),
        (
            re.compile(
                r"(['\"]input['\"]\s*:\s*)\[(?:.|\n)*?\]",
                re.I | re.S,
            ),
            r'\1[REDACTED]',
        ),
        (
            re.compile(
                r"(['\"]messages['\"]\s*:\s*)\[(?:.|\n)*?\]",
                re.I | re.S,
            ),
            r'\1[REDACTED]',
        ),
        (re.compile(r'```.*?```', re.S), '```[REDACTED]```'),
        (
            re.compile(r'(?im)^set-cookie:.*$', re.I | re.M),
            'Set-Cookie: [REDACTED]',
        ),
        (re.compile(r"('set-cookie'\s*,\s*)'[^']*'", re.I), r"\1'[REDACTED]'"),
        (
            re.compile(
                r"('openai-(?:organization|project)'\s*,\s*)'[^']*'",
                re.I,
            ),
            r"\1'[REDACTED]'",
        ),
        (
            re.compile(r'(?im)^(authorization\s*[:=]\s*)([\'"]?)(.*)$'),
            r'\1\2[REDACTED]\2',
        ),
        (
            re.compile(r'(?im)^(api[_-]?key\s*[:=]\s*)([\'"]?)(.*)$'),
            r'\1\2[REDACTED]\2',
        ),
        (
            re.compile(r'(?im)^(idempotency_key\s*[:=]\s*)([\'"]?)(.*)$'),
            r'\1\2[REDACTED]\2',
        ),
    ]


def _parse_unified_diff(content: str) -> dict[str, str]:
    """
    Parse unified diff text into per-file chunks.

    Parameters
    ----------
    content:
        Full diff payload.

    Returns
    -------
    dict[str, str]
        Mapping filename -> diff text chunk.
    """
    if not content.strip():
        return {}

    files: dict[str, str] = {}
    current_file: str | None = None
    bucket: list[str] = []

    lines = content.splitlines(keepends=True)
    for ln in lines:
        if ln.startswith('diff --git a/'):
            if current_file is not None and bucket:
                files[current_file] = ''.join(bucket)
            bucket = [ln]
            current_file = None
            continue

        if current_file is None and ln.startswith('+++ '):
            if ln.startswith('+++ b/'):
                current_file = ln[len('+++ b/') :].strip()
            elif ln.startswith('+++ /dev/null'):
                current_file = '__DELETED__'
            bucket.append(ln)
            continue

        bucket.append(ln)

    if current_file is not None and bucket:
        files[current_file] = ''.join(bucket)

    cleaned: dict[str, str] = {}
    for name, diff in files.items():
        if name != '__DELETED__':
            cleaned[name] = diff
        else:
            guessed = _guess_deleted_filename(diff)
            if guessed:
                cleaned[guessed] = diff
    return cleaned


def _guess_deleted_filename(diff_text: str) -> str | None:
    """
    Try to recover the filename for deleted files from the diff.

    Parameters
    ----------
    diff_text:
        Unified diff.

    Returns
    -------
    str | None
        Filename if found.
    """
    for ln in diff_text.splitlines():
        if ln.startswith('--- a/'):
            return ln[len('--- a/') :].strip()
    return None


class GitHubChatGPTPullRequestReviewer:
    """Review GitHub PR diffs and post a concise, high-signal comment."""

    # Valid reasoning effort levels (verified against OpenAI API docs)
    VALID_REASONING_EFFORTS = {'none', 'low', 'medium', 'high'}

    def __init__(self) -> None:
        self.exclude_globs: list[str] = []
        self._setup_logging()
        self._check_sdk_compatibility()
        self._config_gh()
        self._config_openai()

    def _setup_logging(self) -> None:
        """Configure the logger with redaction."""
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
        fmt = '%(levelname)s %(message)s'
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format=fmt,
        )
        self._log = logging.getLogger(__name__)

        patterns = _redaction_patterns()
        root = logging.getLogger()
        for h in root.handlers:
            h.setFormatter(RedactingFormatter(fmt, patterns))

    def _check_sdk_compatibility(self) -> None:
        """Check and log SDK compatibility status."""
        has_responses, version = _check_openai_sdk()
        self._has_responses_api = has_responses
        self._log.info('OpenAI SDK version: %s', version)
        if not has_responses:
            self._log.warning(
                'OpenAI SDK does not support Responses API. '
                'Reasoning models will use Chat Completions fallback. '
                'Consider upgrading: pip install --upgrade openai'
            )

    def _log_chat_meta(self, obj: Any) -> None:
        """
        Log minimal metadata for Chat Completions.

        Parameters
        ----------
        obj:
            Chat completion response object.
        """
        try:
            usage = getattr(obj, 'usage', None)
            if usage:
                self._log.debug(
                    'Chat usage: prompt=%s completion=%s total=%s',
                    getattr(usage, 'prompt_tokens', None),
                    getattr(usage, 'completion_tokens', None),
                    getattr(usage, 'total_tokens', None),
                )
            choices = list(getattr(obj, 'choices', []) or [])
            finish = [getattr(c, 'finish_reason', None) for c in choices]
            self._log.debug('Chat finish_reasons: %s', finish)
            # Log tool call presence for first choice
            if choices:
                first_msg = getattr(choices[0], 'message', None)
                has_tools = bool(getattr(first_msg, 'tool_calls', None))
                self._log.debug('Chat first choice has_tools=%s', has_tools)
        except Exception as exc:
            self._log.debug('Failed to log chat meta: %s', exc)

    def _log_responses_meta(self, rsp: Any) -> None:
        """
        Log minimal metadata for Responses API.

        Parameters
        ----------
        rsp:
            Responses API object.
        """
        try:
            usage = getattr(rsp, 'usage', None)
            if usage:
                self._log.debug(
                    'Resp usage: input=%s output=%s total=%s',
                    getattr(usage, 'input_tokens', None),
                    getattr(usage, 'output_tokens', None),
                    getattr(usage, 'total_tokens', None),
                )
            self._log.debug(
                'Resp status=%s id=%s',
                getattr(rsp, 'status', None),
                getattr(rsp, 'id', None),
            )
            # Log output item types or None
            output = getattr(rsp, 'output', None)
            if output is None:
                self._log.debug('Resp output is None')
            else:
                item_types = [getattr(item, 'type', None) for item in output]
                self._log.debug('Resp output item types: %s', item_types)
        except Exception as exc:
            self._log.debug('Failed to log response meta: %s', exc)

    def _config_gh(self) -> None:
        """Configure GitHub context from environment variables."""
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

    def _default_reasoning_effort(self) -> str:
        """
        Choose a conservative default reasoning effort.

        Returns
        -------
        str
            Effort string.
        """
        if self.openai_model.lower().startswith('gpt-5'):
            return 'none'
        return 'low'

    def _normalize_reasoning_effort(self, effort: str) -> str:
        """
        Validate and normalize reasoning effort.

        Parameters
        ----------
        effort:
            Raw effort value.

        Returns
        -------
        str
            Normalized effort.
        """
        e = (effort or '').strip().lower()
        if e in self.VALID_REASONING_EFFORTS:
            return e
        fallback = self._default_reasoning_effort()
        self._log.warning(
            'Invalid OPENAI_REASONING_EFFORT="%s"; valid values are %s. '
            'Using "%s".',
            effort,
            sorted(self.VALID_REASONING_EFFORTS),
            fallback,
        )
        return fallback

    def _config_openai(self) -> None:
        """Configure OpenAI client and prompt settings."""
        self.openai_model = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')

        ctx_default, out_default = _model_limits(self.openai_model)
        self._ctx_default = ctx_default
        self._out_default = out_default

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
            self._log.warning(
                'Invalid OPENAI_REASONING="%s"; using "auto".',
                reasoning_mode_env,
            )
            reasoning_mode_env = 'auto'
        self.openai_reasoning_mode = reasoning_mode_env

        effort_env = os.environ.get(
            'OPENAI_REASONING_EFFORT', self._default_reasoning_effort()
        )
        self.openai_reasoning_effort = self._normalize_reasoning_effort(
            effort_env
        )

        self._openai: Any = OpenAI()

        extra_criteria = self._prepare_extra_criteria(
            os.environ.get('PROMPT_EXTRA_CRITERIA', '').strip()
        )
        prompt_project_intro = os.environ.get(
            'PROMPT_PROJECT_INTRODUCTION',
            '',
        ).strip()
        if prompt_project_intro:
            prompt_project_intro = f'{prompt_project_intro}\n\n'

        self.chatgpt_initial_instruction = (
            'You are a GitHub PR reviewer bot. You will receive a PR diff. '
            'Write a short, high-signal review that focuses only on material '
            'risks.\n\n'
            f'{prompt_project_intro}'
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
            'and type annotation, when possible.\n'
            '- Point the line number where the author should apply the '
            'change inside parenthesis, e.g. (L.123).\n'
            '- Do not summarize all the changes; focus only on actionable '
            'suggestions.\n'
            '- Be as short, concise, and objective as possible.\n\n'
        )

        if (
            self._want_reasoning()
            and self.openai_max_completion_tokens < self._out_default
        ):
            self._log.warning(
                'Configured max_output_tokens=%s is below model default=%s; '
                'reviews may truncate.',
                self.openai_max_completion_tokens,
                self._out_default,
            )

    def _is_reasoning_model_selected(self) -> bool:
        """
        Return True if the current model is a reasoning model.

        Returns
        -------
        bool
            Whether the selected model supports reasoning.
        """
        return _is_reasoning_model(self.openai_model)

    def _want_reasoning(self) -> bool:
        """
        Return True if reasoning mode should be used.

        Returns
        -------
        bool
            Whether to use reasoning mode.
        """
        supported = self._is_reasoning_model_selected()

        if self.openai_reasoning_mode == 'off':
            return False

        if self.openai_reasoning_mode == 'on' and not supported:
            self._log.warning(
                'OPENAI_REASONING="on" but model "%s" is not recognized as a '
                'reasoning model; disabling reasoning.',
                self.openai_model,
            )
            return False

        if self.openai_reasoning_mode == 'on':
            return True

        # auto mode
        return supported

    def _can_use_responses_api(self) -> bool:
        """
        Return True if Responses API is available and should be used.

        Returns
        -------
        bool
            Whether Responses API can be used.
        """
        return self._has_responses_api and self._want_reasoning()

    def _prepare_extra_criteria(self, extra_criteria: str) -> str:
        """
        Format extra criteria lines as markdown bullets.

        Parameters
        ----------
        extra_criteria:
            Semi-colon separated criteria.

        Returns
        -------
        str
            Markdown bullet list.
        """
        if not extra_criteria:
            return ''
        lines: list[str] = []
        for item in extra_criteria.split(';'):
            _item = item.strip()
            if _item:
                if not _item.startswith('-'):
                    _item = '- ' + _item
                lines.append(_item)
        return '\n'.join(lines)

    def _is_excluded(self, filename: str) -> bool:
        """
        Return True if filename matches any exclude pattern.

        Parameters
        ----------
        filename:
            Path to test.

        Returns
        -------
        bool
            Whether the file should be excluded.
        """
        if not self.exclude_globs:
            return False
        return any(
            fnmatch.fnmatch(filename, pat) for pat in self.exclude_globs
        )

    def _token_budgets(self) -> tuple[int, int, int]:
        """
        Return (context_max, system_tokens, reply_tokens).

        Returns
        -------
        tuple[int, int, int]
            (context_max, system_tokens, reply_tokens).
        """
        context = (
            self.openai_max_input_tokens
            if self.openai_max_input_tokens > 0
            else self._ctx_default
        )
        system_tokens = _estimate_tokens(self.chatgpt_initial_instruction)
        reply = (
            self.openai_max_completion_tokens
            if self._want_reasoning()
            else self.openai_max_tokens
        )
        self._log.debug(
            'Budgets resolved: context=%s system=%s reply=%s (reasoning=%s)',
            context,
            system_tokens,
            reply,
            self._want_reasoning(),
        )
        return context, system_tokens, reply

    def get_pr_content(self) -> str:
        """
        Fetch PR diff content.

        Returns
        -------
        str
            Raw diff text.
        """
        try:
            resp = requests.get(
                self.gh_pr_url, headers=self.gh_headers, timeout=60
            )
        except Exception as exc:
            self._log.exception('GitHub request failed: %s', exc)
            raise
        if resp.status_code != 200:
            self._log.error(
                'GitHub API error %s: %s', resp.status_code, resp.text[:500]
            )
            raise RuntimeError(
                f'GitHub API error {resp.status_code}: {resp.text}'
            )
        return resp.text or ''

    def get_diff(self) -> dict[str, str]:
        """
        Get diff content as a mapping of filename -> diff chunk.

        Returns
        -------
        dict[str, str]
            Per-file unified diffs (binary excluded).
        """
        repo = self.gh_api.get_repo(cast(str, self.gh_repo_name))
        _ = repo.get_pull(int(cast(str, self.gh_pr_id)))

        content = self.get_pr_content()
        parsed = _parse_unified_diff(content)

        files_diff: dict[str, str] = {}
        for file_name, chunk in parsed.items():
            if self._is_excluded(file_name):
                continue
            if _is_binary_diff(chunk):
                continue
            files_diff[file_name] = chunk
        return files_diff

    def _call_openai_chat(
        self,
        system_text: str,
        user_text: str,
        *,
        use_completion_tokens: bool = False,
    ) -> str:
        """
        Call OpenAI Chat Completions API.

        Parameters
        ----------
        system_text:
            System instruction.
        user_text:
            User message content.
        use_completion_tokens:
            Use max_completion_tokens instead of max_tokens (required for
            reasoning models).

        Returns
        -------
        str
            Model response text.
        """
        is_reasoning = self._is_reasoning_model_selected()

        gpt_args: dict[str, Any] = {'model': self.openai_model}

        # Reasoning models require max_completion_tokens and don't support
        # temperature
        if is_reasoning or use_completion_tokens:
            gpt_args['max_completion_tokens'] = (
                self.openai_max_completion_tokens
            )
        else:
            gpt_args['max_tokens'] = self.openai_max_tokens
            gpt_args['temperature'] = self.openai_temperature

        # For reasoning models, convert system message to user context
        # Some reasoning models handle system prompts differently
        if is_reasoning:
            # Prepend system instruction to user message for better
            # compatibility
            combined_user = (
                f'<instructions>\n{system_text}</instructions>\n\n{user_text}'
            )
            gpt_args['messages'] = [
                {'role': 'user', 'content': combined_user},
            ]
            self._log.debug(
                'Using combined user message for reasoning model (no separate '
                'system prompt)'
            )
        else:
            gpt_args['messages'] = [
                {'role': 'system', 'content': system_text},
                {'role': 'user', 'content': user_text},
            ]

        self._log.info(
            'Chat API call: model=%s reasoning=%s use_completion_tokens=%s',
            self.openai_model,
            is_reasoning,
            is_reasoning or use_completion_tokens,
        )
        self._log.debug(
            'GPT params (excluding messages): %s',
            {k: v for k, v in gpt_args.items() if k != 'messages'},
        )

        try:
            completion = self._openai.chat.completions.create(**gpt_args)
        except Exception as exc:
            self._log.exception('Chat completion failed: %s', exc)
            raise

        self._log_chat_meta(completion)
        return completion.choices[0].message.content or ''

    def _responses_create(
        self,
        system_text: str,
        user_text: str,
        *,
        max_output_tokens: int,
        include_reasoning: bool,
    ) -> Any:
        """
        Create a Responses API request.

        Parameters
        ----------
        system_text:
            System instruction.
        user_text:
            User message content.
        max_output_tokens:
            Token limit for output (includes reasoning + visible text).
        include_reasoning:
            Whether to pass reasoning controls.

        Returns
        -------
        Any
            Responses API object.
        """
        gpt_args: dict[str, Any] = {
            'model': self.openai_model,
            'max_output_tokens': max_output_tokens,
        }
        if include_reasoning and self.openai_reasoning_effort != 'none':
            gpt_args['reasoning'] = {'effort': self.openai_reasoning_effort}

        self._log.info(
            'Responses API call: model=%s reasoning_effort=%s',
            self.openai_model,
            self.openai_reasoning_effort if include_reasoning else 'disabled',
        )
        self._log.debug('GPT params: %s', gpt_args)

        return self._openai.responses.create(
            input=[
                {'role': 'system', 'content': system_text},
                {'role': 'user', 'content': user_text},
            ],
            **gpt_args,
        )

    def _responses_text(self, rsp: Any) -> str:
        """
        Extract visible text from a Responses API object.

        Parameters
        ----------
        rsp:
            Responses API object.

        Returns
        -------
        str
            Visible output text (may be empty).
        """
        text = getattr(rsp, 'output_text', '') or ''
        if text:
            return text

        pieces: list[str] = []
        try:
            for item in getattr(rsp, 'output', None) or []:
                for block in getattr(item, 'content', None) or []:
                    s = getattr(block, 'text', '')
                    if s:
                        pieces.append(s)
        except Exception as exc:
            self._log.debug('Failed to parse Responses output blocks: %s', exc)
        return ''.join(pieces)

    def _is_incomplete_due_to_tokens(self, rsp: Any) -> bool:
        """
        Return True if the response looks incomplete due to token limits.

        Parameters
        ----------
        rsp:
            Responses API object.

        Returns
        -------
        bool
            True if incomplete and likely token-limited.
        """
        status = (getattr(rsp, 'status', '') or '').lower()
        if status != 'incomplete':
            return False

        details = getattr(rsp, 'incomplete_details', None)
        if details is None:
            return True

        reason = getattr(details, 'reason', None)
        if reason is None and isinstance(details, dict):
            reason = details.get('reason')
        if reason is None:
            return True

        return str(reason).lower() in {'max_output_tokens', 'length'}

    def _call_openai_responses(self, system_text: str, user_text: str) -> str:
        """
        Call OpenAI Responses API with safe reasoning controls.

        Parameters
        ----------
        system_text:
            System instruction.
        user_text:
            User message content.

        Returns
        -------
        str
            Model response text.
        """
        include_reasoning = self._want_reasoning()
        try:
            rsp = self._responses_create(
                system_text,
                user_text,
                max_output_tokens=self.openai_max_completion_tokens,
                include_reasoning=include_reasoning,
            )
        except Exception as exc:
            self._log.exception('Responses API call failed: %s', exc)
            raise

        self._log_responses_meta(rsp)
        text = self._responses_text(rsp).strip()

        if text and not self._is_incomplete_due_to_tokens(rsp):
            return text

        can_retry = self.openai_max_completion_tokens < self._out_default
        if self._is_incomplete_due_to_tokens(rsp) and can_retry:
            bumped = min(
                self._out_default, self.openai_max_completion_tokens * 2
            )
            self._log.warning(
                'Incomplete response; retrying with max_output_tokens=%s',
                bumped,
            )
            rsp2 = self._responses_create(
                system_text,
                user_text,
                max_output_tokens=bumped,
                include_reasoning=include_reasoning,
            )
            self._log_responses_meta(rsp2)
            text2 = self._responses_text(rsp2).strip()
            if text2:
                return text2

        note = (
            '_Model output was empty or incomplete. Consider increasing '
            'OPENAI_MAX_COMPLETION_TOKENS or splitting the diff._'
        )
        return note

    def _review_one(self, message_diff: str) -> str:
        """
        Review a single file diff chunk.

        Parameters
        ----------
        message_diff:
            Wrapped diff message.

        Returns
        -------
        str
            Review text.
        """
        sys = self.chatgpt_initial_instruction

        # Determine primary and fallback strategies
        use_responses = self._can_use_responses_api()
        is_reasoning = self._is_reasoning_model_selected()

        self._log.debug(
            'Review strategy: use_responses=%s is_reasoning=%s',
            use_responses,
            is_reasoning,
        )

        try:
            if use_responses:
                return self._call_openai_responses(sys, message_diff)
            return self._call_openai_chat(
                sys,
                message_diff,
                use_completion_tokens=is_reasoning,
            )
        except Exception as exc:
            msg = str(exc)
            self._log.exception('Primary review attempt failed: %s', msg)

            # Handle specific error about max_tokens vs max_completion_tokens
            if 'max_tokens' in msg and 'max_completion_tokens' in msg:
                self._log.info(
                    'Retrying with max_completion_tokens due to API hint'
                )
                return self._call_openai_chat(
                    sys,
                    message_diff,
                    use_completion_tokens=True,
                )

            # Try alternate API
            if use_responses:
                self._log.info('Falling back to Chat Completions API')
                return self._call_openai_chat(
                    sys,
                    message_diff,
                    use_completion_tokens=is_reasoning,
                )
            elif self._has_responses_api:
                self._log.info('Falling back to Responses API')
                return self._call_openai_responses(sys, message_diff)
            else:
                raise

    def _review_file_in_chunks(
        self,
        filename: str,
        diff: str,
    ) -> tuple[str, bool]:
        """
        Review a file diff with token-aware chunking.

        Parameters
        ----------
        filename:
            File path.
        diff:
            Unified diff for the file.

        Returns
        -------
        tuple[str, bool]
            (combined_review, was_chunked).
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

        self._log.info(
            'Chunking "%s": diff_tokens=%s budget=%s',
            filename,
            diff_tokens,
            budget,
        )

        parts = _chunk_by_lines(diff, budget)
        out: list[str] = []
        total = len(parts)

        for i, part in enumerate(parts, start=1):
            hdr = f'Part {i}/{total}'
            msg = f'file:\n```{filename}```\ndiff ({hdr}):\n```{part}```'
            chunk_review = self._review_one(msg).strip()
            # Always include part header, even if review is empty
            if chunk_review:
                out.append(f'**{hdr}**\n\n{chunk_review}')
            else:
                out.append(
                    f'**{hdr}**\n\n_No content returned for this part._'
                )

        return ('\n\n'.join(out) if out else '', True)

    def pr_review(self, pr_diff: dict[str, str]) -> list[str]:
        """
        Generate a per-file PR review.

        Parameters
        ----------
        pr_diff:
            Mapping filename -> diff chunk.

        Returns
        -------
        list[str]
            Render-ready markdown sections.
        """
        if not pr_diff:
            return ['LGTM! (No changes detected in diff)']

        results: list[str] = []
        for filename, diff in pr_diff.items():
            if _is_deleted_file(diff):
                results.append(
                    f'### {filename}\n\n_File deleted; no review._\n\n---'
                )
                continue

            try:
                content, was_chunked = self._review_file_in_chunks(
                    filename, diff
                )
                if not content:
                    self._log.warning('Empty model output for "%s"', filename)
                    content = '_No content returned by model._'
                elif (
                    was_chunked
                    and '_No content returned for this part._' in content
                ):
                    # Log warning but keep the part structure
                    self._log.warning(
                        'Some chunks returned empty for "%s"', filename
                    )

                if was_chunked:
                    note = (
                        '> Note: Too many changes in this file; the diff was '
                        'split into parts due to model limits. Consider '
                        'smaller PRs to make review easier.'
                    )
                    content = f'{note}\n\n{content}'

                results.append(f'### {filename}\n\n{content}\n\n---')
            except Exception as exc:
                self._log.exception(
                    'Review failed for "%s": %s', filename, exc
                )
                results.append(
                    f'### {filename}\n'
                    'ChatGPT was not able to review the file. '
                    f'Error: {html.escape(str(exc))}'
                )
        return results

    def comment_review(self, review: list[str]) -> None:
        """
        Post the review as a PR comment.

        Parameters
        ----------
        review:
            Review sections.
        """
        repo = self.gh_api.get_repo(cast(str, self.gh_repo_name))
        pr = repo.get_pull(int(cast(str, self.gh_pr_id)))
        comment = (
            '# OSL ChatGPT Reviewer\n\n'
            '*NOTE: This is generated by an AI program, so some comments may '
            'not make sense.*\n\n'
        ) + '\n'.join(review)
        try:
            pr.create_issue_comment(comment)
        except Exception as exc:
            self._log.exception('Failed to post PR comment: %s', exc)

    def run(self) -> None:
        """Run the PR reviewer end-to-end."""
        pr_diff = self.get_diff()
        review = self.pr_review(pr_diff)
        self.comment_review(review)


if __name__ == '__main__':
    GitHubChatGPTPullRequestReviewer().run()
