# GitHub Actions ChatGPT PR Reviewer

This GitHub actions uses OpenAI ChatGPT in order to review the changes presented
in a PR and will recommend improvements.

### How to use

Include the following into your github actions step

```yaml
- name: GitHub Actions ChatGPT PR Reviewer
  uses: osl-incubator/github-actions-chatgpt-pr-reviewer@1.0.3
  with:
    openai_api_key: ${{ secrets.OPENAI_API_KEY }} # required
    openai_model: "gpt-4o-mini" # optional
    openai_temperature: 0.5 # optional
    openai_max_tokens: 2048 # optional
    prompt_extra_criteria: | # optional, use `;` for separate the criteria items
      - prefer readable variable name instead of short names like `k` and `v`, when apply;
      - verify `SOLID` principles design pattern violations, when apply;
    github_token: ${{ secrets.GITHUB_TOKEN }} # required
    github_pr_id: ${{ github.event.number }} # required
```

The initial criteria use for the ChatGPT would look like this:

```
  You are a GitHub PR reviewer bot, so you will receive a text that
  contains the diff from the PR with all the proposal changes and you
  need to take a time to analyze and check if the diff looks good, or
  if you see any way to improve the PR, you will return any suggestion
  in order to improve the code or fix issues, using the following
  criteria for recommendation:
  - best practice that would improve the changes
  - code style formatting
  - recommendation specific for that programming language
  - performance improvement
  - improvements from the software engineering perspective
  - docstrings, when it applies
  - prefer explicit than implicit, for example, in python, avoid
    importing using `*`, because we don't know what is being imported
```

Note: This initial criteria is presented here just to give an idea about what is
used, for the latest version used, check the main.py file.

## References

This GitHub actions:

- is based on the following tutorial:
  https://shipyard.build/blog/your-first-python-github-action/
- and very inspired on https://github.com/cirolini/chatgpt-github-actions
