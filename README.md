# GitHub Actions ChatGPT PR Reviewer

This GitHub actions uses OpenAI ChatGPT in order to review the changes
presented in a PR and will recommend improvements.

### How to use

Include the following into your github actions step

```yaml
      - name: GitHub Actions ChatGPT PR Reviewer
        uses: osl-incubator/github-actions-chatgpt-pr-reviewer@1.0.2
        with:
          openai_api_key: ${{ secrets.OPENAI_API_KEY }}
          openai_model: 'gpt-3.5-turbo'
          openai_temperature: 0.5
          openai_max_tokens: 2048
          github_token: ${{ secrets.GITHUB_TOKEN }}
          github_pr_id: ${{ github.event.number }}
```

Note: this GitHub actions is based on the following tutorial:
  https://shipyard.build/blog/your-first-python-github-action/
  and very inspired on https://github.com/cirolini/chatgpt-github-actions
