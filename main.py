import html
import os

from github import Github
import openai
import requests

class GitHubChatGPTPullRequestReviewer:
    def __init__(self):
        self._config_gh()
        self._config_openai()

    def _config_gh(self):
        gh_api_url = "https://api.github.com"

        self.gh_pr_id = os.environ.get("GITHUB_PR_ID")

        self.gh_token = os.environ.get("GITHUB_TOKEN")
        self.gh_repo_name = os.getenv('GITHUB_REPOSITORY')

        self.gh_pr_url = (
            f"{gh_api_url}/repos/{self.gh_repo_name}/pulls/{self.gh_pr_id}"
        )
        self.gh_headers = {
            'Authorization': f"token {self.gh_token}",
            'Accept': 'application/vnd.github.v3.diff'
        }
        self.gh_api = Github(self.gh_token)

    def _config_openai(self):
        openai_model_default = "gpt-3.5-turbo"
        openai_temperature_default = 0.5
        openai_max_tokens_default = 2048

        openai_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.openai_model = os.environ.get("OPENAI_MODEL", openai_model_default)
        self.openai_temperature = os.environ.get("OPENAI_TEMPERATURE", openai_temperature_default)
        self.openai_max_tokens = os.environ.get("OPENAI_MAX_TOKENS", openai_max_tokens_default)

        openai.api_key = openai_api_key

        self.chatgpt_initial_instruction = """
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

            Please, return your response in markdown format.
            If the changes presented by the diff looks good, just say: "LGTM!"
        """.strip()

    def get_pr_content(self):
        response = requests.request("GET", self.gh_pr_url, headers=self.gh_headers)
        if response.status_code != 200:
            raise Exception(response.text)
        return response.text

    def get_diff(self) -> dict:
        repo = self.gh_api.get_repo(self.gh_repo_name)
        pull_request = repo.get_pull(int(self.gh_pr_id))

        content = self.get_pr_content()

        if len(content) == 0:
            pull_request.create_issue_comment(f"PR does not contain any changes")
            return ""

        parsed_text = content.split("diff")

        files_diff = {}
        content = []
        file_name = ""

        for diff_text in parsed_text:
            if len(diff_text) == 0:
                continue

            if not diff_text.startswith(' --git a/'):
                content += [diff_text]
                continue

            if file_name and content:
                files_diff[file_name] = "\n".join(content)

            file_name = diff_text.split("b/")[1].splitlines()[0]
            content = [diff_text]

        if file_name and content:
            files_diff[file_name] = "\n".join(content)

        return files_diff



    def pr_review(self, pr_diff: dict):
        system_message = [
            {"role": "system", "content": self.chatgpt_initial_instruction},
        ]

        results = []

        for filename, diff in pr_diff.items():
            message_diff = f"file: ```{filename}```\ndiff: ```{diff}```"
            messages = [{"role": "user", "content": message_diff}]

            print(
                "Estimated number of tokens: ",
                len(self.chatgpt_initial_instruction + message_diff) / 4
            )

            try:
                # create a chat completion
                chat_completion = openai.ChatCompletion.create(
                    model=self.openai_model,
                    temperature=float(self.openai_temperature),
                    max_tokens=int(self.openai_max_tokens or self.openai_max_tokens_default),
                    messages=system_message + messages
                )
                results.append(
                    f"### {filename}\n\n{chat_completion.choices[0].message.content}\n\n---"
                )
            except Exception as e:
                results.append(
                    f"### {filename}\nChatGPT was not able to review the file."
                    f" Error: {html.escape(str(e))}"
                )

        return results


    def comment_review(self, review: list):
        repo = self.gh_api.get_repo(self.gh_repo_name)
        pull_request = repo.get_pull(int(self.gh_pr_id))
        pull_request.create_issue_comment('\n'.join(review))

    def run(self):
        pr_diff = self.get_diff()
        review = self.pr_review(pr_diff)
        self.comment_review(review)


if __name__ == "__main__":
    reviewer = GitHubChatGPTPullRequestReviewer()
    reviewer.run()
