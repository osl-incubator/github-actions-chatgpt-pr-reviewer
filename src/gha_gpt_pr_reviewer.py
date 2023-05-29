import os

import openai


def main():
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_ENGINE = os.environ.get("OPENAI_ENGINE")
    OPENAI_TEMPERATURE = os.environ.get("OPENAI_TEMPERATURE")
    OPENAI_MAX_TOKENS = os.environ.get("OPENAI_MAX_TOKENS")
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

    openai.api_key = OPENAI_API_KEY

    # hard-coded example for now, it will be replaced by the real
    # content from PR
    pr_diff = '''
        diff --git a/src/gha_gpt_pr_reviewer.py b/src/gha_gpt_pr_reviewer.py
        index 9d3beff..6facd13 100644
        --- a/src/gha_gpt_pr_reviewer.py
        +++ b/src/gha_gpt_pr_reviewer.py
        @@ -14,11 +14,23 @@ def main():

        pr_diff = """
            +    chatgpt_initial_instruction = """
            +        You are a GitHub PR reviewer bot, so you will receive a text that
            +        contains the diff from the PR with all the proposal changes and you
            +        need to take a time to analyze and check if the diff looks good, or
            +        if you see any way to improve the PR, you will return any suggestion
            +        in order to improve, fix issues or suggest any best practice like
            +        code style formatting or any recommendation specific for that
            +        programming language, or text format. Also, check and comment any
            +        improvement from the software engineer perspective.
            +        Please return your response in markdown format.
            +    """.strip()
            +
                # create a chat completion
                chat_completion = openai.ChatCompletion.create(
                    model=(OPENAI_ENGINE or "gpt-3.5-turbo"),
                    messages=[
            -            {"role": "system", "content": "Hello world"},
            +            {"role": "system", "content": chatgpt_initial_instruction},
                        {"role": "user", "content": pr_diff},
                    ]
                )
    '''

    chatgpt_initial_instruction = """
        You are a GitHub PR reviewer bot, so you will receive a text that
        contains the diff from the PR with all the proposal changes and you
        need to take a time to analyze and check if the diff looks good, or
        if you see any way to improve the PR, you will return any suggestion
        in order to improve, fix issues or suggest any best practice like
        code style formatting or any recommendation specific for that
        programming language, or text format. Also, check and comment any
        improvement from the software engineer perspective.
        Please return your response in markdown format.
    """.strip()

    # create a chat completion
    chat_completion = openai.ChatCompletion.create(
        model=(OPENAI_ENGINE or "gpt-3.5-turbo"),
        messages=[
            {"role": "system", "content": chatgpt_initial_instruction},
            {"role": "user", "content": pr_diff},
        ]
    )

    result = chat_completion.choices[0].message.content

    # print the chat completion
    print(result)

    # to set output, print to shell in following syntax
    print(f'::set-output chatgpt_result="{result}"')


if __name__ == "__main__":
    main()
