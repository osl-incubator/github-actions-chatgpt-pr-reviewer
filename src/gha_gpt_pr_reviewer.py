import os

import openai


def main():
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_ENGINE = os.environ.get("OPENAI_ENGINE")
    OPENAI_TEMPERATURE = os.environ.get("OPENAI_TEMPERATURE")
    OPENAI_MAX_TOKENS = os.environ.get("OPENAI_MAX_TOKENS")
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

    openai.api_key = OPENAI_API_KEY

    pr_diff = ""

    # create a chat completion
    chat_completion = openai.ChatCompletion.create(
        model=(OPENAI_ENGINE or "gpt-3.5-turbo"),
        messages=[
            {"role": "system", "content": "Hello world"},
            {"role": "user", "content": pr_diff},
        ]
    )

    # print the chat completion
    print(chat_completion.choices[0].message.content)

    # to set output, print to shell in following syntax
    print(f"::set-output chatgpt_result=OK")


if __name__ == "__main__":
    main()
