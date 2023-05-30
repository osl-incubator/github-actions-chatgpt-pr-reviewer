#!/bin/bash

export OPENAI_API_KEY="$1"
export OPENAI_MODEL="$2"
export OPENAI_TEMPERATURE="$3"
export OPENAI_MAX_TOKENS="$4"
export GITHUB_TOKEN="$5"
export GITHUB_PR_ID="$6"

python /main.py
