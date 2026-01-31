#!/bin/bash
python -m vllm.entrypoints.openai.api_server --model openai/whisper-large-v3-turbo --port 8001 --host 127.0.0.1 --enforce-eager --no-enable-prefix-caching