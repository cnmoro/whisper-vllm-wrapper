This is a FastAPI wrapper to interact with a Whisper Model running on vLLM.

Whisper by default (specially large-v3-turbo) will hallucinate like crazy on silence, so this wrapper will filter out low-quality segments and avoid repetitions, as well as remove silences beforehand using WebRTC VAD (lightweight)

First, install the dependencies:

```bash
pip install -r requirements.txt
```

Install vLLM:

```bash
pip install vllm vllm[audio]
```

Install linux dependencies:

```bash
sudo apt install ffmpeg -y
```

Then, run the server (note we are disabling prefix caching because it most likely won´t matter for audio processing):

```bash
python -m vllm.entrypoints.openai.api_server --model openai/whisper-large-v3-turbo --port 8001 --host 0.0.0.0 --enforce-eager --no-enable-prefix-caching
# or
./run_vllm.sh
```

Then, run the wrapper (specify the number of workers according to your GPU - vllm will hint how many parallel requests it can handle):

```bash
# Example vLLM initialization output:
# Maximum concurrency for 448 tokens per request: 20.5x
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 20
```

Access the swagger UI at http://localhost:8000/docs to check it out.