## Installation
```bash
pip install -r requirements.txt
```

## Run HF Server

```bash
uvicorn hf_server:app --host 0.0.0.0 --port 8000
```

```bash
python -m vllm.entrypoints.openai.api_server \
	--model Qwen/Qwen2.5-7B-Instruct \
	--host 0.0.0.0 \
	--port 3000 \
	--tensor-parallel-size 1 \
	--max-model-len 32768 \
	--max-num-batched-tokens 49152 \
	--max-num-seqs 16 \
	--dtype float16 2>&1 | tee vllm_server.log
```

```bash
python -m vllm.entrypoints.openai.api_server     --model Qwen/Qwen3.5-4B   --host 0.0.0.0     --port 3000     --tensor-parallel-size 1   --max-model-len 32768   --max-num-batched-tokens 32768     --max-num-seqs 8 --dtype float16 --reasoning-parser qwen3 --language-model-only --default-chat-template-kwargs '{"enable_thinking": false}'
```
<!-- vllm serve Qwen/Qwen3.5-4B --port 8000 --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3 --language-model-only
 -->
<!-- 10.141.10.34  -->