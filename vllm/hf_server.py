# openai_server.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import uuid
import torch
import time
import asyncio
from sentence_transformers import CrossEncoder, SentenceTransformer

# import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob, TopLogprob
from huggingface import HuggingFaceModel, ContextLengthExceededError

app = FastAPI()

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    
class RelevanceRequest(BaseModel):
    model: str
    text1: str
    text2: str
    
import threading
processing_lock = threading.Lock()


loaded_llms = {}
loaded_encoders = {}

# -------- Model Loader --------
def get_or_load_model(model_name: str) -> HuggingFaceModel:
    with processing_lock:
        if model_name not in loaded_llms:
            loaded_llms[model_name] = HuggingFaceModel(model_name=model_name)
            loaded_llms[model_name].initialize()
        return loaded_llms[model_name]

def get_or_load_encoder_model(model_name: str) -> SentenceTransformer:
    with processing_lock:
        if model_name not in loaded_encoders:
            loaded_encoders[model_name] = SentenceTransformer(model_name)
        return loaded_encoders[model_name]

@app.exception_handler(ContextLengthExceededError)
async def context_length_exceeded_handler(
    request: Request,
    exc: ContextLengthExceededError,
):
    return JSONResponse(
        status_code=413,  # Payload Too Large
        content={
            "error": {
                "type": "context_length_exceeded",
                "message": (
                    f"Context length {exc.tokens} tokens exceeds "
                    f"maximum supported {exc.max_tokens} tokens."
                ),
                "context_length": exc.tokens,
                "max_context_length": exc.max_tokens,
            }
        },
    )
    
# -------- Endpoint --------
@app.post("/api/v1/chat/completions", response_model=ChatCompletion)
async def chat_completions(request: ChatCompletionRequest):
    model: HuggingFaceModel = get_or_load_model(request.model)

    output = model.chat(
        messages=request.messages,
        temperature=request.temperature,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
    )

    # print(f"Total logs: {len(top_log_probs)}")
    return ChatCompletion(
        id=f"chatcmpl-{uuid.uuid4()}",
        # current unix timestamp in seconds
        created=int(time.time()),
        model=request.model,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=output),
                finish_reason="stop",
            )
        ],
    )
    
# An api endpoint to calculate relevance score of two text using cross-encoder model
@app.post("/api/v1/relevance")
async def relevance_score(request: RelevanceRequest) -> Dict[str, Any]:
    encoder: SentenceTransformer = get_or_load_encoder_model(request.model)
    emb1 = encoder.encode(
        request.text1,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    emb2 = encoder.encode(
        request.text2,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    cosine_sim = torch.sum(emb1 * emb2).item()  # ∈ [-1, 1]

    return {"score": (cosine_sim + 1.0) / 2.0}  # scale to [0, 1]
    # score = encoder.predict([(request.text1, request.text2)])[0]
    # prob = torch.sigmoid(torch.tensor(score))
    # return {"score": prob.item()}

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}

# Usage: uvicorn hf_server:app --host 0.0.0.0 --port 5000
# python -m uvicorn hf_server:app --host 0.0.0.0 --port 5000