from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv(override=True)


class HuggingFaceError(Exception):
    """Custom exception for Hugging Face model errors"""
    pass

class ContextLengthExceededError(Exception):
    def __init__(self, tokens: int, max_tokens: int):
        self.tokens = tokens
        self.max_tokens = max_tokens

class HuggingFaceModel:
    def __init__(self, model_name: str, **kwargs):
        self.name = model_name
        self.max_tokens = kwargs.get("max_tokens", 512)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 1.0)
        self.reasoning_effort = kwargs.get("reasoning_effort", None)
        self.n = kwargs.get("n", 1)
        self.input_tokens = 0
        self.output_tokens = 0
        self.tokenizer = None
        self.model = None

    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name, trust_remote_code=True, use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            load_in_8bit=True,
            # max_memory={0: "20GB"},
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        # terminators = [
        #     self.pipe.tokenizer.eos_token_id,
        #     self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        # ]

        self.generation_args = {
            "max_new_tokens": 512,
            "temperature": self.temperature,
            "do_sample": True,
            "eos_token_id": self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            or self.pipe.tokenizer.eos_token_id,
            "use_cache": True,
        }

    def log_gpu_usage(self, tag=""):
        if not torch.cuda.is_available():
            print("CUDA not available")
            return

        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        max_alloc = torch.cuda.max_memory_allocated(device) / 1024**2

        print(
            f"[GPU {device}] {tag} | "
            f"allocated={allocated:.1f}MB | "
            f"reserved={reserved:.1f}MB | "
            f"max_alloc={max_alloc:.1f}MB"
        )
    
    def count_input_tokens(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        return input_ids.shape[-1]

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> List[str]:
        """Chat completion using tokenizer.apply_chat_template."""

        if not self.tokenizer or not self.model:
            self.initialize()
 
        input_tokens = self.count_input_tokens(messages)

        MAX_CONTEXT_TOKENS = 12288
        if input_tokens > MAX_CONTEXT_TOKENS:
            print(f"[ERROR] Input tokens ({input_tokens}) exceed max context length ({MAX_CONTEXT_TOKENS}).")
            raise ContextLengthExceededError(
                tokens=input_tokens,
                max_tokens=MAX_CONTEXT_TOKENS,
            )
    
        try:
            output = self.pipe(messages, **self.generation_args)
            return output[0]["generated_text"][-1]["content"]

        except Exception as e:
            raise HuggingFaceError(f"Error during chat generation: {str(e)}")