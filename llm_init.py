import os
import torch
from transformers import AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def initialize_LLM():
    """
    Initializes and configures the Llama 3 model for LlamaIndex workflows.

    The Hugging Face token is loaded from the HF_TOKEN environment variable.
    Do not hardcode tokens in source code or push them to GitHub.
    """
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is not set. "
            "Set it locally using: export HF_TOKEN=your_huggingface_token"
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=hf_token,
        )

        stopping_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        llm = HuggingFaceLLM(
            model_name=MODEL_NAME,
            model_kwargs={
                "token": hf_token,
                "torch_dtype": torch.bfloat16,
            },
            generate_kwargs={
                "do_sample": False,
                "max_new_tokens": 120,
            },
            tokenizer_name=MODEL_NAME,
            tokenizer_kwargs={
                "token": hf_token,
            },
            stopping_ids=stopping_ids,
        )

        Settings.llm = llm
        return llm

    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise