import torch
from transformers import AutoTokenizer
# Updated import to use the newer package structure
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

def initialize_LLM():
    """
    Initializes and configures the Llama language model with specific parameters.
    This function handles the setup of tokenizer, model configuration, and 
    establishes the necessary parameters for model generation.
    
    Returns:
        HuggingFaceLLM: A configured instance of the Hugging Face language model
    
    Raises:
        Exception: If there are issues with model initialization or token authentication
    """
    try:
        hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError(
        "HF_TOKEN environment variable is not set. "
        "Create a local .env file or export HF_TOKEN before running the project."
    )
        
        # Initialize the tokenizer with specific model configuration
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            token=hf_token,
        )
        
        # Define stopping criteria for text generation
        # These tokens indicate when the model should stop generating
        stopping_ids = [
            tokenizer.eos_token_id,  # End of sequence token
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # Custom end token
        ]
        
        # Initialize the language model with specific configurations
        llm = HuggingFaceLLM(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            model_kwargs={
                "token": hf_token,
                # Using bfloat16 for efficient memory usage while maintaining precision
                "torch_dtype": torch.bfloat16,
            },
            # Generation parameters to control output quality and diversity
            generate_kwargs={
                "do_sample": True,      # Enable sampling for more diverse outputs
                "temperature": 0.6,     # Controls randomness (lower = more focused)
                "top_p": 0.9,          # Nucleus sampling parameter
            },
            tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
            tokenizer_kwargs={"token": hf_token},
            stopping_ids=stopping_ids,
        )
        
        # Optional: Configure global settings for the LLM
        Settings.llm = llm
        
        return llm
        
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise  # Re-raise the exception after logging for proper error handling

    # Note: Quantization configuration is kept as a comment for future reference
    # To enable 4-bit quantization, uncomment and configure as follows:
    """
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    """import os
import torch
from transformers import AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings


def initialize_LLM():
    """
    Initializes and configures the Llama 3 language model using Hugging Face.

    The Hugging Face token is loaded from the HF_TOKEN environment variable
    instead of being hardcoded in source code.
    """
    try:
        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            raise ValueError(
                "HF_TOKEN environment variable is not set. "
                "Create a local .env file or export HF_TOKEN before running the project."
            )

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            token=hf_token,
        )

        stopping_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        llm = HuggingFaceLLM(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            model_kwargs={
                "token": hf_token,
                "torch_dtype": torch.bfloat16,
            },
            generate_kwargs={
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.9,
            },
            tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
            tokenizer_kwargs={"token": hf_token},
            stopping_ids=stopping_ids,
        )

        Settings.llm = llm

        return llm

    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise
