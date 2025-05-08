# Importing necessarry libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from huggingface_hub import login

# Login HuggingFace Hub
login(token="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")

# Create the model
model_id = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id, use_auth_token=True)

# Attention implementation
def set_attn_implementation():
    if torch.cuda.get_device_capability(0)[0] >= 8:
        return "flash_attention_2"
    else:
        return "sdpa"

# Creating the LLM
llm_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
    use_auth_token=True
)