# Importing necessarry libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import faiss
import fitz

# Login HuggingFace Hub
login(token="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")

# Daha büyük ve güçlü bir model seçimi (ör: google/gemma-7b-it)
model_id = "google/gemma-7b-it"  # Daha büyük model, daha iyi analiz ve uzun cevaplar için

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
    use_auth_token=True,
    attn_implementation=set_attn_implementation(),
    device_map="auto"
)

class ModelFactory:
    @staticmethod
    def get_model(model_type: str):
        if model_type == "gemma-7b-it":
            return AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", ...)
        elif model_type == "gemma-2-2b-it":
            return AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", ...)
        # Diğer modeller...

class EmbeddingStrategy:
    def encode(self, text):
        raise NotImplementedError

class SentenceTransformerEmbedding(EmbeddingStrategy):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def encode(self, text):
        return self.model.encode(text)

class OpenAIEmbedding(EmbeddingStrategy):
    # OpenAI API ile embedding
    ...

class FaissIndexSingleton:
    _instance = None
    @staticmethod
    def get_instance(dim):
        if FaissIndexSingleton._instance is None:
            FaissIndexSingleton._instance = faiss.IndexFlatL2(dim)
        return FaissIndexSingleton._instance

class PDFReaderAdapter:
    def read(self, pdf_stream):
        raise NotImplementedError

class FitzPDFReader(PDFReaderAdapter):
    def read(self, pdf_stream):
        return fitz.open(stream=pdf_stream, filetype='pdf')