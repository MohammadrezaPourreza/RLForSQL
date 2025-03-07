import torch
import re
import pandas as pd
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from peft import PeftModel
from torch import cuda
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "qwen3B-GRPO-llm_judge/checkpoint-849"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2", # use with amper architecture
    torch_dtype=torch.bfloat16,
    device_map = "auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.push_to_hub("MrezaPRZ/Qwen2.5-Coder-3B-grpo-llm_judge_849")
tokenizer.push_to_hub("MrezaPRZ/Qwen2.5-Coder-3B-grpo-llm_judge_849")