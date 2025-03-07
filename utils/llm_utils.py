from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria

import torch

from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [6203]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids


def load_model(model_name: str, quantize: bool = False, use_flash_attn: bool = True) -> AutoModelForCausalLM:
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    if use_flash_attn:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # bnb_config=bnb_config,
        attn_implementation=attn_implementation,
        torch_dtype="auto",
        # device_map="auto"
    )
    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def call_model(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        user_message: str,
        max_new_tokens: int, 
        do_sample: bool = False,
        config: dict = None
    ) -> str:
    messages = [
        {"role": "user", "content": user_message},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        **config if config is not None else {}
    )
    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def call_model_openai(
        model_name: AutoModelForCausalLM,
        user_message: str,
        max_new_tokens: int, 
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> str:
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": user_message}],
        temperature=temperature,
        max_tokens=max_new_tokens,
        # stop=["\n```sql"]
    )
    return response.choices[0].message.content