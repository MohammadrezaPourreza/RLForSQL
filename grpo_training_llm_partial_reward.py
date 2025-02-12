from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
from utils.database_manager import get_db_schema_db_id, schema_linking_scorer
from utils.execution import compare_sqls, execute_sql
from prompts.prompt_loader import load_prompt
from utils.gemini_utils import GeminiModel
from datasets import load_dataset, DatasetDict
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv
from typing import Any
from tqdm import tqdm

import concurrent.futures
import argparse
import os
import torch
import pandas as pd
import re
import wandb


wandb.init(project="grpo-training-all", name="only-ex-feedback")
judge_model = GeminiModel(model_name="gemini-1.5-pro-002")

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

load_dotenv(override=True)

def patch_grpo():
    PatchFastRL("GRPO", FastLanguageModel)

def load_model(model_name: str, max_seq_length: int, lora_rank: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = False, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.5, # Reduce if out of memory
    )
    return model, tokenizer


def get_peft_model(model: Any, lora_alpha: int, lora_rank: int):
    return FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_alpha,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )

def construct_fineutning_dataset(tokenizer: Any):
    dataset_name = "finetuning_datasets/zero_shot_grpo.csv"
    if os.path.exists(dataset_name):
        return load_dataset('csv', data_files=dataset_name)
    df = pd.read_json("data/train/train.json")
    df = df.sample(frac=1).reset_index(drop=True)
    training_datasets = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        question = row["question"]
        db_id = row["db_id"]
        gold_query = row["SQL"]
        evidence = row["evidence"]
        try:
            database_schema = get_db_schema_db_id(
                db_id=db_id,
                bird_database_path=os.getenv("BASE_TRAIN_DATA_PATH"),
                queries=[gold_query],
            )
        except Exception as e:
            print(e)
            continue
        prompt = load_prompt(
            'sql_generation_zero_shot'
        )
        user_messages = prompt.format(
            QUESTION=question,
            DATABASE_SCHEMA=database_schema,
            HINT=evidence,
        )
        messages = [
            {"role": "system", 'content': SYSTEM_PROMPT},
            {"role": "user", "content": user_messages},
        ]
        training_datasets.append({
            'prompt': tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
            'answer': gold_query,
            'db_id': db_id,
            'question': question,
            'evidence': evidence,
        })
    dataset = pd.DataFrame(training_datasets)
    dataset.to_csv(dataset_name)
    os.makedirs("finetuning_datasets", exist_ok=True)
    return load_dataset('csv', data_files=dataset_name)


def extract_sql_queries(text):
    pattern = r"```sql(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        queries = [match.strip() for match in matches]
        return queries[-1] # Return the last query
    else:
        return text 

###### --------------------- REWARD FUNCTIONS --------------------- ######


def execution_acc_reward_func(prompts, completions, answer,db_id ,question ,evidence, **kwargs) -> list[float]:
    print(f"Sample Completions :\n{completions[0]}")
    responses = [extract_sql_queries(completion) for completion in completions]
    judge_prompt = load_prompt("judge_prompt")

    def evaluate(response, db, gold_query, question, evidence):
        try:
            if "SELECT" not in response:
                return 0.0
            exec_res = compare_sqls(
                db_directory_path=os.getenv("BASE_TRAIN_DATA_PATH"),
                db_id=db,
                predicted_sql=response,
                ground_truth_sql=gold_query,
            )
            if exec_res.get('exec_res'):
                return 3.0
            else:
                judge_prompt_formatted = judge_prompt.format(
                    QUESTION=question,
                    HINT=evidence,
                    PREDICTED_QUERY=response,
                    GOLD_QUERY=gold_query,
                )
                judge_res = judge_model.call(judge_prompt_formatted)
                try:
                    judge_res = float(judge_res)
                    return judge_res
                except Exception:
                    return 0.0
        except Exception:
            return 0.0

    # Use ThreadPoolExecutor to process items in parallel.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        rewards = list(executor.map(evaluate, responses, db_id, answer, question, evidence))
    
    print(f"Rewards: {rewards}")
    return rewards


def syntax_check_reward_func(prompts, completions, answer, db_id, **kwargs) -> list[float]:
    responses = [extract_sql_queries(completion) for completion in completions]
    rewards = []
    for response, db_id in zip(responses, db_id):
        db_path  = os.getenv("BASE_TRAIN_DATA_PATH") + f"/{db_id}/{db_id}.sqlite"
        try:
            execute_sql(db_path=db_path, sql=response, fetch="one")
            rewards.append(1.0)
        except Exception as e:
            rewards.append(0.0)
    return rewards

def schema_linking_reward_func(prompts, completions, answer, db_id, **kwargs) -> list[float]:
    responses = [extract_sql_queries(completion) for completion in completions]
    rewards = []
    for response, gold_query in zip(responses, answer):
        if "SELECT" not in response:
            rewards.append(0.0)
            continue
        try:
            schema_linking_score = schema_linking_scorer(
                gold_query, response
            )
            rewards.append(schema_linking_score)
        except Exception as e:
            rewards.append(0.0)
    return rewards
    


### formatting reward functions:

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, r) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"(?s)<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, r) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(c) for c in completions]



def train_model(dataset: Any, args: argparse.Namespace, tokenizer: Any, model: Any):
    training_args = GRPOConfig(
        use_vllm = False, # use vLLM for fast inference!
        learning_rate = 5e-5,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        num_generations = args.num_generations,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = args.epochs, # Set to 1 for a full training run
        # max_steps = 250,
        save_steps = 250,
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = "outputs5",
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            execution_acc_reward_func,
            # syntax_check_reward_func,
            # schema_linking_reward_func,
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
        ],
        args = training_args,
        train_dataset = dataset["train"],
        # eval_dataset= dataset["validation"],
    )
    trainer.train()

    return trainer

def filter_samples_based_on_length(example: Any, max_seq_length: int, tokenizer: Any):
    user_messages = example["prompt"]
    messages = [
        {"role": "user", "content": user_messages},
    ]
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)) <= max_seq_length


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    args.add_argument("--max_seq_length", type=int, default=2500)
    args.add_argument("--max_prompt_length", type=int, default=1500)
    args.add_argument("--max_completion_length", type=int, default=800)
    args.add_argument("--lora_rank", type=int, default=16)
    args.add_argument("--lora_alpha", type=int, default=16)
    args.add_argument("--per_device_train_batch_size", type=int, default=6)
    args.add_argument("--epochs", type=int, default=10)
    args.add_argument("--gradient_accumulation_steps", type=int, default=2)
    args.add_argument("--num_generations", type=int, default=4) # Decrease if out of memory
    args = args.parse_args()

    patch_grpo()

    model, tokenizer = load_model(args.model_name, args.max_seq_length, args.lora_rank)
    dataset = construct_fineutning_dataset(tokenizer)
    # dataset = dataset['train'].train_test_split(test_size=0.001, shuffle=True)
    dataset = DatasetDict({
        'train': dataset['train'],
        # 'validation': dataset['test']
    })
    dataset = dataset.filter(filter_samples_based_on_length, fn_kwargs={'max_seq_length': args.max_prompt_length, 'tokenizer': tokenizer})
    print(f"No of samples: {dataset['train'].shape[0]}")
    model = get_peft_model(model, args.lora_alpha, args.lora_rank)
    trainer = train_model(dataset, args, tokenizer, model)
    model.save_lora("grpo_saved_lora_ex_syntax_schema_gold_schema")
    model.save_pretrained_merged("grpo_model_ex_syntax_schema_gold_schema", tokenizer, save_method = "merged_16bit",)
    # model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")
