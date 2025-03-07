# from unsloth import FastLanguageModel, PatchFastRL
# from unsloth import is_bfloat16_supported
from utils.database_manager import get_db_schema_db_id, schema_linking_scorer
from utils.execution import compare_sqls, execute_sql
from utils.ngrams import jaccard_similarity
from prompts.prompt_loader import load_prompt
from utils.gemini_utils import GeminiModel
from datasets import load_dataset, DatasetDict 
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType, PeftModel
from utils.llm_utils import load_model, load_tokenizer
from dotenv import load_dotenv
from typing import Any
from tqdm import tqdm

import concurrent.futures
import argparse
import os
import ast
import torch
import pandas as pd
import re
import wandb
import json


# def patch_vllm_deepcopy():
#     try:
#         from vllm.engine import llm_engine
#         def engine_deepcopy(self, memo):
#             return self  # Simply return the same instance instead of copying.
#         llm_engine.LLMEngine.__deepcopy__ = engine_deepcopy
#         print("Patched vLLM engine deepcopy successfully.")
#     except ImportError:
#         print("vllm module not found, skipping deepcopy patch.")

# patch_vllm_deepcopy()
NAME = "llm_all"

# wandb.init(project="grpo-training-14B-model", name=NAME)
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

def wrong_question():
    df = pd.read_csv("schema_linking/train_filtered.csv")
    wrong_ids = {}
    for index, row in df.iterrows():
        if row['overall_confidence'] == 10:
            wrong_ids[row['question_id'].split("_",1)[0]] = row['question']
    return wrong_ids


def construct_fineutning_dataset(tokenizer: Any, wrong_ids, schema_linking_result):
    dataset_name = "finetuning_datasets/zero_shot_grpo.csv"
    if os.path.exists(dataset_name):
        dataset = load_dataset('csv', data_files=dataset_name)
        dataset = dataset.map(lambda x: {"prompt": json.loads(x["prompt"])})  # Convert string to dict
        return dataset
    df = pd.read_json("data/train/train.json")
    # df = df.sample(frac=1).reset_index(drop=True)
    training_datasets = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if str(index+1) in wrong_ids:
            continue
        question = row["question"]
        db_id = row["db_id"]
        gold_query = row["SQL"]
        evidence = row["evidence"]
        try:
            tentative_schema = schema_linking_result[schema_linking_result['question_id'] == index]['tentative_schema'].values[0]
            tentative_schema = json.loads(json.dumps(ast.literal_eval(tentative_schema)))
            database_schema = get_db_schema_db_id(
                db_id=db_id,
                bird_database_path=os.getenv("BASE_TRAIN_DATA_PATH"),
                queries=[gold_query],
                tentative_schema=tentative_schema
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
        # messages = [
        #     {"role": "system", 'content': SYSTEM_PROMPT},
        #     {"role": "user", "content": user_messages},
        # ]
        training_datasets.append({
            'prompt': [
                {"role": "system", 'content': SYSTEM_PROMPT},
                {"role": "user", "content": user_messages},
            ],
            'answer': gold_query,
            'db_id': db_id,
            'question': question,
            'evidence': evidence,
        })
    os.makedirs("finetuning_datasets", exist_ok=True)
    dataset = pd.DataFrame(training_datasets)
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    dataset["prompt"] = dataset["prompt"].apply(json.dumps)
    dataset.to_csv(dataset_name)
    dataset = load_dataset('csv', data_files=dataset_name)
    dataset = dataset.map(lambda x: {"prompt": json.loads(x["prompt"])})  # Convert string to dict
    return dataset


def extract_sql_queries(text):
    pattern = r"```sql\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        queries = [match.strip() for match in matches]
        return queries[-1]  # Return the last query
    else:
        return text

###### --------------------- REWARD FUNCTIONS --------------------- ######


def execution_acc_reward_func(prompts, completions, answer,db_id ,question ,evidence, **kwargs) -> list[float]:
    print(f"Sample Completions :\n{completions[0][0]['content']}")
    responses = [extract_sql_queries(completion[0]['content']) for completion in completions]
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


def old_execution_acc_reward_func(prompts, completions, answer, db_id, **kwargs) -> list[float]:
    print(f"Sample Completions :\n{completions[0][0]['content']}")
    responses = [extract_sql_queries(completion[0]['content']) for completion in completions]

    def evaluate(response, db, gold_query):
        try:
            if "SELECT" not in response:
                return 0.0
            exec_res = compare_sqls(
                db_directory_path=os.getenv("BASE_TRAIN_DATA_PATH"),
                db_id=db,
                predicted_sql=response,
                ground_truth_sql=gold_query,
            )
            return 2.0 if exec_res.get('exec_res') else 0.0
        except Exception:
            return 0.0

    # Use ThreadPoolExecutor to process items in parallel.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        rewards = list(executor.map(evaluate, responses, db_id, answer))
    
    print(f"Rewards: {rewards}")
    return rewards

def sql_ngram_similarity(prompts, completions, answer, db_id, **kwargs) -> list[float]:
    responses = [extract_sql_queries(completion[0]['content']) for completion in completions]
    rewards = []
    for response, gold_query in zip(responses, answer):
        try:
            rewards.append(jaccard_similarity(response, gold_query, n = 2))
        except Exception as e:
            rewards.append(0.0)
    return rewards


def syntax_check_reward_func(prompts, completions, answer, db_id, **kwargs) -> list[float]:
    responses = [extract_sql_queries(completion[0]['content']) for completion in completions]
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
    responses = [extract_sql_queries(completion[0]['content']) for completion in completions]
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
    """Strict reward function that checks if the completion has an exact format."""
    pattern = r"^\s*<reasoning>\s*.*?\s*</reasoning>\s*<answer>\s*.*?\s*</answer>\s*$"
    matches = [re.fullmatch(pattern, r, re.DOTALL) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Soft reward function that checks if the completion loosely follows the format."""
    pattern = r"<reasoning>\s*.*?\s*</reasoning>\s*<answer>\s*.*?\s*</answer>"
    matches = [re.search(pattern, r, re.DOTALL) for r in completions]
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
    return [count_xml(c[0]['content']) for c in completions]



def train_model(dataset: Any, args: argparse.Namespace, tokenizer: Any, model: Any):
    training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        vllm_device='cuda:7',
        vllm_gpu_memory_utilization=0.5,
        learning_rate = 5e-5,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        save_total_limit=1,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "constant_with_warmup",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        temperature=args.temperature,
        num_generations = args.num_generations,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = args.epochs, # Set to 1 for a full training run
        # max_steps = 250,
        save_steps = 250,
        max_grad_norm = 0.2,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = args.output_model_name
    )

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            execution_acc_reward_func,
            syntax_check_reward_func,
            schema_linking_reward_func,
            xmlcount_reward_func,
            sql_ngram_similarity,
            # soft_format_reward_func,
            # strict_format_reward_func,
        ],
        args = training_args,
        train_dataset = dataset["train"],
        # eval_dataset= dataset["validation"],
        peft_config=peft_config
    )
    train_results = trainer.train()

    return trainer, train_results

def filter_samples_based_on_length(example: Any, max_seq_length: int, tokenizer: Any):
    # user_messages = example["prompt"]
    # messages = [
    #     {"role": "user", "content": user_messages},
    # ]
    messages = example["prompt"]
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)) <= max_seq_length

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-14B-Instruct")
    args.add_argument("--max_seq_length", type=int, default=4000)
    args.add_argument("--max_prompt_length", type=int, default=2500)
    args.add_argument("--max_completion_length", type=int, default=1500)
    args.add_argument("--lora_rank", type=int, default=32)
    args.add_argument("--lora_alpha", type=int, default=16)
    args.add_argument("--per_device_train_batch_size", type=int, default=6)
    args.add_argument("--epochs", type=int, default=3)
    args.add_argument("--gradient_accumulation_steps", type=int, default=4)
    args.add_argument("--num_generations", type=int, default=6) # Decrease if out of memory
    args.add_argument("--hf_username", type=str, default="MrezaPRZ")
    args.add_argument("--output_model_name", type=str, default=f"wen2.5-Coder-14B-Instruct-{NAME}")
    args.add_argument("--temperature", type=int, default=0.8)
    args = args.parse_args()
    wrong_ids = wrong_question()
    print(f"Number of wrong train set samples: {len(wrong_ids)}")
    new_model_name = f"{args.hf_username}/{args.output_model_name}"
    model = load_model(args.model_name, quantize=False)
    tokenizer = load_tokenizer(args.model_name)
    schema_linking_result = pd.read_csv("schema_linking/train_schema_info.csv")
    dataset = construct_fineutning_dataset(tokenizer, wrong_ids, schema_linking_result)
    dataset = dataset['train'].train_test_split(test_size=0.01, shuffle=True)
    dataset = DatasetDict({
        'train': dataset['train'],
        'validation': dataset['test']
    })
    dataset = dataset.filter(filter_samples_based_on_length, fn_kwargs={'max_seq_length': args.max_prompt_length, 'tokenizer': tokenizer})
    print(f"No of samples: {dataset['train'].shape[0]}")
    trainer, train_results = train_model(dataset, args, tokenizer, model)
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model(args.output_model_name)

    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(args.output_model_name)
        trainer.tokenizer.push_to_hub(new_model_name)
        del model
        del trainer
        torch.cuda.empty_cache()
        base_model = load_model(args.model_name)
        model = PeftModel.from_pretrained(base_model, args.output_model_name)
        model = model.merge_and_unload()
        model.push_to_hub(new_model_name)
