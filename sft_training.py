import pandas as pd
import dataclasses
import argparse
import os
import torch

from utils.llm_utils import load_model, load_tokenizer
from utils.database_manager import get_db_schema_db_id
from prompts.prompt_loader import load_prompt
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from trl.trainer import SFTConfig
from peft import LoraConfig, TaskType, PeftModel
from typing import Any

def construct_fineutning_dataset(zero_shot: bool = False):
    dataset_name = "finetuning_datasets/zero_shot_sft.csv" if zero_shot else "finetuning_datasets/few_shot_sft.csv"
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
            )
        except Exception as e:
            print(f"Error in getting database schema: {e}")
            continue
        if zero_shot:
            prompt = load_prompt(
                'sql_generation_zero_shot'
            )
            prompt = prompt.format(
                QUESTION=question,
                DATABASE_SCHEMA=database_schema,
                HINT=evidence,
            )
            user_message = prompt
            ai_message = "```sql\n" + gold_query + "\n```"
            training_datasets.append({
                "user_message": user_message,
                "ai_message": ai_message
            })
        else:
            raise NotImplementedError("Few-shot SFT is not implemented yet.")
    dataset = pd.DataFrame(training_datasets)
    dataset.to_csv(dataset_name)
    if not os.path.exists("finetuning_datasets"):
        os.makedirs("finetuning_datasets")
    return load_dataset('csv', data_files=dataset_name)

def formatting_prompt_func(training_dataset: Any):
    output_texts = []
    user_messages = training_dataset["user_message"]
    ai_messages = training_dataset["ai_message"]
    for user_message, ai_message in zip(user_messages, ai_messages):
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ai_message}
        ]
        output_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return output_texts

def filter_samples_based_on_length(example: Any, max_seq_length: int = 8192):
    user_messages = example["user_message"]
    ai_messages = example["ai_message"]
    messages = [
        {"role": "user", "content": user_messages},
        {"role": "assistant", "content": ai_messages}
    ]
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)) <= max_seq_length

def train_model(dataset: Any, args: argparse.Namespace, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    dataset = dataset['train'].train_test_split(test_size=0.01, shuffle=True)
    dataset = DatasetDict({
        'train': dataset['train'],
        'validation': dataset['test']
    })
    collator = DataCollatorForCompletionOnlyLM(args.response_template, tokenizer=tokenizer)
    model.config.use_cache = False

    # Training Arguments
    lora_r = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = 0.1
    output_dir = f"models/{args.adapter_name}"  
    adapter_path = f"adapters/{args.adapter_name}"
    if not os.path.exists(adapter_path):
        os.makedirs(adapter_path)  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_train_epochs = args.num_train_epochs
    bf16 = True
    overwrite_output_dir = True
    per_device_train_batch_size = args.batch_size
    per_device_eval_batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    gradient_checkpointing = True
    eval_strategy = "steps"
    learning_rate = 1e-4
    weight_decay = 0.01
    lr_scheduler_type = "cosine"
    gradient_checkpointing_kwargs = {"use_reentrant": False}
    warmup_ratio = 0.1
    max_grad_norm = 0.5
    group_by_length = True
    auto_find_batch_size = False
    save_steps = 100
    logging_steps = 100
    load_best_model_at_end = True
    packing = False
    save_total_limit = 1
    max_seq_length = args.max_seq_length

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
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

    training_arguments = SFTConfig(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        num_train_epochs=num_train_epochs,
        load_best_model_at_end=load_best_model_at_end,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_strategy=eval_strategy,
        max_grad_norm=max_grad_norm,
        auto_find_batch_size=auto_find_batch_size,
        save_total_limit=save_total_limit,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        bf16=bf16,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        packing=packing,
        report_to="wandb",
        max_seq_length=max_seq_length,
    )

    trainer = SFTTrainer(
            model=model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            peft_config=peft_config,
            formatting_func=formatting_prompt_func,
            data_collator=collator,
            tokenizer=tokenizer,
            args=training_arguments,
        )
    trainer.train()

    # save lora adapter
    trainer.save_model(adapter_path)


    if args.merge_adapter:
        # merge adapter
        new_model_name = f"{args.hf_username}/{args.adapter_name}"
        trainer.tokenizer.push_to_hub(new_model_name)

        del model
        del trainer
        torch.cuda.empty_cache()

        base_model = load_model(args.model_name)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()

        model.push_to_hub(new_model_name)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    args.add_argument("--adapter_name", type=str, default="Qwen2.5-Coder-3B-Instruct")
    args.add_argument("--lora_rank", type=int, default=64)
    args.add_argument("--lora_alpha", type=float, default=64)
    args.add_argument("--zero_shot", type=bool, default=True)
    args.add_argument("--num_train_epochs", type=int, default=3)
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--gradient_accumulation_steps", type=int, default=32)
    args.add_argument("--response_template", type=str, default="assistant")
    args.add_argument("--max_seq_length", type=int, default=6000)
    args.add_argument("--merge_adapter", type=bool, default=True)
    args.add_argument("--hf_username", type=str, default="MrezaPRZ")

    args = args.parse_args()

    model = load_model(args.model_name)
    tokenizer = load_tokenizer(args.model_name)
    dataset = construct_fineutning_dataset(args.zero_shot)
    dataset = dataset.filter(filter_samples_based_on_length, fn_kwargs={'max_seq_length': args.max_seq_length})
    train_model(dataset, args, tokenizer, model)