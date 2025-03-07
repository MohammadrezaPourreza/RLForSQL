import argparse
import dataclasses
import pandas as pd
import concurrent.futures
import os
import time
import json
import re
import torch

from prompts.prompt_loader import load_prompt
from utils.llm_utils import load_model, load_tokenizer, call_model, call_model_openai
from utils.execution import compare_sqls
from utils.database_manager import get_db_schema_db_id
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclasses.dataclass
class QuestionArgs:
  """Class for question arguments."""

  question_row: pd.Series
#   model: AutoModelForCausalLM
#   tokenizer: AutoTokenizer
  model_name: str
  prompt_template: str


def extract_sql_queries(text):
    pattern = r"```sql\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        queries = [match.strip() for match in matches]
        return queries[-1]  # Return the last query
    else:
        return text


def process_sample(question_args: QuestionArgs) -> int:
    question = question_args.question_row["question"]
    db_id = question_args.question_row["db_id"]
    gold_query = question_args.question_row["SQL"]
    evidence = question_args.question_row["evidence"]
    # database_schema = get_db_schema_db_id(
    #    db_id=db_id,
    #    bird_database_path=os.getenv("BASE_DEV_DATA_PATH"),
    #    queries=[gold_query],
    # )

    #with schema linking
    schema_linking_result = pd.read_csv("schema_linking/col_selection_schema.csv")
    database_schema = schema_linking_result[schema_linking_result["question_id"] == question_args.question_row["question_id"]]["selected_schema_with_connections"].values[0]

    prompt = load_prompt(
    question_args.prompt_template
    )
    prompt = prompt.format(
        QUESTION=question,
        DATABASE_SCHEMA=database_schema,
        HINT=evidence,
    )
    # model_reponse = call_model(
    #     model=question_args.model,
    #     tokenizer=question_args.tokenizer,
    #     user_message=prompt,
    #     max_new_tokens=800,
    #     do_sample=False,
    #     config = {
    #         "top_k": None,
    #         "top_p": None,
    #         "temperature": None,
    #     }
    # )
    model_response = call_model_openai(
        model_name=  args.model_name,
        user_message=prompt,
        max_new_tokens=1500,
        do_sample=False,
    )
    print(model_response)
    response = extract_sql_queries(model_response)
    accuracy = 0
    if compare_sqls(os.getenv("BASE_DEV_DATA_PATH"), db_id, response, gold_query)['exec_res']:
        print("correct")
        accuracy = 1
    else:
        print("incorrect")

    return {
        "question": question,
        "evidence": evidence,
        "db_id": db_id,
        "gold_query": gold_query,
        "predicted_query": response,
        "model_response": model_response,
        "accuracy": accuracy,
    }


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="MrezaPRZ/Qwen2.5-Coder-7B-Instruct-SQL-COT-llm_all")
    args.add_argument("--number_of_workers", type=int, default=10)
    args.add_argument("--output_results_path", type=str, default="results/dev_schema_linking")
    args.add_argument("--prompt_template", type=str, default="sql_generation_zero_shot")
    args.add_argument("--output_file_name", type=str, default="Qwen2.5-Coder-7B-Instruct-SQL-COT-llm_all")
    args = args.parse_args()

    if not os.path.exists(args.output_results_path):
        os.makedirs(args.output_results_path)

    current_time = time.strftime("%Y%m%d-%H%M%S")
    model_id = args.model_name.split("/")[-1]
    output_path = os.path.join(args.output_results_path, f"results_{current_time}_{model_id}.json")
    dataset = pd.read_json("data/dev/dev.json")
    # model = load_model(args.model_name, use_flash_attn=True)
    # tokenizer = load_tokenizer(args.model_name)
    results = []
    if args.number_of_workers == 1:
        for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
            result = process_sample(QuestionArgs(row, args.model_name, args.prompt_template))
            if result is not None:
                results.append(result)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.number_of_workers) as executor:
            futures = [executor.submit(process_sample, QuestionArgs(row, args.model_name, args.prompt_template)) for _, row in dataset.iterrows()]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(dataset)):
                result = future.result()
                if result is not None:
                    results.append(result)

    # Dump all results to file
    print("Final Accuracy: ", sum([result["accuracy"] for result in results]) / len(results))
    with open(output_path, 'w') as file:
        json.dump(results, file, indent=4)