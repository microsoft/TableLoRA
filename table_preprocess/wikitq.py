# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datasets import load_dataset
import pandas as pd
from llamafactory.table_lora.prompt_tuning import prompt_tuning_table_prompt, TABLE_TOKEN
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import shutil


def preprocess_table(dataset, tokenizer, prompt_tuning=False, max_length=1024, save_original_table=False, pretraining=False):
    dataset_preprocessed = []
    Max_col=0
    Max_row=0

    if pretraining:
        table_string_set = set()
    for i in tqdm(range(len(dataset))):
        table = dataset[i]["table"]
        question = dataset[i]["question"]
        answer = dataset[i]["answers"]
        table=pd.DataFrame(table["rows"],columns=table["header"])

        if not prompt_tuning:
            table_string = table.to_markdown()
        else:
            table_string = prompt_tuning_table_prompt(table)
        
        Max_col = max(Max_col, len(table.columns))
        Max_row = max(Max_row, len(table_string.split("\n")))

        table_string = tokenizer.decode(tokenizer(table_string, max_length=int(max_length*0.9),truncation=True)["input_ids"][1:-1])

        prompt = "\n".join(
            [
                "Here is the table to answer this question. Answer the question.",
                "/*",
                table_string,
                "*/",
                f"Question: {question}",
                "The answer is:"
                # table.to_markdown(),
                "\r\n",
            ]
        )
        if not save_original_table and not pretraining:
            dataset_preprocessed.append({
                "prompt": prompt,
                "response": ",".join(answer)
            })
        elif pretraining:
            if table_string in table_string_set:
                continue
            else:
                table_string_set.add(table_string)
            prompt = "\n".join(
            [
                "Here is the table to answer this question. Answer the question.",
                "/*",
                table_string,
                "*/",
                f"Question: How many columns are there in the table?",
                "The answer is:"
                # table.to_markdown(),
                "\r\n",
            ]
        )
            dataset_preprocessed.append({
                "prompt": prompt,
                "response": str(len(table.columns))
            })
            prompt = "\n".join(
            [
                "Here is the table to answer this question. Answer the question.",
                "/*",
                table_string,
                "*/",
                f"Question: How many rows are there in the table (excluding the headers)?",
                "The answer is:"
                # table.to_markdown(),
                "\r\n",
            ]
        )
            dataset_preprocessed.append({
                "prompt": prompt,
                "response": str(len(table))
            })
            col_idx = int(len(table.columns)/2)
            prompt = "\n".join(
            [
                "Here is the table to answer this question. Answer the question.",
                "/*",
                table_string,
                "*/",
                f"Question: What is the header for the column with column_idx {col_idx}? column_idx counts from 0.",
                "The answer is:"
                # table.to_markdown(),
                "\r\n",
            ]
        )
            dataset_preprocessed.append({
                "prompt": prompt,
                "response": table.columns[col_idx]
            })
        else: 
            dataset_preprocessed.append({
                "prompt": prompt,
                "response": ",".join(answer),
                "table": dataset[i]["table"],
                "question": dataset[i]["question"],
                "answers": dataset[i]["answers"]
            })
    print("Max_col: ",Max_col)
    print("Max_row: ",Max_row)
    return dataset_preprocessed

def preprocess_wikitq(model_name="meta-llama/Llama-2-7b-chat-hf", max_length=1024, prompt_tuning=False, save_original_table=False, pretraining=False):
    dataset = load_dataset("wikitablequestions", trust_remote_code=True)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        add_eos_token=True,  
        add_bos_token=True,  
    )
    if prompt_tuning:
        new_token=[i.value for i in TABLE_TOKEN]
        num_added_toks = tokenizer.add_tokens(new_token, special_tokens=True)
        print("We have added", num_added_toks, "tokens: ", new_token)

    if not pretraining:
        print("Preprocessing train dataset")
        train_dataset_preprocessed = preprocess_table(train_dataset, tokenizer, prompt_tuning, max_length, save_original_table, pretraining)        
        with open("data/wikitq_train.json", "w") as f:
        # with open("/scratch/table_bench/wikitq/wikitq_train.json", "w") as f:
            json.dump(train_dataset_preprocessed, f)
        print(f"Train dataset ({len(train_dataset_preprocessed)}) preprocessed and saved to data/wikitq_train.json")

        print("Preprocessing test dataset")
        test_dataset_preprocessed = preprocess_table(test_dataset, tokenizer, prompt_tuning, max_length, save_original_table, pretraining)
        with open("data/wikitq_test.json", "w") as f:
        # with open("/scratch/table_bench/wikitq/wikitq_test.json", "w") as f:
            json.dump(test_dataset_preprocessed, f)
        print(f"Test dataset ({len(test_dataset_preprocessed)}) preprocessed and saved to data/wikitq_test.json")
    elif pretraining:
        train_dataset_preprocessed = preprocess_table(train_dataset, tokenizer, prompt_tuning, max_length, save_original_table, pretraining)      
        with open("data/wikitq_pretraining.json", "w") as f:
            json.dump(train_dataset_preprocessed, f)
        print(f"Train dataset ({len(train_dataset_preprocessed)}) preprocessed and saved to data/wikitq_pretraining.json")  

        test_dataset_preprocessed = preprocess_table(test_dataset, tokenizer, prompt_tuning, max_length, save_original_table, pretraining)
        with open("data/wikitq_pretraining_test.json", "w") as f:
            json.dump(test_dataset_preprocessed, f)
        print(f"Test dataset ({len(test_dataset_preprocessed)}) preprocessed and saved to data/wikitq_pretraining_test.json")

