# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datasets import load_dataset
import pandas as pd
from llamafactory.table_lora.prompt_tuning import prompt_tuning_table_prompt, TABLE_TOKEN
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import shutil


def preprocess_table(dataset, tokenizer, prompt_tuning=False, max_length=1024, save_original_table=False):
    dataset_preprocessed = []
    Max_col=0
    Max_row=0
    for i in tqdm(range(len(dataset))):
        table = dataset[i]["table_array"]
        question = dataset[i]["question"]
        answer = dataset[i]["answer"]
        table=pd.DataFrame(table[1:],columns=table[0])

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
                f"Table page title: {dataset[i]['table_page_title']}",
                f"Table section title: {dataset[i]['table_section_title']}",
                f"Question: {question}",
                "The answer is:"
                # table.to_markdown(),
                "\r\n",
            ]
        )
        if not save_original_table:
            dataset_preprocessed.append({
                "prompt": prompt,
                "response": answer
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

def preprocess_fetaqa(model_name="meta-llama/Llama-2-7b-chat-hf", max_length=1024, prompt_tuning=False, save_original_table=False):
    dataset = load_dataset("DongfuJiang/FeTaQA", trust_remote_code=True)
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

    print("Preprocessing train dataset")
    train_dataset_preprocessed = preprocess_table(train_dataset, tokenizer, prompt_tuning, max_length, save_original_table)        
    with open("data/fetaqa_train.json", "w") as f:
    # with open("/scratch/table_bench/fetaqa/fetaqa_train.json", "w") as f:
        json.dump(train_dataset_preprocessed, f)
    print(f"Train dataset ({len(train_dataset_preprocessed)}) preprocessed and saved to data/fetaqa_train.json")

    print("Preprocessing test dataset")
    test_dataset_preprocessed = preprocess_table(test_dataset, tokenizer, prompt_tuning, max_length, save_original_table)
    with open("data/fetaqa_test.json", "w") as f:
    # with open("/scratch/table_bench/fetaqa/fetaqa_test.json", "w") as f:
        json.dump(test_dataset_preprocessed, f)
    print(f"Test dataset ({len(test_dataset_preprocessed)}) preprocessed and saved to data/fetaqa_test.json")
