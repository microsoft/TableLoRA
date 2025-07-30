# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datasets import load_dataset, Features, Value
import pandas as pd
from llamafactory.table_lora.prompt_tuning import prompt_tuning_table_prompt, TABLE_TOKEN
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from table_preprocess.hitab_table_type import table_deep

question2table_id = {}
with open("src/table_preprocess/hitab/train_samples.jsonl") as f:
    train_samples = [json.loads(line) for line in f]

for sample in train_samples:
    answer = sample["answer"][0] if len(sample["answer"]) == 1 else ", ".join([f"<{a}>" for a in sample["answer"]])
    question_answer = sample["question"] + ", " + str(answer)
    if question_answer in question2table_id and question2table_id[question_answer] != sample["table_id"]:
        print(f"Duplicate question-answer pair: {question_answer}")
    question2table_id[question_answer] = sample["table_id"]

def preprocess_table(dataset, tokenizer, prompt_tuning=False, max_length=1024, save_original_table=False, pretraining=False):
    dataset_preprocessed = []
    Max_col=0
    Max_row=0

    if pretraining:
        table_string_set = set()
    for i in tqdm(range(len(dataset))):
        question = dataset[i]["question"]
        answer = dataset[i]["output"]
    
        if "table_id" not in dataset[i]:
            table_id = question2table_id[question + ", " + answer]
        else:
            table_id = dataset[i]["table_id"]
        with open(f"src/table_preprocess/hitab/hmt/{table_id}.json") as f:
            table_data = json.load(f)
        left_deep = table_deep(table_data['left_root'])
        top_deep = table_deep(table_data['top_root'])

        input_seg = dataset[i]["input_seg"]
        if " | | " in input_seg:
            input_seg = input_seg.replace(" | | "," |  | ")
        caption = input_seg.split(" [TAB] ")[0].split(" [TLE] ")[1]
        try:
            table=pd.DataFrame([row.split(" | ")[1:-1] for row in input_seg.split("[TAB]")[1].split("[SEP]")[1:]],columns=input_seg.split("[TAB]")[1].split("[SEP]")[0].split(" | ")[1:-1])
        except Exception as e:
            print(e)
            print(input_seg)
            raise ValueError("Error in parsing table")
        if not prompt_tuning:
            # table_string = table.to_markdown()
            # table.columns = [f"(0, {i}) {col}" for i, col in enumerate(table.columns)]  
            # table = table.astype(str)  
            # rows, cols = table.shape 
            # for row in range(rows):  
            #     for col in range(cols):  
            #         table.iat[row, col] = f"({row + 1}, {col}) {table.iat[row, col]}"  
            table_string = table.to_markdown()
            # table_string = table.to_html(index=False)
            # table_string = table.to_csv()
            # table_string = table.to_latex()
            # table_string = table.to_json(orient="split")
            # table_string = table.to_dict(orient="split")
        else:
            table_string = prompt_tuning_table_prompt(table)
        
        Max_col = max(Max_col, len(table.columns))
        Max_row = max(Max_row, len(table_string.split("\n")))

        table_string = tokenizer.decode(tokenizer(table_string, max_length=int(max_length*0.9),truncation=True)["input_ids"][1:-1])

        prompt = "\n".join(
            [
                "This is a hierarchical table question answering task. The goal for this task is to answer the given question based on the given table. The table might be hierarchical. Here is the table to answer this question. Answer the question.",
                "/*",
                table_string,
                "*/",
                f"Table Caption: {caption}",
                f"Question: {question}",
                "The answer is:"
                # table.to_markdown(),
                "\r\n",
            ]
        )
        if not save_original_table and not pretraining:
            dataset_preprocessed.append({
                "prompt": prompt,
                "response": answer
            })
        elif pretraining:
            if table_string in table_string_set:
                continue
            table_string_set.add(table_string)
            prompt = "\n".join(
            [
                "Here is the table to answer this question. Answer the question.",
                "/*",
                table_string,
                "*/",
                f"Question: How deep are the top header layers?",
                "The answer is:"
                # table.to_markdown(),
                "\r\n",
            ]
        )
            dataset_preprocessed.append({
                "prompt": prompt,
                "response": str(top_deep)
            })
            prompt = "\n".join(
            [
                "Here is the table to answer this question. Answer the question.",
                "/*",
                table_string,
                "*/",
                f"Question: How deep are the left header layers? (at least 1)",
                "The answer is:"
                # table.to_markdown(),
                "\r\n",
            ]
        )
            dataset_preprocessed.append({
                "prompt": prompt,
                "response": str(left_deep)
            })
            col_idx = int(len(table_data["data"][0])/2)
            row_idx = int(len(table_data["data"])/4)
            prompt = "\n".join(
            [
                "Here is the table to answer this question. Answer the question.",
                "/*",
                table_string,
                "*/",
                f"Question: What is the {col_idx} column {row_idx} row cell value excluding the top header and left header? column_idx and row_idx counts from 0.",
                "The answer is:"
                # table.to_markdown(),
                "\r\n",
            ]
        )
            dataset_preprocessed.append({
                "prompt": prompt,
                "response": str(table_data["data"][row_idx][col_idx]['value'])
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

def preprocess_hitab(model_name="meta-llama/Llama-2-7b-chat-hf", max_length=4000, prompt_tuning=False, save_original_table=False, pretraining=False):
    train_dataset = load_dataset("osunlp/TableInstruct",data_files="data_v3/hitab_train_7417.json", split="train", trust_remote_code=True)
    # This script has bugs!!
    # test_dataset = load_dataset("osunlp/TableInstruct",
    #                             data_files="eval_data/in_domain_test/hitab_test.json", 
    #                             split="test", 
    #                             trust_remote_code=True,
    #                             features=Features({ 
    #                         'table_id': Value('string'),
    #                         'instruction': Value('string'),
    #                         "input": Value('string'),
    #                         "question": Value('string'),
    #                         'output': Value('string'),
    #                         'raw_answer': Value('string'),
    #                         'input_seg': Value('string'),
    #                    }))
    with open("src/table_preprocess/hitab/hitab_test.json") as f:
        test_dataset = json.load(f)

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
        with open("data/hitab_train.json", "w") as f:
            json.dump(train_dataset_preprocessed, f)
        print(f"Train dataset ({len(train_dataset_preprocessed)}) preprocessed and saved to data/hitab_train.json")

        print("Preprocessing test dataset")
        test_dataset_preprocessed = preprocess_table(test_dataset, tokenizer, prompt_tuning, max_length, save_original_table, pretraining)
        with open("data/hitab_test.json", "w") as f:
            json.dump(test_dataset_preprocessed, f)
        print(f"Test dataset ({len(test_dataset_preprocessed)}) preprocessed and saved to data/hitab_test.json")
    elif pretraining:
        train_dataset_preprocessed = preprocess_table(train_dataset, tokenizer, prompt_tuning, max_length, save_original_table, pretraining)      
        with open("data/hitab_pretraining.json", "w") as f:
            json.dump(train_dataset_preprocessed, f)
        print(f"Train dataset ({len(train_dataset_preprocessed)}) preprocessed and saved to data/hitab_pretraining.json")  

        test_dataset_preprocessed = preprocess_table(test_dataset, tokenizer, prompt_tuning, max_length, save_original_table, pretraining)
        with open("data/hitab_pretraining_test.json", "w") as f:
            json.dump(test_dataset_preprocessed, f)
        print(f"Test dataset ({len(test_dataset_preprocessed)}) preprocessed and saved to data/hitab_pretraining_test.json")

if __name__ == '__main__':
    preprocess_hitab()