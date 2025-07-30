# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    if data_args.emb_lora:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "row_ids": [], "col_ids": []}

        from copy import deepcopy
        ori_tokenizer = deepcopy(tokenizer)
        tokenizer.add_tokens(["<TAB>"], special_tokens=True)
    else:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    if processor is not None:
        model_inputs["pixel_values"] = []
        if hasattr(processor, "image_seq_length"):  # paligemma models
            model_inputs["token_type_ids"] = []

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue
        
        if data_args.emb_lora:
            if len(examples["prompt"][i]) != 1:
                raise ValueError(f'prompt should have only one turn')

            pattern = r"/\*\n(.*?)\n\*/"  
            # Function to replace matched pattern and extract "table"  
            def replace_and_extract(text):  
                # Find all matches in the text  
                matches = re.findall(pattern, text, re.DOTALL)  
            
                # Replace the matched patterns  
                replaced_text = re.sub(pattern, "/*\n<TAB>\n*/", text, flags=re.DOTALL)  # we don't use pad token, because pad_token_id may = eos_token_id
            
                # Return the replaced text and the extracted matches  
                return replaced_text, matches  

            prompt = examples["prompt"][i][0]['content']
            examples["prompt"][i][0]['content'], table_texts = replace_and_extract(prompt)

        input_ids, labels = _encode_supervised_example(
            prompt=examples["prompt"][i],
            response=examples["response"][i],
            system=examples["system"][i],
            tools=examples["tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        if not data_args.emb_lora or len(table_texts) == 0:
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            if data_args.emb_lora:
                model_inputs["row_ids"].append([0] * len(input_ids))
                model_inputs["col_ids"].append([0] * len(input_ids))
        elif "<ROW>" in table_texts[0] and "<COL>" in table_texts[0]:
            row_ids_total = [0] * len(input_ids)
            col_ids_total = [0] * len(input_ids)
            for table_text in table_texts:
                # Calculate row ids and col ids
                table_token_ids = []
                col_ids = []
                row_ids = []
                table_token_ids.extend(tokenizer.encode(table_text.split("<ROW>")[0])[1:])
                row_ids.extend([0] * len(table_token_ids))
                col_ids.extend([0] * len(table_token_ids))
                row_id = 1
                col_id = 0
                for row_text in table_text.split("<ROW>")[1:]:
                    col_id = 0
                    row_pre_ids = tokenizer.encode("<ROW>"+row_text.split("<COL>")[0])[1:]
                    table_token_ids.extend(row_pre_ids)
                    row_ids.extend([row_id] * len(row_pre_ids))
                    col_ids.extend([0] * len(row_pre_ids))
                    col_id += 1
                    for col_text in row_text.split("<COL>")[1:]:
                        col_cell_ids=tokenizer.encode("<COL>"+col_text)[1:]
                        table_token_ids.extend(col_cell_ids)
                        row_ids.extend([row_id] * len(col_cell_ids))
                        col_ids.extend([col_id] * len(col_cell_ids))
                        col_id += 1
                    row_id += 1

                if input_ids.count(tokenizer.convert_tokens_to_ids("<TAB>")) ==0: # truncate the table token
                    break 
                table_insert_idx = input_ids.index(tokenizer.convert_tokens_to_ids("<TAB>")) # The first <TAB> token
                row_ids_total = row_ids_total[:table_insert_idx] + row_ids + row_ids_total[table_insert_idx+1:]
                col_ids_total = col_ids_total[:table_insert_idx] + col_ids + col_ids_total[table_insert_idx+1:]
                if data_args.train_on_prompt:
                    labels = labels[:table_insert_idx] + table_token_ids + labels[table_insert_idx+1:]
                else:
                    labels = labels[:table_insert_idx] + [IGNORE_INDEX] * len(table_token_ids) + labels[table_insert_idx+1:]
                input_ids = input_ids[:table_insert_idx] + table_token_ids + input_ids[table_insert_idx+1:]
            model_inputs["row_ids"].append(row_ids_total)
            model_inputs["col_ids"].append(col_ids_total)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            if len(input_ids) != len(row_ids_total) != len(col_ids_total):
                raise ValueError(f'input_ids {len(input_ids)}, row_ids {len(row_ids_total)}, col_ids {len(col_ids_total)} should have the same length')
            if tokenizer.convert_tokens_to_ids("<TAB>") in tokenizer.convert_ids_to_tokens(input_ids):
                raise ValueError(f'input_ids should not have <TAB> token')
        else: # markdown
            row_ids_total = [0] * len(input_ids)
            col_ids_total = [0] * len(input_ids)
            for table_text in table_texts:
                # Calculate row ids and col ids
                table_token_ids = []
                col_ids = []
                row_ids = []
                # table_token_ids.extend(tokenizer.encode(table_text.split("\n")[0])[1:])
                # row_ids.extend([0] * len(table_token_ids))
                # col_ids.extend([0] * len(table_token_ids))
                row_id = 1
                col_id = 0
                for row_text in table_text.split("\n"):
                    col_id = 0
                    if set(row_text) == set(["|","-",":"]):
                        row_id -= 1
                        token_ids = tokenizer.encode(row_text+"\n")[1:]
                        table_token_ids.extend(token_ids)
                        row_ids.extend([row_id] * len(token_ids))
                        col_ids.extend([0] * len(token_ids))
                        row_id += 1
                    else:
                        row_pre_ids = tokenizer.encode("|"+"|".join(row_text.split("|")[:2])+"|")[1:]
                        table_token_ids.extend(row_pre_ids)
                        row_ids.extend([row_id] * len(row_pre_ids))
                        col_ids.extend([0] * len(row_pre_ids))
                        col_id += 1
                        for col_text in row_text.split("|")[2:]:
                            col_cell_ids=tokenizer.encode(col_text+"|")[1:]
                            table_token_ids.extend(col_cell_ids)
                            row_ids.extend([row_id] * len(col_cell_ids))
                            col_ids.extend([col_id] * len(col_cell_ids))
                            col_id += 1
                        row_id += 1

                if input_ids.count(tokenizer.convert_tokens_to_ids("<TAB>")) ==0: # truncate the table token
                    break 
                table_insert_idx = input_ids.index(tokenizer.convert_tokens_to_ids("<TAB>")) # The first <TAB> token
                row_ids_total = row_ids_total[:table_insert_idx] + row_ids + row_ids_total[table_insert_idx+1:]
                col_ids_total = col_ids_total[:table_insert_idx] + col_ids + col_ids_total[table_insert_idx+1:]
                if data_args.train_on_prompt:
                    labels = labels[:table_insert_idx] + table_token_ids + labels[table_insert_idx+1:]
                else:
                    labels = labels[:table_insert_idx] + [IGNORE_INDEX] * len(table_token_ids) + labels[table_insert_idx+1:]
                input_ids = input_ids[:table_insert_idx] + table_token_ids + input_ids[table_insert_idx+1:]
            model_inputs["row_ids"].append(row_ids_total)
            model_inputs["col_ids"].append(col_ids_total)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            if len(input_ids) != len(row_ids_total) != len(col_ids_total):
                raise ValueError(f'input_ids {len(input_ids)}, row_ids {len(row_ids_total)}, col_ids {len(col_ids_total)} should have the same length')  
            if tokenizer.convert_tokens_to_ids("<TAB>") in tokenizer.convert_ids_to_tokens(input_ids):
                raise ValueError(f'input_ids should not have <TAB> token')

        if processor is not None:
            model_inputs["pixel_values"].append(get_pixel_values(examples["images"][i], processor))
            if hasattr(processor, "image_seq_length"):  # paligemma models
                model_inputs["token_type_ids"].append(get_paligemma_token_type_ids(len(input_ids), processor))
    if data_args.emb_lora:
        tokenizer = ori_tokenizer
    return model_inputs