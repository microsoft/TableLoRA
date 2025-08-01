# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

from table_preprocess.wikitq import preprocess_wikitq
from table_preprocess.fetaqa import preprocess_fetaqa
from table_preprocess.tabfact import preprocess_tabfact
from table_preprocess.hitab import preprocess_hitab

# paser
def parse_args():
    parser = argparse.ArgumentParser(description='Extract code snippets from a given codebase')
    parser.add_argument('--dataset_name', type=str, default='wikitq', help='Dataset name')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='Model name or path')
    parser.add_argument('--max_length', type=int, default=1024, help='Max length of the prompt')
    parser.add_argument('--prompt_tuning', type=bool, default=False, help='Use prompt tuning or not')
    parser.add_argument('--save_original_table', type=bool, default=False, help='Save original table or not')
    parser.add_argument('--pretraining', type=bool, default=False, help='Pretraining or not')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.dataset_name == 'wikitq':
        preprocess_wikitq(args.model_name, args.max_length, args.prompt_tuning, args.save_original_table, args.pretraining)
    elif args.dataset_name == 'fetaqa':
        preprocess_fetaqa(args.model_name, args.max_length, args.prompt_tuning, args.save_original_table)
    elif args.dataset_name == 'tabfact':
        preprocess_tabfact(args.model_name, args.max_length, args.prompt_tuning, args.save_original_table, args.pretraining)
    elif args.dataset_name == 'hitab':
        preprocess_hitab(args.model_name, args.max_length, args.prompt_tuning, args.save_original_table, args.pretraining)
    else:
        print('Dataset not supported')