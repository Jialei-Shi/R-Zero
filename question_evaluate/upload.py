from pathlib import Path
from tempfile import TemporaryDirectory
from datasets import Dataset, DatasetDict
from huggingface_hub import login
import argparse
import json
import os
STORAGE_PATH = os.getenv("STORAGE_PATH")
HUGGINGFACENAME = os.getenv("HUGGINGFACENAME")
print(STORAGE_PATH)
parser = argparse.ArgumentParser()
destination = parser.add_mutually_exclusive_group()
destination.add_argument("--repo_name", type=str, default="")
destination.add_argument("--local_output", type=Path)
parser.add_argument("--max_score", type=float, default=0.7)
parser.add_argument("--min_score", type=float, default=0.3)
parser.add_argument("--experiment_name", type=str, default="Qwen_Qwen3-4B-Base_all")
parser.add_argument("--num_shards", type=int, default=8)
args = parser.parse_args()

if args.local_output is None:
    with open('tokens.json', 'r') as f:
        token = json.load(f)['huggingface']
    login(token=token)

datas= []
for i in range(args.num_shards):
    try:
        with open(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json', 'r') as f:
            data = json.load(f)
            datas.extend(data)
    except FileNotFoundError:
        print(f"File {args.experiment_name}_{i}_results.json not found")
        continue


if args.local_output is None:
    for i in range(args.num_shards):
        try:
            os.remove(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json')
        except FileNotFoundError:
            print(f"File {args.experiment_name}_{i}_results.json not found")

scores = [data['score'] for data in datas]
#  print the distribution of scores
import matplotlib.pyplot as plt
plt.hist(scores, bins=11)
plt.savefig('scores_distribution.png')

#count the number  of score between 0.2 and 0.8 
if args.local_output is not None or args.repo_name:
    filtered_datas = [{'problem':data['question'],'answer':data['answer'],'score':data['score']} for data in datas if data['score'] >= args.min_score and data['score'] <= args.max_score and data['answer'] != '' and data['answer']!= 'None']
    print(len(filtered_datas))
    if args.local_output is not None and not filtered_datas:
        raise SystemExit(f"No questions passed filtering for {args.experiment_name}")
    train_dataset = Dataset.from_list(filtered_datas)
    if args.local_output is not None:
        args.local_output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.local_output.parent) as temp_dir:
            temp_path = Path(temp_dir) / args.local_output.name
            train_dataset.to_parquet(str(temp_path))
            temp_path.replace(args.local_output)
        print(f"Saved {len(train_dataset)} questions to {args.local_output}")
    else:
        dataset = DatasetDict({"train": train_dataset})
        dataset.push_to_hub(
            f"{HUGGINGFACENAME}/{args.repo_name}", private=True, config_name=args.experiment_name
        )







