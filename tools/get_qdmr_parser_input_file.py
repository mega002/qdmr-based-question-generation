
import argparse
import csv
import json


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--data-file-path", type=str, help="path to a dataset file", required=True)
    parse.add_argument("--split", choices=["train", "dev"], required=True)
    parse.add_argument("--data-format", choices=["iirc", "drop", "hotpot-squad"], required=True)
    return parse.parse_args()


def main():
    args = get_args()

    records = []
    with open(args.data_file_path, "r") as fd:
        data = json.load(fd)
    if args.data_format == "iirc":
        for para in data:
            for qa in para["questions"]:
                records.append({
                    'question_id': f"IIRC_{args.split}_{para['pid']}_{qa['qid']}",
                    'question': qa['question']
                })
    elif args.data_format == "drop":
        for para_id in data:
            for qa_pair in data[para_id]["qa_pairs"]:
                records.append({
                    'question_id': f"DROP_{args.split}_{para_id}_{qa_pair['query_id']}",
                    'question': qa_pair['question']
                })
    elif args.data_format == "hotpot-squad":
        for page in data["data"]:
            for para in page["paragraphs"]:
                for qa_pair in para["qas"]:
                    records.append({
                        'question_id': f"HOTPOT_{args.split}_{qa_pair['id']}",
                        'question': qa_pair['question']
                    })

    output_path = args.data_file_path.replace(".json", "_decomp_input.csv")
    with open(output_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['question_id', 'source'])
        for record in records:
            writer.writerow([record['question_id'], record['question']])


if __name__ == "__main__":
    main()
