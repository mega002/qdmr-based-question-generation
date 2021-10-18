
import argparse
import copy
import json


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--break_data_path", type=str, help="predictions path", required=True)
    parse.add_argument("--orig_data_path", type=str, help="data path", required=True)
    parse.add_argument("--data_format", choices=['drop', 'hotpot-squad'], help="data path", required=True)
    parse.add_argument("--output_path", type=str, default="", required=True)

    return parse.parse_args()


def load_break_qids(break_data_path, data_format):
    break_qids = {}
    with open(break_data_path, "r") as fd:
        for line in fd.readlines()[1:]:
            question_id = line.split(",")[0]
            dataset, _, qid = question_id.split("_", 2)
            if dataset.lower() in data_format:
                break_qids[qid] = True

    print(f"loaded {len(break_qids)} qids for {data_format} from: {break_data_path}")

    return break_qids


def extract_drop_subset(break_qids, orig_data_path, just_qids=False):
    with open(orig_data_path, "r") as fd:
        drop_json = json.load(fd)

    num_remaining_examples = 0
    passage_id_to_pop = []
    for passage_id in drop_json:
        for qa_pair_i in range(len(drop_json[passage_id]["qa_pairs"]))[::-1]:
            qa_pair = drop_json[passage_id]["qa_pairs"][qa_pair_i]
            if just_qids:
                orig_id = qa_pair['query_id']
            else:
                orig_id = f"{passage_id}_{qa_pair['query_id']}"
            if orig_id not in break_qids:
                del drop_json[passage_id]["qa_pairs"][qa_pair_i]

        num_remaining_examples += len(drop_json[passage_id]["qa_pairs"])

        if len(drop_json[passage_id]["qa_pairs"]) == 0:
            passage_id_to_pop.append(passage_id)

    for passage_id in passage_id_to_pop:
        drop_json.pop(passage_id)

    print(f"ended with a subset of {num_remaining_examples} drop examples.")

    return drop_json


def extract_hotpotqa_subset(break_qids, orig_data_path):
    with open(orig_data_path, "r") as fd:
        hotpot_json = json.load(fd)

    num_remaining_examples = 0
    for page_i in range(len(hotpot_json["data"]))[::-1]:
        page = hotpot_json["data"][page_i]
        for paragraph_i in range(len(page["paragraphs"]))[::-1]:
            for qa_i in range(len(page["paragraphs"][paragraph_i]["qas"]))[::-1]:
                if page["paragraphs"][paragraph_i]["qas"][qa_i]["id"] not in break_qids:
                    del page["paragraphs"][paragraph_i]["qas"][qa_i]
            num_remaining_examples += len(page["paragraphs"][paragraph_i]["qas"])
            if not page["paragraphs"][paragraph_i]["qas"]:
                del page["paragraphs"][paragraph_i]
        if not page["paragraphs"]:
            del hotpot_json["data"][page_i]

    print(f"ended up with a subset of {num_remaining_examples} hotpotqa examples.")

    return hotpot_json


def main():
    args = get_args()

    break_qids = load_break_qids(args.break_data_path, args.data_format)

    data = None
    if args.data_format == "drop":
        data = extract_drop_subset(break_qids, args.orig_data_path)
    elif args.data_format == "hotpot-squad":
        data = extract_hotpotqa_subset(break_qids, args.orig_data_path)

    if data is not None:
        with open(args.output_path, "w") as fd:
            json.dump(data, fd, indent=4)
        print(f"wrote {args.data_format}-format predictions to: {args.output_path}")


if __name__ == "__main__":
    main()
