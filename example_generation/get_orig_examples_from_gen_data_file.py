
import argparse
import json
import numpy as np

from example_generation.utils import extract_drop_subset, extract_hotpotqa_subset
from qdmr_transforms.qdmr_example import get_orig_example_id
from src.data.dataset_readers.drop import DropReader
from src.data.dataset_readers.hotpotqa import HotpotQASQuADReader


np.random.seed(42)


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--orig_data_path", type=str, help="original data file path (json)", required=True)
    parse.add_argument("--gen_data_path", type=str, help="generated data file path (json)", required=True)
    parse.add_argument("--dataset_name", choices=['drop', 'hotpot-squad', 'iirc'], help="data format", required=True)
    parse.add_argument("--output_path", type=str, help="examples output path")

    return parse.parse_args()


def main():
    args = get_args()

    if args.dataset_name in ["drop", "iirc"]:
        reader = DropReader()
    elif args.dataset_name == "hotpot-squad":
        reader = HotpotQASQuADReader()
    else:
        raise NotImplementedError

    records = reader.get_qa_records(args.gen_data_path, keep_fields=["question_id"])
    orig_example_ids = [
        get_orig_example_id(record["question_id"], clean=True)
        for record in records
    ]
    print(f"loaded {len(orig_example_ids)} generated question ids from: {args.gen_data_path}")

    data_json = None
    if args.dataset_name in ["drop", "iirc"]:
        data_json = extract_drop_subset(orig_example_ids, args.orig_data_path, just_qids=True)
    elif args.dataset_name == "hotpot-squad":
        data_json = extract_hotpotqa_subset(orig_example_ids, args.orig_data_path)

    if args.output_path is not None:
        output_path = args.output_path
    else:
        assert args.gen_data_path.endswith(".json")
        output_path = args.gen_data_path.replace(".json", "_orig_subset.json")
    with open(output_path, "w") as fd:
        json.dump(data_json, fd, indent=4)
    print(f"\nwrote the original data subset corresponding to the generated examples: {output_path}")


if __name__ == "__main__":
    main()
