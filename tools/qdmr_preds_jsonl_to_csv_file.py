
import argparse
import csv
import json

from src.data.dataset_readers.standardization_utils import fix_references_back


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--predictions_path", type=str, help="predictions path (jsonl file)", required=True)
    parse.add_argument("--output_path", type=str, help="final predictions csv file path", required=True)
    parse.add_argument("--fix_refs_back", action="store_true", help="convert @@2@@ to #2 in predicted decompositions.")
    return parse.parse_args()


def main():
    args = get_args()

    records = [json.loads(line) for line in open(args.predictions_path, "r")]

    with open(args.output_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['question_id', 'question_text', 'decomposition', 'operators', 'split'])
        for record in records:
            steps = ['return ' + step for step in record["decomposition"]]
            decomposition = ' ;'.join(steps)
            if args.fix_refs_back is True:
                decomposition = fix_references_back(decomposition)
            writer.writerow([record['qid'], record['question'], decomposition, '', ''])


if __name__ == "__main__":
    main()
