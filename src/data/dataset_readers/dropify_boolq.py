
import argparse
import json
import os


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("data_path", type=str, help="path to a BoolQ dataset file")
    parse.add_argument("--output_path", type=str, default="")
    return parse.parse_args()


def main():
    args = get_args()

    drop_data = {}
    with open(args.data_path, "r") as fd:
        data = [json.loads(line) for line in fd]

    # use the filename as base string for example ids.
    data_basename = os.path.basename(args.data_path)

    for record_i, record in enumerate(data):
        # use the following id both as qid and pid
        qid = f"{data_basename}_{record_i}"

        question = record["question"] + "? yes no"
        if record["answer"] is True:
            answer = "yes"
        elif record["answer"] is False:
            answer = "no"
        else:
            raise AssertionError

        drop_data[qid] = {
            "passage": record["passage"],
            "qa_pairs": [{
                "query_id": qid,
                "question": question,
                "answer": {
                    "date": {"day": "", "month": "", "year": ""},
                    "number": "",
                    "spans": [answer]
                }
            }],
            "wiki_url": record["title"]
        }

    if args.output_path != "":
        output_path = args.output_path
    else:
        output_path = args.data_path.replace(".json", "_drop_format.json")
    with open(output_path, 'w') as fd:
        json.dump(drop_data, fd, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
