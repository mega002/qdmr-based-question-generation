
import argparse
import json
import os


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("data_path", type=str, help="path to a SQuAD 2.0 dataset file")
    parse.add_argument("--output_path", type=str, default="")
    return parse.parse_args()


def main():
    args = get_args()

    drop_data = {}
    with open(args.data_path, "r") as fd:
        data_json = json.load(fd)
        data = data_json["data"]

    for article in data:
        for i, paragraph in enumerate(article["paragraphs"]):
            qa_pairs = []
            for qa in paragraph["qas"]:
                if qa["is_impossible"] is False:
                    answer = qa["answers"][0]["text"]
                    qa_pair = {
                        "query_id": qa["id"],
                        "question": qa["question"],
                        "answer": {
                            "date": {"day": "", "month": "", "year": ""},
                            "number": "",
                            "spans": [answer]
                        }
                    }
                    validated_answers = [candidate_answer["text"] for candidate_answer in qa["answers"]]
                    if "plausible_answers" in qa:
                        validated_answers.extend(
                            [candidate_answer["text"] for candidate_answer in qa["plausible_answers"]]
                        )
                    validated_answers = [
                        {
                            "date": {"day": "", "month": "", "year": ""},
                            "number": "",
                            "spans": [validated_answer]
                        }
                        for validated_answer in list(set(validated_answers))
                        if validated_answer != answer
                    ]
                    if len(validated_answers) > 0:
                        qa_pair["validated_answers"] = validated_answers
                    qa_pairs.append(qa_pair)

            pid = f'{article["title"]}_{i}'
            drop_data[pid] = {
                "passage": paragraph["context"],
                "qa_pairs": qa_pairs,
                "wiki_url": article["title"]
            }

    if args.output_path != "":
        output_path = args.output_path
    else:
        output_path = args.data_path.replace(".json", "_drop_format.json")
    with open(output_path, 'w') as fd:
        json.dump(drop_data, fd, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
