
import argparse
import json
import numpy as np
import pandas as pd

from qdmr_transforms.qdmr_example import get_transform_base_info

np.random.seed(42)

dataset_to_num_train_examples = {
    "drop": 77409,
    "hotpot-squad": 90447,
    "iirc": 10849,
}

hard_sampling_conf = {
    "drop": {
        "append_boolean_step": 247,
        "change_last_step": 38,
        "op_replace_arithmetic": 152,
        "op_replace_comparison": 221,
        "prune_last_step": 198,
        "prune_last_step_rm_unused": 194,
    },
    "hotpot-squad": {
        "append_boolean_step": 196,
        "change_last_step": 169,
        "op_replace_boolean": 124,
        "op_replace_comparison": 177,
        "prune_last_step": 180,
        "prune_last_step_rm_unused": 169,
        "prune_step": 164,
    },
    "iirc": {
        "append_boolean_step": 194,
        "change_last_step": 33,
        "op_replace_arithmetic": 0,
        "op_replace_boolean": 1,
        "op_replace_comparison": 10,
        "prune_last_step": 184,
        "prune_last_step_rm_unused": 83,
    },
}



def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--example_info_path", type=str, help="predictions path", required=True)
    parse.add_argument("--data_format", choices=['drop', 'hotpot-squad', 'iirc'], help="data path", required=True)
    parse.add_argument("--output_path", type=str, help="examples output path", required=True)
    parse.add_argument("--append_yes_no", action="store_true",
                       help="Whether to append yes/no to the question in DROP/IIRC examples")
    parse.add_argument("--allow_yes_no", action="store_true",
                       help="Include examples with yes/no answer when generating DROP examples")
    parse.add_argument("--sampling", type=int, default=-1,
                       help="Maximum number of examples per transformation, "
                            "in terms of percentage of the training data (default -1 means no sampling)")
    parse.add_argument("--hard_sampling", action="store_true",
                       help="Sample hard-coded number of examples from each transformation.")
    parse.add_argument("--validated_qids_path", type=str, default="",
                       help="Path to a text file with qids separated by newlines. "
                            "If a path is given, then the output file will include only the subset of these qids "
                            "(sampling is done after this filtering of qids).")

    return parse.parse_args()


def answer_to_drop_format(record, allow_yes_no=False):
    answer = record["transformed_answer"]
    answer_dict = {
        "number": "",
        "date": {
            "day": "",
            "month": "",
            "year": ""
        },
        "spans": []
    }

    if type(answer) in [int, float]:
        answer_dict["number"] = str(answer)
    else:
        assert type(answer) in [str, list]
        if type(answer) == str:
            answer_list = [answer]
        else:
            answer_list = answer

        is_answer_in_context = sum([
            span_answer in record["question"] or answer in record["passage"]
            for span_answer in answer_list
        ]) == len(answer_list)
        if allow_yes_no:
            assert is_answer_in_context or answer in ["yes", "no"]
        else:
            # explicitly check that the answer is not yes/no because "no" is likely
            # to appear in the context as a sub-word (e.g. "no" in "know" --> True).
            assert is_answer_in_context and answer not in ["yes", "no"]
        answer_dict["spans"].extend(answer_list)

    return answer_dict


def answer_to_hotpot_squad_format(record):
    answer = str(record["transformed_answer"])

    answer_dict = {
        "text": str(answer),
        "answer_start": record["passage"].find(answer)
    }

    return answer_dict


def get_drop_data(records, allow_yes_no=False, append_yes_no=False):
    drop_data = {}
    no_ans_count = 0
    final_records = []
    for record in records:
        if record["passage_id"] not in drop_data:
            drop_data[record["passage_id"]] = {
                "passage": record["passage"],
                "qa_pairs": [],
                "wiki_url": ""
            }

        question = record["generated_question"]
        if append_yes_no:
            question = question + " yes no"
        qa_pair = {
            "question": question,
            "query_id": record["qid"],
            # "validated_answers": [],
        }
        if "transformed_answer" in record:
            try:
                qa_pair["answer"] = answer_to_drop_format(record, allow_yes_no=allow_yes_no)
                final_records.append(record)
                drop_data[record["passage_id"]]["qa_pairs"].append(qa_pair)
            except AssertionError:
                print(f"[-] failed to get a drop-format answer for {record['qid']} "
                      f"with answer \"{record['transformed_answer']}\".")
        else:
            no_ans_count += 1

    print(f"examples without an answer: {no_ans_count}")

    # If we dropped some records (either because there was no answer generated, or because of bad format),
    # re-calculate the number of example per-transformation.
    if len(final_records) < len(records):
        df = pd.DataFrame.from_records(final_records)
        df['transformation_base'] = df.transformation.apply(lambda x: get_transform_base_info(x)[0])
        print(f"breakdown by transformation, "
              f"after removing examples without a generated answer or with bad-format answers:")
        print(df['transformation_base'].value_counts())

    return drop_data


def get_hotpotqa_data(records):
    hotpot_data_dict = {}
    no_ans_count = 0
    for record in records:
        title, passage_idx = record["passage_id"].rsplit("_", 1)
        if title not in hotpot_data_dict:
            hotpot_data_dict[title] = {
                "title": title,
                "paragraphs": {}
            }
        if passage_idx not in hotpot_data_dict[title]["paragraphs"]:
            hotpot_data_dict[title]["paragraphs"][passage_idx] = {
                "context": record["passage"],
                "qas": []
            }
        qa_pair = {
            "question": record["generated_question"],
            "id": record["qid"],
        }
        if "transformed_answer" in record:
            qa_pair["answers"] = [answer_to_hotpot_squad_format(record)]
            qa_pair["is_impossible"] = False
            qa_pair["is_distant"] = False
        else:
            no_ans_count += 1
        hotpot_data_dict[title]["paragraphs"][passage_idx]["qas"].append(qa_pair)

    # flat paragraph and title dictionaries
    for title in hotpot_data_dict:
        hotpot_data_dict[title]["paragraphs"] = [
            paragraph_data
            for paragraph_idx, paragraph_data in hotpot_data_dict[title]["paragraphs"].items()
        ]
    hotpot_data = [
        title_data
        for title, title_data in hotpot_data_dict.items()
    ]

    print(f"examples without an answer: {no_ans_count}")
    return {"version": "v2.0", "data": hotpot_data}


def sample_from_records(records, sampling, hard_sampling, dataset):
    assert not (hard_sampling and sampling > -1)

    df = pd.DataFrame.from_records(records)
    df['transformation_base'] = df.transformation.apply(lambda x: get_transform_base_info(x)[0])

    if hard_sampling:
        sampling_conf = hard_sampling_conf[dataset]
        df = df.groupby('transformation_base').apply(
            lambda g: g.sample(n=sampling_conf[g.iloc[0]['transformation_base']])
        )

    elif sampling > -1:
        max_examples = round(0.01 * sampling * dataset_to_num_train_examples[dataset])
        df = df.groupby('transformation_base').apply(
            lambda g: g.sample(n=min(max_examples, len(g)))
        )

    print(f"breakdown by transformation, after {sampling}% sampling:")
    print(df['transformation_base'].value_counts())

    records = df.to_dict('records')
    return records


def main():
    args = get_args()

    with open(args.example_info_path, "r") as fd:
        records = json.load(fd)
    print(f"loaded {len(records)} records from {args.example_info_path}.")

    if args.validated_qids_path != "":
        with open(args.validated_qids_path, "r") as fd:
            valid_qids = [qid.strip('\n') for qid in fd]
        records = [record for record in records if record['qid'] in valid_qids]
        assert len(records) == len(valid_qids), \
            "all validated qids were supposed to be extracted from the given example info file " \
            "(corresponding to the contrast set)."
        print(f"left with {len(records)} records after keeping only ({len(valid_qids)}) validated question ids.")

    records = sample_from_records(records, args.sampling, args.hard_sampling, args.data_format)
    print(f"after {args.sampling}% sampling, left with {len(records)} records.")

    data = None
    if args.data_format == "drop":
        data = get_drop_data(records, allow_yes_no=args.allow_yes_no, append_yes_no=args.append_yes_no)
    elif args.data_format == "iirc":
        data = get_drop_data(records, allow_yes_no=True, append_yes_no=args.append_yes_no)
    elif args.data_format == "hotpot-squad":
        data = get_hotpotqa_data(records)

    with open(args.output_path, "w") as fd:
        json.dump(data, fd, indent=4)
    print(f"wrote {args.data_format}-format predictions to: {args.output_path}")


if __name__ == "__main__":
    main()
