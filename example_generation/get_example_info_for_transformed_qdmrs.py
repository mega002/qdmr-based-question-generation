
import argparse
import json

from collections import defaultdict
from random import random

from example_generation.answer_generators.generate_answer import get_transformed_qdmr_answer
from example_generation.answer_generators.generate_answer_constraints import get_transformed_qdmr_answer_constraints
from example_generation.utils import extract_drop_subset, extract_hotpotqa_subset
from example_generation.get_examples_from_example_info import answer_to_drop_format, answer_to_hotpot_squad_format
from src.data.dataset_readers.drop import DropReader
from src.data.dataset_readers.hotpotqa import HotpotQASQuADReader
from src.data.dataset_readers.transformed_qdmrs import read_qdmrs
from qdmr_transforms.qdmr_example import get_transform_base_info


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--qdmrs_path", type=str, help="transformed QDMRs path", required=True)
    parse.add_argument("--orig_data_path", type=str, help="data path", required=True)
    parse.add_argument("--gen_qs_path", type=str, help="generated questions path (jsonl)", default="")
    parse.add_argument("--pred_step_ans_path", type=str, default="",
                       help="path to predicted decomposition step answers (json)")
    parse.add_argument("--dataset_name", choices=['drop', 'hotpot-squad', 'iirc'], help="dataset name", required=True)
    parse.add_argument("--output_path", type=str, default="",
                       help="if not specified, output will be written to the predictions path directory")
    parse.add_argument("--allow_yes_no", action="store_true",
                       help="Include examples with yes/no answer when generating DROP examples "
                            "(in HotpotQA and IIRC this is allowed by default).")
    parse.add_argument("--debug_prints_ratio", type=float, default=0, help="print this ratio of examples for debugging")
    parse.add_argument("--debug_get_only_failue_cases", action="store_true",
                       help="output only examples for which we failed generating an answer.")

    return parse.parse_args()


def print_stats(stats, ans_stats, bad_form_ans_stats, no_ans_stats, ans_consts_stats, ans_gen_func_stats, qdmrs):
    num_effective_qdmrs = len(qdmrs) - stats["skipped"]
    print(f"\nhandled {num_effective_qdmrs} QDMRs for which we have a generated question + single-step answers, "
          f"out of {len(qdmrs)} transformed qdmrs.")
    num_ans_info = sum([stats["ans_and_ans_consts"], stats["just_ans"], stats["just_ans_consts"]])
    print(f"\ngenerated answer/constraints for {num_ans_info} "
          f"({round(num_ans_info * 100.0 / num_effective_qdmrs, 1)}) examples "
          f"out of {num_effective_qdmrs} transformed qdmrs:")
    print(stats)

    num_trans_ans = sum(ans_stats.values())
    print(f"\ngenerated answers to {num_trans_ans} ({round(num_trans_ans * 100.0 / num_effective_qdmrs, 1)}) "
          f"out of {num_effective_qdmrs} transformed qdmrs.")
    print(ans_stats)

    num_trans_bad_form_ans = sum(bad_form_ans_stats.values())
    print(f"\ndiscarded examples with bad-form answers to {num_trans_bad_form_ans} "
          f"({round(num_trans_bad_form_ans * 100.0 / num_effective_qdmrs, 1)}) "
          f"out of {num_effective_qdmrs} transformed qdmrs.")
    print(bad_form_ans_stats)

    num_trans_no_ans = sum(no_ans_stats.values())
    print(f"\ncouldn't generate answers to {num_trans_no_ans} "
          f"({round(num_trans_no_ans * 100.0 / num_effective_qdmrs, 1)}) "
          f"out of {num_effective_qdmrs} transformed qdmrs.")
    print(no_ans_stats)

    num_ans_gen_funcs = sum(ans_gen_func_stats.values())
    print(f"\nanswer generation functions stats (overall {num_ans_gen_funcs} records):")
    print(ans_gen_func_stats)

    num_trans_ans_conts = sum(ans_consts_stats.values())
    print(f"\ngenerated answer constraints to {num_trans_ans_conts} "
          f"({round(num_trans_ans_conts * 100.0 / num_effective_qdmrs, 1)}) "
          f"out of {num_effective_qdmrs} transformed qdmrs.")
    print(ans_consts_stats)

    print('\n')


def print_qdmr_info(qdmr):
    print(json.dumps(qdmr, indent=4), '\n')


def add_example_info(qdmrs, qa_pairs, generated_questions, predicted_step_ans, dataset_name,
                     allow_yes_no, debug_prints):
    ans_stats = defaultdict(int)
    no_ans_stats = defaultdict(int)
    bad_form_ans_stats = defaultdict(int)
    ans_consts_stats = defaultdict(int)
    ans_gen_func_stats = defaultdict(int)
    stats = defaultdict(int)

    for qdmr in qdmrs:
        if qdmr["qid"] not in generated_questions or qdmr["qid"] not in predicted_step_ans:
            stats["skipped"] += 1
            continue
        instance = qa_pairs[qdmr["orig_qid"]]
        qdmr["passage_id"] = instance['metadata']['passage_id']
        qdmr["passage"] = instance['metadata']["original_passage"]
        qdmr["orig_answer_texts"] = instance['metadata']["answer_texts"]
        qdmr["orig_answer_annotations"] = instance['metadata']["answer_annotations"]
        qdmr["orig_numbers"] = instance['metadata']["original_numbers"][:-1]
        qdmr["generated_question"] = generated_questions[qdmr["qid"]]["questions"][0][0]
        qdmr["step_answers"] = predicted_step_ans[qdmr["qid"]]["step_answers"]

        # get answer constraints
        get_transformed_qdmr_answer_constraints(qdmr)
        if "transformed_answer_constraints" in qdmr:
            transform_base, _ = get_transform_base_info(qdmr["transformation"])
            ans_consts_stats[transform_base] += 1

        # get answer
        get_transformed_qdmr_answer(qdmr)
        transform_base, _ = get_transform_base_info(qdmr["transformation"])
        if "transformed_answer" in qdmr:
            try:
                if dataset_name == "drop":
                    _ = answer_to_drop_format(qdmr, allow_yes_no=allow_yes_no)
                elif dataset_name == "iirc":
                    _ = answer_to_drop_format(qdmr, allow_yes_no=True)
                elif dataset_name == "hotpot-squad":
                    answer_dict = answer_to_hotpot_squad_format(qdmr)
                    assert qdmr["transformed_answer"] in ["yes", "no"] or answer_dict["answer_start"] > -1

                ans_stats[transform_base] += 1
                ans_gen_func = qdmr["transformed_answer_gen_func"]
                ans_gen_func_stats[ans_gen_func] += 1

            except:
                qdmr.pop("transformed_answer")
                qdmr.pop("transformed_answer_gen_func")
                bad_form_ans_stats[transform_base] += 1

        else:
            no_ans_stats[transform_base] += 1

        if "transformed_answer_constraints" in qdmr and "transformed_answer" in qdmr:
            stats["ans_and_ans_consts"] += 1
        elif "transformed_answer_constraints" in qdmr:
            stats["just_ans_consts"] += 1
        elif "transformed_answer" in qdmr:
            stats["just_ans"] += 1
        else:
            stats["no_ans_and_and_consts"] += 1

        if random() < debug_prints:
            print_qdmr_info(qdmr)

    print_stats(stats, ans_stats, bad_form_ans_stats, no_ans_stats, ans_consts_stats, ans_gen_func_stats, qdmrs)


def main():
    args = get_args()

    # load qdmrs
    qdmrs = read_qdmrs(args.qdmrs_path, args.dataset_name)

    # load generated questions
    generated_questions = None
    if args.gen_qs_path != "":
        generated_questions = [json.loads(line) for line in open(args.gen_qs_path, "r")]
        generated_questions = {
            gen_q["qid"]: gen_q
            for gen_q in generated_questions
        }

    # load predicted decomposition step answers
    predicted_step_ans = None
    if args.pred_step_ans_path != "":
        with open(args.pred_step_ans_path, "r") as fd:
            predicted_step_ans = {
                record["qid"]: record
                for record in json.load(fd)
            }

    # add original data info + generate answers to transformed qdmrs
    dataset_reader = None
    just_qids = True
    if args.dataset_name in ["drop", "iirc"]:
        dataset_reader = DropReader()
        if args.dataset_name == "drop":
            just_qids = False
    elif args.dataset_name == "hotpot-squad":
        dataset_reader = HotpotQASQuADReader()
    qa_pairs = dataset_reader.get_qa_pairs_dict(args.orig_data_path, just_qids=just_qids)

    add_example_info(qdmrs, qa_pairs, generated_questions, predicted_step_ans, args.dataset_name,
                     args.allow_yes_no, args.debug_prints_ratio)
    num_qdmrs = len(qdmrs)

    # remove qdmrs where the generated question is different from the original question
    qdmrs = [
        qdmr for qdmr in qdmrs
        if "generated_question" in qdmr and qdmr["generated_question"] != qdmr["question"]
    ]

    # get answer constraints
    qdmrs_ans_const = [
        qdmr for qdmr in qdmrs
        if "transformed_answer_constraints" in qdmr
    ]

    if args.debug_get_only_failue_cases:
        # keep only qdmrs for which there is NO answer generated.
        qdmrs = [
            qdmr for qdmr in qdmrs
            if "transformed_answer" not in qdmr and
               "generated_question" in qdmr and
               "step_answers" in qdmr
        ]
        print(f"left with {len(qdmrs)} ({round(len(qdmrs) * 100.0 / num_qdmrs, 1)}%) out of {num_qdmrs} qdmrs, "
              f"after removing examples with (a) generated answer or (b) generated question = original question.")
        output_path = args.qdmrs_path.replace(".csv", f"_{args.dataset_name}_example_info_failure_cases.json")
    else:
        # keep only qdmrs for which there is an answer.
        qdmrs = [
            qdmr for qdmr in qdmrs
            if "transformed_answer" in qdmr
        ]
        print(f"left with {len(qdmrs)} ({round(len(qdmrs) * 100.0 / num_qdmrs, 1)}%) out of {num_qdmrs} qdmrs, "
              f"after removing cases of (a) no answer or (b) generated question = original question.")
        output_path = args.qdmrs_path.replace(".csv", f"_{args.dataset_name}_example_info.json")

    # save generated examples
    if args.output_path != "":
        output_path = args.output_path
    with open(output_path, "w") as fd:
        json.dump(qdmrs, fd, indent=4)
    print(f"\nwrote transformed QDMRs with answers in example info format to: {output_path}")

    # save generated answer constraints
    ans_const_json_output_path = output_path.replace(".json", "_ans_const.json")
    with open(ans_const_json_output_path, "w") as fd:
        json.dump(qdmrs_ans_const, fd, indent=4)
    print(f"\nwrote answer constraints for {len(qdmrs_ans_const)} transformed QDMRs to: {ans_const_json_output_path}")

    # get corresponding original evaluation set
    orig_example_break_ids = [qdmr["orig_qid"] for qdmr in qdmrs]
    eval_data_json = None
    if args.dataset_name == "drop":
        eval_data_json = extract_drop_subset(orig_example_break_ids, args.orig_data_path, just_qids=False)
    elif args.dataset_name == "iirc":
        eval_data_json = extract_drop_subset(orig_example_break_ids, args.orig_data_path, just_qids=True)
    elif args.dataset_name == "hotpot-squad":
        eval_data_json = extract_hotpotqa_subset(orig_example_break_ids, args.orig_data_path)

    eval_data_json_output_path = output_path.replace(".json", "_orig_subset.json")
    if eval_data_json is not None:
        with open(eval_data_json_output_path, "w") as fd:
            json.dump(eval_data_json, fd, indent=4)
        print(f"\nwrote the original data subset corresponding to the generated examples: {eval_data_json_output_path}")


if __name__ == "__main__":
    main()
