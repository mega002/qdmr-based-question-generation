
import argparse

from qdmr_transforms.write_transforms import apply_qdmr_transformations


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--input_file", type=str,
                       help="path to a csv file with QDMRs", required=True)
    parse.add_argument("--output_file", type=str,
                       help="path to a csv output file of perturbed QDMRs", required=True)
    parse.add_argument("--numeric_qa_file", type=str, default="data/numeric_qa_examples.json",
                       help="path to a file mapping question IDs to their numeric answers")
    parse.add_argument("--dataset_name", choices=["drop", "hotpotqa", "iirc"], required=True)
    parse.add_argument("--limit_append_boolean_step_per_qdmr", type=int, default=-1,
                       help="the maximum number of append-boolean-step perturbations per example (-1 means no limit).")

    return parse.parse_args()


def main():
    args = get_args()

    filters = ["data_operators", "time_diff_sum", "self_diff", "single_noun_phrase"]
    apply_qdmr_transformations(args.input_file, args.output_file, args.dataset_name,
                               filters=filters, transformations=None, limit=None,
                               limit_append_boolean_step_per_qdmr=args.limit_append_boolean_step_per_qdmr,
                               numeric_qa_file=args.numeric_qa_file)


if __name__ == '__main__':
    main()
