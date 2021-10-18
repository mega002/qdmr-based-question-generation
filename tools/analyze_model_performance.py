
import argparse
import json
import pandas as pd

from qdmr_transforms.qdmr_example import \
    get_orig_example_id, get_transform_base_info, get_transform_from_example_id
from src.data.dataset_readers.drop import DropReader

reader = DropReader()
HLINE = "\n---------------------------------------------"


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--gen_preds_path", type=str,
                       help="path of predictions file on generated examples (jsonl)")
    parse.add_argument("--orig_preds_path", type=str,
                       help="path of predictions file on subset of original examples (jsonl)")

    parse.add_argument("--gen_const_preds_path", type=str,
                       help="path to predictions file on generated constraints examples (json/jsonl)")
    parse.add_argument("--orig_const_preds_path", type=str,
                       help="path of predictions file on subset of original constraints examples (jsonl)")
    parse.add_argument("--example_info_file", type=str,
                       help="path to the example info file of the constraint set (json)")

    parse.add_argument("--only_merged", action="store_true",
                       help="Calculate and print evaluation scores only for the merged sets of answers+constraints.")
    parse.add_argument("--merge_prune_trans", action="store_true",
                       help="When aggregating scores per transformation, "
                            "consider all prune step variants as a single transformation.")

    return parse.parse_args()


def extract_numeric_value(value):
    if type(value) not in [str, int, float]:
        return None

    return reader.extract_number_from_text(value)


def check_prediction_constraint(prediction, constraint, orig_answer):

    prediction_numeric_value = extract_numeric_value(prediction)
    is_prediction_numeric = prediction_numeric_value is not None
    orig_answer_numeric_value = extract_numeric_value(orig_answer)
    is_orig_answer_numeric = orig_answer_numeric_value is not None

    if constraint == "boolean":
        if type(prediction) != str:
            return False

        return prediction.lower() in ["yes", "no"]

    if constraint == "arithmetic":
        return is_prediction_numeric

    if constraint == ">=" and is_orig_answer_numeric:
        return is_prediction_numeric and prediction_numeric_value >= orig_answer_numeric_value

    if constraint == "<=" and is_orig_answer_numeric:
        return is_prediction_numeric and prediction_numeric_value <= orig_answer_numeric_value

    return -1


def are_gen_orig_correct(row, f1_threshold=-1):
    if f1_threshold > 0:
        return int(row["f1_orig"] > f1_threshold and row["f1_gen"] > f1_threshold)
    else:
        return int(row["em_orig"] == 1 and row["em_gen"] == 1)


def are_gen_orig_satisfied(row, f1_threshold=-1):
    if f1_threshold > 0:
        return int(row["f1_orig"] > f1_threshold and row["score_gen"] == 1)
    else:
        return int(row["em_orig"] == 1 and row["score_gen"] == 1)


def are_gen_orig_correct_satisfied(row, f1_threshold=-1):
    if f1_threshold > 0:
        # this is an answer example
        if pd.isna(row["score_gen"]):
            return int(row["f1_orig"] > f1_threshold and row["f1_gen"] > f1_threshold)
        # this is a constraint example
        else:
            return int(row["f1"] > f1_threshold and row["score_gen"] == 1)
    else:
        # this is an answer example
        if pd.isna(row["score_gen"]):
            return int(row["em_orig"] == 1 and row["em_gen"] == 1)
        # this is a constraint example
        else:
            return int(row["em"] == 1 and row["score_gen"] == 1)


def load_prediction_files(gen_preds_path, orig_preds_path, gen_const_preds_path, orig_const_preds_path):
    gen_preds = None
    orig_preds = None
    if gen_preds_path is not None:
        assert orig_preds_path is not None
        with open(gen_preds_path, "r") as fd:
            gen_preds = [json.loads(line) for line in fd]
        with open(orig_preds_path, "r") as fd:
            orig_preds = [json.loads(line) for line in fd]
        print(f"read predictions for {len(gen_preds)} generated examples and {len(orig_preds)} original examples.")

    gen_const_preds_raw = None
    orig_const_preds = None
    if gen_const_preds_path is not None:
        assert orig_const_preds_path is not None
        with open(gen_const_preds_path, "r") as fd:
            if gen_const_preds_path.endswith("jsonl"):
                gen_const_preds_records = [json.loads(line) for line in fd]
                gen_const_preds_raw = {
                    pred["query_id"]: pred["answer"]["value"]
                    for pred in gen_const_preds_records
                }
            elif gen_const_preds_path.endswith("json"):
                gen_const_preds_raw = json.load(fd)
            else:
                raise NotImplementedError

        with open(orig_const_preds_path, "r") as fd:
            orig_const_preds = [json.loads(line) for line in fd]

        print(f"read predictions for {len(gen_const_preds_raw)} generated examples and "
              f"{len(orig_const_preds)} original examples.")

    return gen_preds, orig_preds, gen_const_preds_raw, orig_const_preds


def constraint_evaluation(constraints, gen_const_preds_raw):
    gen_const_preds = []
    for qid in gen_const_preds_raw:
        constraint = constraints[qid]["transformed_answer_constraints"]
        orig_answer = ' '.join(constraints[qid]["orig_answer_texts"])
        score = check_prediction_constraint(gen_const_preds_raw[qid], constraint, orig_answer)
        if score == -1:
            print(f"[-] issue with evaluation of example: {qid}")
        else:
            transform_base, transform_info = get_transform_base_info(get_transform_from_example_id(qid))
            gen_const_preds.append({
                "qid": qid,
                "base_id": get_orig_example_id(qid, clean=True),
                "transform_base": transform_base,
                "transform_info": transform_info,
                "prediction": gen_const_preds_raw[qid],
                "constraint": constraint,
                "score_gen": float(score)
            })

    return gen_const_preds


def build_gen_orig_ans_preds_df(gen_ans_preds, orig_ans_preds):
    if "qid" in gen_ans_preds[0]:
        qid_field = "qid"
    elif "query_id" in gen_ans_preds[0]:
        qid_field = "query_id"
    else:
        raise NotImplementedError

    for gen_pred in gen_ans_preds:
        gen_pred["base_id"] = get_orig_example_id(gen_pred[qid_field], clean=True)
        gen_pred["transform_base"], gen_pred["transform_info"] = get_transform_base_info(
            get_transform_from_example_id(gen_pred[qid_field])
        )
        if qid_field == "query_id":
            gen_pred["qid"] = gen_pred["query_id"]
    for orig_pred in orig_ans_preds:
        orig_pred["base_id"] = orig_pred[qid_field]

    df_gen = pd.DataFrame.from_records(gen_ans_preds)
    df_orig = pd.DataFrame.from_records(orig_ans_preds)
    df = pd.merge(df_gen, df_orig, on="base_id", suffixes=("_gen", "_orig"))
    df["gen_orig_correct_em"] = df.apply(
        lambda row: are_gen_orig_correct(row),
        axis=1
    )
    df["gen_orig_correct_f1_th0.8"] = df.apply(
        lambda row: are_gen_orig_correct(row, f1_threshold=0.8),
        axis=1
    )

    return df_gen, df_orig, df


def build_gen_orig_const_preds_df(gen_const_preds, orig_const_preds):
    if "qid" in orig_const_preds[0]:
        qid_field = "qid"
    elif "query_id" in orig_const_preds[0]:
        qid_field = "query_id"
    else:
        raise NotImplementedError

    for orig_pred in orig_const_preds:
        orig_pred["base_id"] = orig_pred[qid_field]
        orig_pred["f1_orig"] = orig_pred["f1"]
        orig_pred["em_orig"] = orig_pred["em"]

    df_gen_const = pd.DataFrame.from_records(gen_const_preds)
    df_orig_const = pd.DataFrame.from_records(orig_const_preds)
    df_const = pd.merge(df_gen_const, df_orig_const, on="base_id", suffixes=("_gen", "_orig"))
    df_const["gen_orig_correct_em"] = df_const.apply(
        lambda row: are_gen_orig_satisfied(row),
        axis=1
    )
    df_const["gen_orig_correct_f1_th0.8"] = df_const.apply(
        lambda row: are_gen_orig_satisfied(row, f1_threshold=0.8),
        axis=1
    )

    return df_gen_const, df_orig_const, df_const


def merge_ans_const_dfs(df_ans, df_const):
    # remove constraints that do not exist in df_ans (for which there is no answer)
    qids_with_ans = df_ans.base_id.unique()
    df_const_ = df_const[df_const.base_id.isin(qids_with_ans)]
    print(f"constraints cover {len(df_const.base_id.unique())} original examples, and for "
          f"{len(df_const_.base_id.unique())} of them there is also an answer.")

    # concatenate dfs
    df_merged = pd.concat([df_ans, df_const_])
    print(f"merged {len(df_ans)} answer examples with {len(df_const_)} constraint examples to overall "
          f"{len(df_merged)} examples. ( len(df_ans) + len(df_const_) = {len(df_ans)+len(df_const_)} )")

    df_merged["gen_orig_correct_em"] = df_merged.apply(
        lambda row: are_gen_orig_correct_satisfied(row),
        axis=1
    )
    df_merged["gen_orig_correct_f1_th0.8"] = df_merged.apply(
        lambda row: are_gen_orig_correct_satisfied(row, f1_threshold=0.8),
        axis=1
    )

    return df_merged


def print_evaluation_scores(df_gen, df_orig, df):
    print("*** ANSWER EVALUATION SCORES ***")
    for df_, df_name_ in zip([df_gen, df_orig], ["gen", "orig"]):
        print(f"\noverall example-level performance: {df_name_}" + HLINE)
        print(df_["f1"].
              mean().
              round(3))

    print("\nperformance per transformation type: gen" + HLINE)
    print(df_gen[["f1", "transform_base"]].
          groupby("transform_base").
          agg(["mean", "count"]).
          round(3))
    print("\nperformance per transformation type: orig" + HLINE)
    print(df[["base_id", "f1_orig", "transform_base"]].
          drop_duplicates().
          groupby("transform_base").
          agg(["mean", "count"]).
          round(3))

    print("\ncontrast consistency" + HLINE)
    sets_cols = ["base_id", "gen_orig_correct_em", "gen_orig_correct_f1_th0.8"]
    print(df[sets_cols].
          groupby("base_id").
          agg("min").
          mean().
          round(3))


def print_const_evaluation_scores(df_gen_const, df_orig_const, df_const):
    print("*** CONSTRAINT EVALUATION SCORES ***")
    for df_, df_name_ in zip([df_gen_const, df_orig_const], ["gen", "orig"]):
        print(f"\noverall example-level performance: {df_name_}" + HLINE)
        if df_name_ == "orig":
            score_field = "f1"
        else:
            score_field = "score_gen"
        print(df_[score_field].
              mean().
              round(3))

    print("\nperformance per constraint type: gen" + HLINE)
    print(df_gen_const[["score_gen", "constraint"]].
          groupby("constraint").
          agg(["mean", "count"]).
          round(3))
    print("\nperformance per constraint type: orig" + HLINE)
    print(df_const[["base_id", "f1_orig", "constraint"]].
          drop_duplicates().
          groupby("constraint").
          agg(["mean", "count"]).
          round(3))
    print("\nconstraint consistency" + HLINE)
    sets_cols = ["base_id", "gen_orig_correct_em", "gen_orig_correct_f1_th0.8"]
    print(df_const[sets_cols].
          groupby("base_id").
          agg("min").
          mean().
          round(3))


def print_merged_evaluation_scores(df_merged):
    print("*** ANSWER + CONSTRAINT EVALUATION SCORES ***")
    print("\nanswer+constraint consistency" + HLINE)
    sets_cols = ["base_id", "gen_orig_correct_em", "gen_orig_correct_f1_th0.8"]
    print(df_merged[sets_cols].
          groupby("base_id").
          agg("min").
          mean().
          round(3))


def main():
    args = get_args()

    #
    # load prediction and constraints files
    #
    gen_ans_preds, orig_ans_preds, gen_const_preds_raw, orig_const_preds = \
        load_prediction_files(args.gen_preds_path, args.orig_preds_path,
                              args.gen_const_preds_path, args.orig_const_preds_path)

    gen_const_preds = None
    if args.gen_const_preds_path is not None:
        assert args.example_info_file is not None
        with open(args.example_info_file, "r") as fd:
            info = json.load(fd)
            constraints = {
                record["qid"]: record
                for record in info
            }
        print(f"read constraints for {len(constraints)} examples.")

        # constraint evaluation of model predictions
        gen_const_preds = constraint_evaluation(constraints, gen_const_preds_raw)

    #
    # build dataframes
    #
    df_gen_ans = None
    df_orig_ans = None
    df_ans = None
    if gen_ans_preds is not None:
        assert orig_ans_preds is not None
        df_gen_ans, df_orig_ans, df_ans = build_gen_orig_ans_preds_df(gen_ans_preds, orig_ans_preds)

    df_gen_const = None
    df_orig_const = None
    df_const = None
    if gen_const_preds is not None:
        assert orig_const_preds is not None
        df_gen_const, df_orig_const, df_const = build_gen_orig_const_preds_df(gen_const_preds, orig_const_preds)

    df_merged = None
    if df_ans is not None and df_const is not None:
        df_merged = merge_ans_const_dfs(df_ans, df_const)


    #
    # print evaluation scores
    #
    if args.merge_prune_trans:
        for df_ in [df_gen_ans, df_orig_ans, df_ans,
                    df_gen_const, df_orig_const, df_const,
                    df_merged]:
            if df_ is not None and "transform_base" in df_.columns:
                df_["transform_base"] = df_["transform_base"].apply(
                    lambda x: "prune_step" if x.startswith("prune") else x
                )

    if df_gen_ans is not None and not args.only_merged:
        print_evaluation_scores(df_gen_ans, df_orig_ans, df_ans)
    if df_gen_const is not None and not args.only_merged:
        print_const_evaluation_scores(df_gen_const, df_orig_const, df_const)
    if df_merged is not None:
        print_merged_evaluation_scores(df_merged)


if __name__ == "__main__":
    main()
