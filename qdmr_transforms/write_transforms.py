import pandas as pd
from typing import Dict
from csv import DictWriter
from tqdm import tqdm

from qdmr_transforms.qdmr_transformations import *
from qdmr_transforms.transformation_filters import *
from qdmr_transforms.utils import load_json
from qdmr_transforms.qdmr_example import \
    get_transform_from_example_id, get_transform_base_info, get_transform_base_info_parts


def read_qdmr_data(csv_file):
    data = pd.read_csv(csv_file)
    return data


def read_qdmr_example(example: Dict[str, str]):
    return QDMRExample(example['question_id'], example['question_text'], example['decomposition'])


def apply_qdmr_transformations(input_file, output_file, dataset_name,
                               transformations=None, filters=None, limit=None,
                               limit_append_boolean_step_per_qdmr=-1,
                               numeric_qa_file=None):
    qdmr_data = read_qdmr_data(input_file)
    qdmr_data = qdmr_data[:limit] if (limit is not None) else qdmr_data
    numeric_qa_data = load_json(numeric_qa_file) if numeric_qa_file is not None else None
    if filters:
        assert set(filters).issubset(set(TRANSFORM_FILTERS))
        transform_filter = TransformFilter(filters, \
                                           qdmr_data=input_file, \
                                           operator_dist_threshold=0.15)
    with open(output_file, mode='w', encoding='utf-8') as csv_file:
        field_names = ['id','question','decomposition','transformation', 'type', 'transformed_question']
        writer = DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
    written = 0
    invalid = 0
    for i in tqdm(range(len(qdmr_data)), desc="Loading…", ascii=False, ncols=75):
#         qdmr_example = read_qdmr_example(qdmr_data.iloc[i])
#         written += write_qdmr_transformations(qdmr_example,\
#                                               output_file,\
#                                               field_names,\
#                                               transformations=transformations)
        try:
            qdmr_example = read_qdmr_example(qdmr_data.iloc[i])
            if qdmr_example.is_valid_qdmr() is False:
                invalid += 1
                print("* Invalid example: ", qdmr_example.qdmr)
                continue
            written += write_qdmr_transformations(qdmr_example,\
                                                  output_file,\
                                                  field_names,\
                                                  transformations=transformations,\
                                                  transform_filter=transform_filter, \
                                                  limit_append_boolean_step_per_qdmr=limit_append_boolean_step_per_qdmr,
                                                  numeric_qa_data=numeric_qa_data)
        except:
            print("* Error with example: ", qdmr_example.qdmr)
            continue

    # remove duplicates from output CSV
    df = pd.read_csv(output_file)
    total_with_duplicates = len(df)
    df.drop_duplicates(subset=None, inplace=True)

    # write the generated decomposition-question pairs that were created for
    # augmenting the data used to train the question generation model.
    df_augmented_cols = [
        "id", "transformed_question", "transformation"
    ]  # should match the break data format (csv)
    df_augmented = df[df_augmented_cols][~df.transformed_question.isna()]
    df_augmented.rename(columns={
        "id": "question_id",
        "transformed_question": "question_text",
        "transformation": "decomposition"
    }, inplace=True)
    df_augmented["operators"] = ""
    df_augmented["split"] = df["id"].apply(lambda x: x.split('_', 2)[1])
    df_augmented.to_csv(output_file.replace(".csv", "_augmented_qs.csv"), index=False)
    total_augmented = len(df_augmented)

    # now filter the generated transformations to keep only transformations
    # that result in high-quality examples (based on a manual quality-analysis)
    total_no_duplicates = len(df)
    df["to_drop"] = df.id.apply(lambda x: is_bad_transformation(x, dataset_name))
    df_filtered = df[~df.to_drop]

    df_transform_cols = [
        "id", "question", "decomposition", "transformation", "type"
    ]
    df_filtered[df_transform_cols].to_csv(output_file, index=False)
    total_after_filter = len(df_filtered)

    print(f"Ignored {invalid} invalid QDMR transformations.")
    print(f"Overall, generated {written} QDMR transformations.")
    print(f"After duplicate removal ({total_with_duplicates-total_no_duplicates}) and "
          f"dropping bad transformations ({total_no_duplicates-total_after_filter}), "
          f"wrote {total_after_filter} QDMR transformations to file.")
    print(f"Wrote {total_augmented} generated questions for augmentation to file.")
    print("Complete.")

    return True


def is_bad_transformation(example_id, dataset_name):
    transformation = get_transform_from_example_id(example_id)
    transform_base, transform_info = get_transform_base_info(transformation)
    transform_info_parts = get_transform_base_info_parts(transform_info)

    if dataset_name in ["drop", "iirc", "hotpotqa", "break"]:
        if transform_base == "op_replace_aggregate":
            return True
        if transform_base == "prune_last_step":
            pruned_step = transform_info_parts[-1]
            if pruned_step != "project":
                return True

    if dataset_name in ["drop", "iirc"] and transform_base == "prune_step":
        return True

    if transform_base == "change_last_step":
        _, _, new_op, new_op_subtype = transform_info_parts
        if dataset_name == "drop" and new_op != "arithmetic":
            return True
        elif dataset_name in ["iirc", "hotpotqa"] and \
                (new_op != "boolean" or new_op_subtype != "the_same_as"):
            return True

    return False


def write_qdmr_transformations(qdmr_example, output_file, field_names,
                               transformations, transform_filter=None,
                               limit_append_boolean_step_per_qdmr=-1,
                               numeric_qa_data=None):
    if transformations is not None:
        # TODO: handle subset of transformation
        x = 1
    transformations = []
    # step operator replacement
    op_rep_transform = OpReplaceTransform(qdmr_example)
    transformations += op_rep_transform.transformations()
    # prune last step
    prune_last_transform = PruneLastTransform(qdmr_example)
    transformations += prune_last_transform.transformations()
    # prune middle step
    prune_step_transform = PruneStepTransform(qdmr_example)
    transformations += prune_step_transform.transformations()
    # change last step
    change_last_step_transform = ChangeLastStepTransform(qdmr_example)
    transformations += change_last_step_transform.transformations()
    # append boolean condition step
    append_last_step_transform = AppendBooleanTransform(qdmr_example,
                                                        limit=limit_append_boolean_step_per_qdmr,
                                                        numeric_qa_data=numeric_qa_data)
    transformations += append_last_step_transform.transformations()
    if transform_filter:
        # filter out problematic transformations
        transformations = list(filter(lambda example: (not transform_filter.filter_out(example)), transformations))
    for trans_example in transformations:
        row_dict = {}
        row_dict['id'] = trans_example.example_id
        row_dict['question'] = qdmr_example.question
        row_dict['decomposition'] = qdmr_example.qdmr
        row_dict['transformation'] = trans_example.qdmr
        row_dict['type'] = trans_example.transform
        row_dict['transformed_question'] = trans_example.transformed_question
        # write results
        append_dict_as_row(output_file, row_dict, field_names)
    return len(transformations)


def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='', encoding='utf-8') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)
