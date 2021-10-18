
import csv
from qdmr_transforms.qdmr_example import get_orig_example_id
from src.data.dataset_readers.break_reader import process_target


def read_qdmrs(qdmrs_path, dataset_name):
    qdmrs = []
    with open(qdmrs_path) as fd:
        lines = csv.reader(fd)
        header = next(lines, None)
        num_fields = len(header)
        assert num_fields == 5

        for i, line in enumerate(lines):
            assert len(line) == num_fields, "read {} fields, and not {}".format(
                len(line), num_fields
            )
            question_id, source, decomposition, transformed, transformation = line
            orig_id = get_orig_example_id(question_id)
            dataset, _, qid = orig_id.split("_", 2)
            if dataset.lower() in dataset_name:
                if dataset_name == "iirc":
                    qid = qid.split("_", 2)[-1]
                decomposition = process_target(decomposition, fix_refs=False)
                transformed = process_target(transformed, fix_refs=False)
                item = {
                    "orig_qid": qid,
                    "qid": question_id,
                    "question": source,
                    "decomposition": decomposition,
                    "transformed": transformed,
                    "transformation": transformation
                }
                qdmrs.append(item)

    print(f"read {len(qdmrs)} transformed QDMRs for {dataset_name}.")

    return qdmrs
