import argparse
import json
import os
import sys

import namegenerator

from run import main as run_main


def run(args):
    overrides_dict = {}
    overrides_dict.update({
        "dataset_reader": {"save_elasticsearch_cache": False},
        "validation_dataset_reader": {"save_elasticsearch_cache": False}
    })
    if args.debug:
        overrides_dict.update(
            {"validation_dataset_reader": {"max_instances": 50, "pickle": None},}
        )

    overrides_dict.update(json.loads(args.overrides))
    overrides = json.dumps(overrides_dict)

    output_file = args.output_file
    if output_file is None:
        base_dirname = os.path.dirname(args.model)
        data_name = os.path.splitext(os.path.basename(args.model))[0]
        output_name = args.output_name
        if output_name is None:
            output_name = f"eval_{data_name}.json"
        output_file = os.path.join(base_dirname, output_name)

    sys.argv = (
        ["run.py"]
        + (["--debug"] if args.debug else [])
        + [
            "allennlp",
            "evaluate",
            "--include-package",
            "src",
            "--cuda-device",
            args.gpu,
            "--output-file",
            output_file,
            args.model,
            args.data,
            "-o",
            overrides,
        ]
    )

    print(sys.argv)
    print(" ".join(sys.argv))
    run_main()


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--debug", action="store_true", default=False)
    parse.add_argument("-g", "--gpu", type=str, default="-1", help="CUDA device")
    parse.add_argument("--output-file", type=str)
    parse.add_argument("--output-name", type=str)
    parse.add_argument("--model", type=str, help="model.tar.gz", required=True)
    parse.add_argument("--data", type=str, help="data path", required=True)
    parse.add_argument("-o", "--overrides", type=str, default="{}", help="Overrides")
    args = parse.parse_args()
    return run(args)


if __name__ == "__main__":
    main()
