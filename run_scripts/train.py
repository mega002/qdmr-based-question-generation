import argparse
import json
import os
import sys

import namegenerator

from run import main as run_main
from predict import main as run_predict


def run(args):
    if args.pre_serialization_dir:
        os.environ["weights"] = f"{args.pre_serialization_dir}/best.th"
        os.environ["pre_serialization_dir"] = args.pre_serialization_dir

    overrides_dict = {}
    if args.debug:
        overrides_dict.update(
            {
                "dataset_reader": {"max_instances": 50, "pickle": None},
                "validation_dataset_reader": {"max_instances": 50, "pickle": None},
            }
        )

    cuda_device = args.gpu
    if cuda_device != "":
        cuda_device = eval(cuda_device)
        if isinstance(cuda_device, int):
            overrides_dict["trainer"] = {"cuda_device": cuda_device}

        elif isinstance(cuda_device, list):
            overrides_dict["distributed"] = {"cuda_devices": cuda_device}
        else:
            raise ValueError("cuda_devices must be a list of a int")

    if args.serialization_dir is None:
        if args.auto_dir:
            index = 0
            basename = os.path.splitext(os.path.basename(args.config_file))[0]
            base_dirname = os.path.basename(os.path.dirname(args.config_file))
            base_serialization_dir = f"../experiments/{base_dirname}_{basename}"
            while True:
                serialization_dir = f"{base_serialization_dir}_{index}"
                if os.path.exists(serialization_dir):
                    index += 1
                else:
                    break
            if args.force and index > 0:
                index -= 1
                serialization_dir = f"{base_serialization_dir}_{index}"
        else:
            serialization_dir = f"../experiments/{namegenerator.gen()}"
    else:
        serialization_dir = args.serialization_dir

    overrides_dict.update(json.loads(args.overrides))
    overrides = json.dumps(overrides_dict)

    sys.argv = (
        ["run.py"]
        + (["--debug"] if args.debug else [])
        + (["--force"] if args.force else [])
        + (["--recover"] if args.recover else [])
        + [
            "allennlp",
            "train",
            args.config_file,
            "-s",
            serialization_dir,
            "--include-package",
            "src",
            "-o",
            overrides,
        ]
    )

    print(sys.argv)
    print(" ".join(sys.argv))
    run_main()

    if args.predict_with_best:
        if cuda_device != "":
            cuda_device = (
                cuda_device if isinstance(cuda_device, int) else cuda_device[0]
            )
        else:
            cuda_device = -1

        for data_path in ["data/strategy_qa/dev.json", "data/strategy_qa/test.json"]:
            sys.argv = (
                ["predict.py"]
                + (["--debug"] if args.debug else [])
                + (["--silent"])
                + [
                    "-g",
                    str(cuda_device),
                    "--model",
                    os.path.join(serialization_dir, "model.tar.gz"),
                    "--data",
                    data_path,
                    "-o",
                    overrides,
                ]
            )
            print(sys.argv)
            print(" ".join(sys.argv))
            run_predict()

    return serialization_dir


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--debug", action="store_true", default=False)
    parse.add_argument("--force", action="store_true", default=False)
    parse.add_argument("--recover", action="store_true", default=False)
    parse.add_argument(
        "-a",
        "--auto-dir",
        action="store_true",
        default=False,
        help="Auto-dir: Define serialization dir by the path of the config file",
    )
    parse.add_argument("-g", "--gpu", type=str, default="", help="CUDA device")
    parse.add_argument(
        "-c",
        "--config-file",
        type=str,
        help="Config file, must for train",
        required=True,
    )
    parse.add_argument(
        "-s", "--serialization-dir", type=str, help="Serialization dir, must for train"
    )
    parse.add_argument("-o", "--overrides", type=str, default="{}", help="Overrides")
    parse.add_argument(
        "-p", "--pre-serialization-dir", type=str, help="Pretrained serialization dir"
    )
    parse.add_argument("--predict-with-best", action="store_true", default=False)
    args = parse.parse_args()
    return run(args)


if __name__ == "__main__":
    main()
