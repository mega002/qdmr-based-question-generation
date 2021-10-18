import sklearn

import sys
import argparse
import json

from allennlp.commands import main as allennlp_main
from allennlp.common.params import with_fallback

import logging


def run(args):
    debug = args.debug
    force = args.force
    recover = args.recover
    hard_overrides = args.hard_overrides

    overrides_dict = {}

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    sys.argv = sys.argv[sys.argv.index("allennlp") :]

    overrides_index = -1
    try:
        overrides_index = sys.argv.index("-o") + 1
    except Exception as e1:
        try:
            overrides_index = sys.argv.index("--overrides") + 1
        except Exception as e2:
            pass

    if overrides_index != -1:
        overrides = sys.argv[overrides_index]
        if hard_overrides:
            new_overrides_dict = json.loads(overrides)
            new_overrides_dict.update(overrides_dict)
            sys.argv[overrides_index] = json.dumps(new_overrides_dict)
        else:
            sys.argv[overrides_index] = json.dumps(
                with_fallback(preferred=overrides_dict, fallback=json.loads(overrides))
            )
    else:
        sys.argv += ["-o", json.dumps(overrides_dict)]

    if force:
        sys.argv += ["--force"]

    if recover:
        sys.argv += ["--recover"]

    if args.silent:
        sys.argv += ["--silent"]

    print(sys.argv)
    print(" ".join(sys.argv))
    allennlp_main()


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--debug", action="store_true", default=False)
    parse.add_argument("--force", action="store_true", default=False)
    parse.add_argument("--recover", action="store_true", default=False)
    parse.add_argument("--hard-overrides", action="store_false", default=True)
    parse.add_argument("--silent", action="store_true", default=False)
    args, _ = parse.parse_known_args()
    run(args)


if __name__ == "__main__":
    main()
