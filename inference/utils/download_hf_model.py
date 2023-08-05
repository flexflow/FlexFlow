#!/usr/bin/env python
import flexflow.serve as ff
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", type=str, nargs="+", help="Name of the model(s) to download"
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        help="Folder to use to store the weights",
        default="",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--full-precision-only",
        action="store_true",
        help="Only download the full precision version",
    )
    group.add_argument(
        "--half-precision-only",
        action="store_true",
        help="Only download the half precision version",
    )
    args = parser.parse_args()
    return args


def main(args):
    ff.init_cpu()
    print(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
