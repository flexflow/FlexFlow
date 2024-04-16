#!/usr/bin/env python
import flexflow.serve as ff
import argparse, os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_names", type=str, nargs="+", help="Name of the model(s) to download"
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        help="Folder to use to store the model(s) assets in FlexFlow format",
        default=os.environ.get("FF_CACHE_PATH", ""),
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Use this flag to force the refresh of the model(s) weights/tokenizer cache",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--full-precision-only",
        action="store_true",
        help="Only download the full precision version of the weights",
    )
    group.add_argument(
        "--half-precision-only",
        action="store_true",
        help="Only download the half precision version of the weights",
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.full_precision_only:
        data_types = (ff.DataType.DT_FLOAT,)
    elif args.half_precision_only:
        data_types = (ff.DataType.DT_HALF,)
    else:
        data_types = (ff.DataType.DT_FLOAT, ff.DataType.DT_HALF)

    for model_name in args.model_names:
        for data_type in data_types:
            llm = ff.LLM(
                model_name,
                data_type=data_type,
                cache_path=args.cache_folder,
                refresh_cache=args.refresh_cache,
            )
            llm.download_hf_weights_if_needed()
            llm.download_hf_tokenizer_if_needed()
            llm.download_hf_config()


if __name__ == "__main__":
    args = parse_args()
    main(args)
