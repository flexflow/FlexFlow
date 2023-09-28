#! /usr/bin/env python3

from .framework.response import Response
import json
import sys
from .linters import register_all
from .framework.manager import Manager 

def lint_main():
    mgr = Manager()
    register_all(mgr)
    # mgr.register(Specification(
    #     name='include-guards',
    #     make_args=(
    #         lambda mod, method: mod.Args(
    #             path=DIR,
    #             fix=(method == Method.FIX),
    #             force=False,
    #             log_level=logging.WARN,
    #     )),
    #     source_path=DIR / 'fix-include-guards',
    #     supported_methods=frozenset({
    #         Method.FIX, Method.CHECK
    #     }),
    #     requires = frozenset()
    # ))
    # mgr.register(Specification(
    #     name='format',
    #     make_args=(
    #         lambda mod, method: mod.Args(
    #             path=DIR,
    #             fix=(method == Method.FIX),
    #             force=False,
    #             log_level=logging.WARN,
    #         )
    #     ),
    #     source_path=DIR / 'format',
    #     supported_methods=frozenset({
    #         Method.FIX, Method.CHECK
    #     })
    # ))
    # mgr.register(Specification(
    #     name='missing-files',
    #     make_args=(
    #         lambda mod, method: mod.Args(
    #             path=DIR,
    #             log_level=logging.WARN,
    #         )
    #     ),
    #     source_path=DIR / 'find-missing-files',
    #     supported_methods=frozenset({
    #         Method.CHECK
    #     })
    # ))
    # mgr.register(Specification(
    #     name='create-missing-src-files',
    #     make_args=(
    #     ),
    #     source_path=DIR / 'create-missing-src-files',
    #     supported_methods=frozenset({
    #         Method.FIX
    #     })
    # ))


    def handle_response_data(response: Response):
        if response.json_data is not None:
            return response.json_data
        elif response.message is not None:
            return response.message
        else: 
            return None


    responses = mgr.check()
    data = {k : handle_response_data(r) for k, r in responses.items()}
    print(json.dumps(data, indent=2))

    for response in responses.values():
        if response.return_code != 0:
            sys.exit(1)
