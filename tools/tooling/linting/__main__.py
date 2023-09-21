#! /usr/bin/env python3

import importlib.util
from pathlib import Path
from typing import Dict, Callable, Any
import importlib.machinery
from dataclasses import dataclass
from enum import Enum, auto
from .linting.lint_response import LintResponse
import logging
import json
import sys
from .linters import register_all
from .linter_framework import LinterManager 

DIR = Path(__file__).parent

if __name__ == '__main__':
    mgr = LinterManager()
    mgr.register(LinterSpecification(
        name='include-guards',
        make_args=(
            lambda mod, method: mod.Args(
                path=DIR,
                fix=(method == LinterMethod.FIX),
                force=False,
                log_level=logging.WARN,
        )),
        source_path=DIR / 'fix-include-guards',
        supported_methods=frozenset({
            LinterMethod.FIX, LinterMethod.CHECK
        }),
        requires = frozenset()
    ))
    mgr.register(LinterSpecification(
        name='format',
        make_args=(
            lambda mod, method: mod.Args(
                path=DIR,
                fix=(method == LinterMethod.FIX),
                force=False,
                log_level=logging.WARN,
            )
        ),
        source_path=DIR / 'format',
        supported_methods=frozenset({
            LinterMethod.FIX, LinterMethod.CHECK
        })
    ))
    mgr.register(LinterSpecification(
        name='missing-files',
        make_args=(
            lambda mod, method: mod.Args(
                path=DIR,
                log_level=logging.WARN,
            )
        ),
        source_path=DIR / 'find-missing-files',
        supported_methods=frozenset({
            LinterMethod.CHECK
        })
    ))
    mgr.register(LinterSpecification(
        name='create-missing-src-files',
        make_args=(
        ),
        source_path=DIR / 'create-missing-src-files',
        supported_methods=frozenset({
            LinterMethod.FIX
        })
    ))


    def handle_response_data(response: LintResponse):
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
