import tooling.linting.linters.find_missing_files as find_missing_files
import tooling.linting.linters.fix_include_guards as fix_include_guards 
import tooling.linting.linters.clang_format as clang_format
import tooling.linting.linters.clang_tidy as clang_tidy
from tooling.linting.framework.manager import Manager

def all_linters() -> Manager:
    return Manager(frozenset({
        find_missing_files.spec,
        fix_include_guards.spec,
        clang_format.spec,
        clang_tidy.spec
    }))
    
    # mgr.register(LinterSpecification(
    #     name='include-guards',
    #     func=
    #         lambda mod, method: mod.Args(
    #             path=DIR,
    #             fix=(method == LinterMethod.FIX),
    #             force=False,
    #             log_level=logging.WARN,
    #     )),
    #     source_path=DIR / 'fix-include-guards',
    #     supported_methods=frozenset({
    #         LinterMethod.FIX, LinterMethod.CHECK
    #     }),
    #     requires = frozenset()
    # ))
    # mgr.register(LinterSpecification(
    #     name='format',
    #     make_args=(
    #         lambda mod, method: mod.Args(
    #             path=DIR,
    #             fix=(method == LinterMethod.FIX),
    #             force=False,
    #             log_level=logging.WARN,
    #         )
    #     ),
    #     source_path=DIR / 'format',
    #     supported_methods=frozenset({
    #         LinterMethod.FIX, LinterMethod.CHECK
    #     })
    # ))
    # mgr.register(LinterSpecification(
    #     name='create-missing-src-files',
    #     make_args=(
    #     ),
    #     source_path=DIR / 'create-missing-src-files',
    #     supported_methods=frozenset({
    #         LinterMethod.FIX
    #     })
    # ))

