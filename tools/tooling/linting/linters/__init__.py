from ..framework.manager import Manager
# from ..framework.specification import Specification
from .find_missing_files import register as register_find_missing_files
from .fix_include_guards import register as register_fix_include_guards
from .clang_format import register as register_clang_format
from .clang_tidy import register as register_clang_tidy

def register_all(mgr: Manager) -> None:
    register_find_missing_files(mgr)
    register_fix_include_guards(mgr)
    register_clang_format(mgr)
    register_clang_tidy(mgr)
    
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

