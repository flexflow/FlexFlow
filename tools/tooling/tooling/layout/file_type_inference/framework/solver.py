from typing import FrozenSet, Union, Iterator, Dict, DefaultDict
import time
import logging
from .rule import Rule
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.path import AbsolutePath, children
from .extra_data import ExprExtraBackend, DictBackend
from collections import defaultdict
from .digraph import DiGraph


_l = logging.getLogger(__name__)


def construct_dependency_graph(rules: FrozenSet[Rule]) -> DiGraph[Union[Rule, FileAttribute]]:
    g: DiGraph[Union[Rule, FileAttribute]] = DiGraph()
    for rule in rules:
        g.add_node(rule)
        for inp in rule.inputs:
            _l.debug(f"Adding dependency from {inp} to {rule.name}")
            g.add_edge(inp, rule)
        for out in rule.outputs:
            _l.debug(f"Adding dependency from {rule.name} to {out}")
            g.add_edge(rule, out)
    _l.debug("Built dependency graph:\n" + g.dot())

    _l.debug("Checking dependency graph for cycles")
    assert g.is_acyclic()
    return g


class InferenceSolver:
    """
    File type inference is performed similar to a logical programming language, e.g., Datalog, in that 
    the programmer provides a set of implications ("rules"), and then the solver iteratively applies these
    rules until a fixed point is reached. `InferenceSolver` differs from datalog into two primary ways:
    (1) no implied negation is allowed, and (2) rules may perform side effects. The first restriction is added for
    simplicity (as implied negation is not necessary for any of our rules). Side effects provide the mechanism by 
    which linters are run--when the sufficient conditions are met for a linter rule, the linter rule then runs in a 
    "dry-run" mode to determine if changes need to be made, and then can update the file if the corresponding arguments
    are passed on the command line. This provides a nice unification of "fixing" and "linting", and also allows 
    dependencies between linters.

    Attributes: 
        rules: A set of inference rules. For more information, see `Rule`
        root: The root filesystem directory over. Only paths within this directory are considered.
    """
    rules: FrozenSet[Rule]
    root: AbsolutePath

    _solution: Dict[FileAttribute, FrozenSet[AbsolutePath]]
    _dependency_graph: DiGraph[Union[Rule, FileAttribute]]
    _backend: ExprExtraBackend

    def __init__(self, rules: FrozenSet[Rule], root: AbsolutePath) -> None:
        self.rules = rules
        self.root = root
        self._dependency_graph = construct_dependency_graph(rules)
        self._solution = dict()
        self._backend = DictBackend()

    def _get_attr_execution_plan(self, attr: FileAttribute) -> Iterator[Rule]:
        dependencies = self._dependency_graph.ancestors(attr)
        for node in self._dependency_graph.topological_order():
            if isinstance(node, Rule) and (node in dependencies) and (node not in self._solution):
                yield node

    def _run_rule(self, rule: Rule) -> None:
        all_children = list(children(self.root))
        attrs: DefaultDict[AbsolutePath, FrozenSet[FileAttribute]] = defaultdict(frozenset)
        for k, v in self.predefined.items():
            attrs[k] = v
        for node in self._dependency_graph.topological_order():
            if isinstance(node, Rule):
                _l.debug(f"Running rule {node.signature}")
                start_time = time.perf_counter()
                extra = backend.for_rule(node)
                num_added = 0
                for p in all_children:
                    if node.condition.evaluate(p, lambda path: attrs[path], extra=extra):
                        num_added += 1
                        attrs[p] |= node.outputs
                end_time = time.perf_counter()
                rule_time = end_time - start_time
                _l.debug(
                    f"Rule {node.name} found {num_added} paths that satisfy {node.result} (took {rule_time*1000:.4f}ms)"
                )

    def _eval_attr(self, attr: FileAttribute) -> None:
        for rule in self._get_attr_execution_plan(attr):
            pass

    def __getitem__(self, attr: FileAttribute) -> FrozenSet[AbsolutePath]:
        if attr not in self._solution:
            self._eval_attr(attr)
        return self._solution[attr]
