from dataclasses import dataclass
from typing import (
    Mapping,
    FrozenSet,
    Union,
    DefaultDict,
)
import time
import logging
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.path import AbsolutePath, children
from collections import defaultdict
from .digraph import DiGraph
from .rule import Rule
from .inference_result import InferenceResult
from .extra_data import DictBackend


_l = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleCollection:
    predefined: Mapping[AbsolutePath, FrozenSet[FileAttribute]]
    rules: FrozenSet[Rule]

    def run(self, root: AbsolutePath) -> "InferenceResult":
        backend = DictBackend()
        dependency_graph: DiGraph[Union[Rule, FileAttribute]] = DiGraph()

        all_children = list(children(root))
        attrs: DefaultDict[AbsolutePath, FrozenSet[FileAttribute]] = defaultdict(frozenset)
        for k, v in self.predefined.items():
            attrs[k] = v
        for node in dependency_graph.topological_order():
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
        return InferenceResult(dict(attrs), get_saved=backend.result())

