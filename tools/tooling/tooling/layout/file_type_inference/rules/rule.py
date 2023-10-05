from abc import ABC, abstractmethod, abstractproperty
from tooling.layout.path import AbsolutePath, full_suffix, children
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet, Callable, Iterable, Any, Set, Optional
from dataclasses import dataclass

Attrs = Callable[[AbsolutePath], FrozenSet[FileAttribute]]
SaveFunc = Callable[[Any], None]

@dataclass(frozen=True)
class ExprExtra:
    save: SaveFunc

class Expr(ABC):
    def __and__(self, other: 'Expr') -> 'Expr':
        return And.from_iter([self, other])

    def __or__(self, other: 'Expr') -> 'Expr':
        return Or.from_iter([self, other])

    @abstractmethod
    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        ...

    @abstractproperty
    def inputs(self) -> FrozenSet[FileAttribute]:
        ...

@dataclass(frozen=True)
class HasAttribute(Expr):
    attribute: FileAttribute

    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        return self.attribute in attrs(p)

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return frozenset([self.attribute])

    def __repr__(self) -> str:
        return f'?{self.attribute}'

@dataclass(frozen=True)
class HasAnyOfAttributes(Expr):
    attributes: FrozenSet[FileAttribute]

    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        return any(attr in attrs(p) for attr in self.attributes)

    @staticmethod
    def from_iter(it: Iterable[FileAttribute]) -> 'HasAnyOfAttributes':
        return HasAnyOfAttributes(frozenset(it))

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return self.attributes

    def __repr__(self) -> str:
        return 'HasAny(' + ', '.join(repr(a) for a in self.attributes) + ')'

@dataclass(frozen=True)
class HasAllOfAttributes(Expr):
    attributes: FrozenSet[FileAttribute]

    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        return all(attr in attrs(p) for attr in self.attributes)

    @staticmethod
    def from_iter(it: Iterable[FileAttribute]) -> 'HasAllOfAttributes':
        return HasAllOfAttributes(frozenset(it))

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return self.attributes


@dataclass(frozen=True)
class And(Expr):
    children: FrozenSet[Expr]

    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        return all(child.evaluate(p, attrs, extra=extra) for child in self.children)

    def __and__(self, other: Expr) -> Expr:
        if isinstance(other, And):
            return And(self.children | other.children)
        else:
            return And.from_iter([*self.children, other])

    @staticmethod
    def from_iter(it: Iterable[Expr]) -> 'And':
        return And(frozenset(it))

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        result: FrozenSet[FileAttribute] = frozenset()
        for child in self.children:
            result |= child.inputs
        return result

    def __repr__(self) -> str:
        return 'And(' + ', '.join(repr(c) for c in self.children) + ')'

@dataclass(frozen=True)
class Or(Expr):
    children: FrozenSet[Expr]

    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        return any(child.evaluate(p, attrs, extra=extra) for child in self.children)

    @staticmethod
    def from_iter(it: Iterable[Expr]) -> 'Or':
        return Or(frozenset(it))

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        result: FrozenSet[FileAttribute] = frozenset()
        for child in self.children:
            result |= child.inputs
        return result

@dataclass(frozen=True)
class Not(Expr):
    expr: Expr

    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        return not self.expr.evaluate(p, attrs, extra=extra)

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return self.expr.inputs

    def __repr__(self) -> str:
        return f'!{self.expr}'

@dataclass(frozen=True)
class ChildSatisfies(Expr):
    name: str
    expr: Expr

    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        return (p / self.name).exists() and self.expr.evaluate((p / self.name), attrs, extra=extra)

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return self.expr.inputs

    def __repr__(self) -> str:
        return f'v@/{self.name}/{self.expr}'

@dataclass(frozen=True)
class IsNamed(Expr):
    name: str

    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        return p.name == self.name

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return frozenset()

    def __repr__(self) -> str:
        return f'?/{self.name}/'

@dataclass(frozen=True)
class IsFile(Expr):
    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        return p.is_file()

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return frozenset()

    def __repr__(self) -> str:
        return '?F?'

@dataclass(frozen=True)
class IsDir(Expr):
    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        return p.is_dir()

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return frozenset()

    def __repr__(self) -> str:
        return '?D?'

@dataclass(frozen=True)
class HasExtension(Expr):
    extension: str

    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        return full_suffix(p) == self.extension

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return frozenset()

    def __repr__(self) -> str:
        return f'?/*{self.extension}/'

@dataclass(frozen=True)
class ParentSatisfies(Expr):
    expr: Expr

    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        return self.expr.evaluate(p.parent, attrs, extra=extra)

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return self.expr.inputs

    def __repr__(self) -> str:
        return f'^{self.expr}'

@dataclass(frozen=True)
class AncestorSatisfies(Expr):
    expr: Expr

    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        return any(self.expr.evaluate(parent, attrs, extra=extra) for parent in p.parents)

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return self.expr.inputs

    def __repr__(self) -> str:
        return f'^^{self.expr}'

@dataclass(frozen=True)
class FileContentsSatisfy(Expr):
    condition: Callable[[AbsolutePath, str], bool]

    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        if not p.is_file(): 
            return False
        try:
            with p.open('r') as f:
                contents = f.read()
            return self.condition(p, contents)
        except UnicodeDecodeError:
            return False

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return frozenset()

@dataclass(frozen=True)
class OpaqueFunction(Expr):
    precondition: Expr
    func: Callable[[AbsolutePath, Attrs, ExprExtra], bool]
    extra_inputs: FrozenSet[FileAttribute] = frozenset()

    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        if not self.precondition.evaluate(p, attrs, extra=extra):
            return False

        return self.func(p, lambda path: attrs(path).intersection(self.inputs), extra)

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return self.precondition.inputs.union(self.extra_inputs)

@dataclass(frozen=True)
class DoesNotCreateNesting(Expr):
    expr: Expr

    def evaluate(self, p: AbsolutePath, attrs: Attrs, extra: ExprExtra) -> bool:
        for parent in p.parents:
            if self.expr.evaluate(parent, attrs, extra=extra):
                return False
        if p.is_dir():
            for child in children(p):
                if self.expr.evaluate(child, attrs, extra=extra):
                    return False
        return True

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return self.expr.inputs

    def __repr__(self) -> str:
        return f'<!>{self.expr}'

@dataclass(frozen=True)
class Rule:
    name: str
    condition: Expr
    result: FileAttribute

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return self.condition.inputs

    @property
    def outputs(self) -> FrozenSet[FileAttribute]:
        return frozenset({self.result})

    def __repr__(self) -> str:
        return self.name

    @property
    def signature(self) -> str:
        return f'{self.name} :: ({", ".join(map(str, self.inputs))}) -> {self.result}'

def make_update_rules(
        base_name: str,
        is_supported: FileAttribute,
        old_incorrect: FileAttribute, 
        old_correct: FileAttribute, 
        new_incorrect: FileAttribute, 
        new_correct: FileAttribute, 
        did_fix: Optional[FileAttribute] = None
) -> FrozenSet[Rule]:
    rules: Set[Rule] = set()
    rules.add(Rule(
        f'{base_name}.old_to_new_correct',
        HasAttribute(old_correct), 
        new_correct
    ))
    rules.add(Rule(
        f'{base_name}.old_incorrect',
        HasAttribute(is_supported) & Not(HasAttribute(old_correct)), 
        old_incorrect
    ))
    rules.add(Rule(
        f'{base_name}.new_incorrect',
        HasAttribute(old_incorrect) & Not(HasAttribute(new_correct)), 
        new_incorrect
    ))
    if did_fix is not None:
        rules.add(Rule(
            f'{base_name}.old_to_new_when_fixed',
            HasAttribute(old_incorrect) & HasAttribute(did_fix), 
            new_correct
        ))
    return frozenset(rules)
