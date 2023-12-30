from tooling.layout.file_type_inference.framework.rule import (
    Expr, 
    Rule, 
    Not, 
    HasAttribute, 
    And,
)
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import (
    FrozenSet,
    Optional,
    Set,
)

def exclude_blacklisted(expr: Expr) -> Expr:
    return And.from_iter([expr, Not(HasAttribute(FileAttribute.IS_BLACKLISTED))])


def make_update_rules(
    base_name: str,
    is_supported: FileAttribute,
    old_incorrect: FileAttribute,
    old_correct: FileAttribute,
    new_incorrect: FileAttribute,
    new_correct: FileAttribute,
    did_fix: Optional[FileAttribute] = None,
) -> FrozenSet[Rule]:
    rules: Set[Rule] = set()
    rules.add(Rule(f"{base_name}.old_to_new_correct", HasAttribute(old_correct), new_correct))
    rules.add(
        Rule(f"{base_name}.old_incorrect", HasAttribute(is_supported) & Not(HasAttribute(old_correct)), old_incorrect)
    )
    rules.add(
        Rule(f"{base_name}.new_incorrect", HasAttribute(old_incorrect) & Not(HasAttribute(new_correct)), new_incorrect)
    )
    if did_fix is not None:
        rules.add(
            Rule(f"{base_name}.old_to_new_when_fixed", HasAttribute(old_incorrect) & HasAttribute(did_fix), new_correct)
        )
    return frozenset(rules)

