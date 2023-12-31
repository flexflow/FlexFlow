from .parse import parse, name_literal, string_literal, token, child, wschar
from .ast import StringLiteral, NameLiteral, SExpr
from .combinators_test import fails_to_parse
from .combinators import repeat

def test_name_literal() -> None:
    assert name_literal('a') == (NameLiteral('a'), '')
    assert name_literal('abc') == (NameLiteral('abc'), '')
    assert name_literal('abc def') == (NameLiteral('abc'), ' def')
    assert name_literal('abc)') == (NameLiteral('abc'), ')')

    with fails_to_parse('(abc)') as s:
        name_literal(s)

def test_string_literal() -> None:
    assert string_literal('""') == (StringLiteral(''), '')
    assert string_literal('"abc def" ijk') == (StringLiteral('abc def'), ' ijk')
    assert string_literal(r'"abc\"def" ijk') == (StringLiteral('abc"def'), ' ijk')
    assert string_literal(r'"abc"def" ijk') == (StringLiteral('abc'), 'def" ijk')
    assert string_literal('"(a b c)"') == (StringLiteral('(a b c)'), '')

    with fails_to_parse('"a') as s:
        string_literal(s)
    with fails_to_parse('ab') as s:
        string_literal(s)

def test_child() -> None:
    assert child('a') == (NameLiteral('a'), '')
    assert child('a)') == (NameLiteral('a'), ')')
    assert child('  a )') == (NameLiteral('a'), ')')

def test_parse() -> None:
    assert parse('(a)') == SExpr.from_iter([
        NameLiteral('a'),
    ])
    assert parse('(a (b  "c(e)")\nd)') == SExpr.from_iter([
        NameLiteral('a'),
        SExpr.from_iter([
            NameLiteral('b'),
            StringLiteral('c(e)'),
        ]),
        NameLiteral('d'),
    ])

    assert parse('"abcd"') == StringLiteral('abcd')
    assert parse(r'"abc\"def"') == StringLiteral('abc"def')

    with fails_to_parse('"abc"def"') as s:
        parse(s)
