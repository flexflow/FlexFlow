from tooling.util.lisp.combinators import (
    anychar,
    char,
    matches,
    ParseError,
    exactly,
    or_else,
    and_then,
    repeat_until,
    Parser,
    maybe,
    undo,
    negate,
    repeat,
    one_or_more,
    leaftree,
)
from contextlib import contextmanager
from typing import Iterator

import pytest
import hypothesis.strategies as st
from hypothesis import given, assume

@contextmanager
def fails_to_parse(s: str) -> Iterator[str]:
    with pytest.raises(ParseError):
        yield s

def test_anychar() -> None:
    with pytest.raises(ParseError):
        anychar('')

    anychar('abc') == ('a', 'bc')
    anychar('a') == ('a', '')

def test_char() -> None:
    with fails_to_parse('b') as s:
        char('a')(s)

    char('a')('abc') == ('a', 'bc')
    char('a')('a') == ('a', '')

    with pytest.raises(ValueError):
        char('ab')

def test_undo() -> None:
    assert undo(char('a'))('abc') == ('a', 'abc')

    with fails_to_parse('abc') as s:
        undo(char('b'))(s)

def test_negate() -> None:
    assert negate(char('b'))('abc') == (None, 'abc')

    with fails_to_parse('abc') as s:
        assert negate(char('a'))(s)

def test_matches() -> None:
    assert matches(char('a'))('abc') == (True, 'abc')
    assert matches(char('b'))('abc') == (False, 'abc')

def test_lshift() -> None:
    assert (char('a') << char('b'))('abc') == ('a', 'c')

    with fails_to_parse('abc') as s:
        (char('b') << char('a'))(s)

def test_rshift() -> None:
    assert (char('a') >> char('b'))('abc') == ('b', 'c')

    with fails_to_parse('abc') as s:
        (char('b') >> char('a'))(s)

def test_exactly() -> None:
    assert (exactly('a'))('abc') == ('a', 'bc')
    assert (exactly('ab'))('abc') == ('ab', 'c')
    assert (exactly('abc'))('abc') == ('abc', '')
    assert (exactly(''))('abc') == ('', 'abc')

    with fails_to_parse('abc') as s:
        exactly('abcd')(s)

    with fails_to_parse('cbc') as s:
        exactly('abc')(s)

    with fails_to_parse('ab') as s:
        exactly('abc')(s)

def test_or_else() -> None:
    assert or_else(char('a'), exactly('ab'))('abc') == ('a', 'bc')
    assert or_else(exactly('acb'), exactly('ab'))('abc') == ('ab', 'c')

    with fails_to_parse('abc') as s:
        or_else(char('b'), char('c'))(s)

    with fails_to_parse('') as s:
        or_else(char('b'), char('c'))(s)


def test_or_operator() -> None:
    assert (char('a') | exactly('ab'))('abc') == ('a', 'bc')
    assert (exactly('acb') | exactly('ab'))('abc') == ('ab', 'c')

    with fails_to_parse('abc') as s:
        (char('b') | char('c'))(s)

    with fails_to_parse('') as s:
        (char('b') | char('c'))(s)

def test_and_then() -> None:
    assert and_then(char('a'), char('b'))('abc') == (('a', 'b'), 'c')

    with fails_to_parse('abc') as s:
        assert and_then(char('a'), char('a'))(s)

    with fails_to_parse('abc') as s:
        assert and_then(char('b'), char('b'))(s)

def test_and_operator() -> None:
    assert (char('a') & char('b'))('abc') == (('a', 'b'), 'c')

    with fails_to_parse('abc') as s:
        assert (char('a') & char('a'))(s)

    with fails_to_parse('abc') as s:
        assert (char('b') & char('b'))(s)

@given(...)
def test_constant(vv: str, rest: str) -> None:
    assert Parser.constant(vv)(rest) == (vv, rest)

@given(
    f=st.functions(like=lambda x: x, pure=True),
    s=...,
    vv=...,
)
def test_fmap(f, vv: str, s: str) -> None:
    assert Parser.constant(vv).fmap(f)(s) == (f(vv), s)

class TestBind:
    @given(...)
    def properties(self, s: str) -> None:
        assert maybe(char('a').bind(lambda v: Parser.constant(v)))(s) == maybe(char('a'))(s)

    def manual(self) -> None:
        assert char('a').bind(lambda x: char('b').fmap(lambda y: (x, y)))('abc') == (('a', 'b'), 'c')

        with fails_to_parse('abc') as s:
            char('b').bind(lambda _: Pure.constant(None))(s)

        with fails_to_parse('abc') as s:
            char('a').bind(lambda _: Pure.fail())(s)

# @pytest.xfail('TODO: iter_of')
# def test_iter_of() -> None:
#     pass

def test_repeat_until() -> None:
    vv, rest = repeat_until(char('b'), char('a'))('aabba')
    assert list(vv) == ['a', 'a']
    assert rest == 'ba'

    vv, rest = repeat_until(undo(char('b')), char('a'))('aabba')
    assert list(vv) == ['a', 'a']
    assert rest == 'bba'

    vv, rest = repeat_until(char('a'), char('a'))('aaba')
    assert list(vv) == []
    assert rest == 'aba'

    vv, rest = repeat_until(char('a'), exactly('ba'))('bababaaba')
    assert list(vv) == ['ba', 'ba', 'ba']
    assert rest == 'ba'

    with fails_to_parse('bababa') as s:
        repeat_until(char('a'), exactly('ba'))(s)

def test_repeat() -> None:
    assert repeat(exactly('ab'))('aba') == ('ab', 'a')
    
    vv, rest = repeat(exactly('ab'))('abababa')
    assert list(vv) == ['ab', 'ab', 'ab']
    assert rest == 'a'

    with fails_to_parse('baab') as s:
        repeat(exactly('ab'))(s)

def test_one_or_more() -> None:
    vv, rest = one_or_more(exactly('ab'))('aba')
    assert list(vv) == ['ab']
    assert rest == 'a'
    
    vv, rest = one_or_more(exactly('ab'))('abababa')
    assert list(vv) == ['ab', 'ab', 'ab']
    assert rest == 'a'

    with fails_to_parse('baab') as s:
        one_or_more(exactly('ab'))(s)

# @pytest.xfail('TODO: leaftree')
# def test_leaftree(): 
#     pass
