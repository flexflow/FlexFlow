from typing import TypeVar, Callable, Tuple, Generic, Optional, Iterable, List, Union, Any
from dataclasses import dataclass
import itertools

class ParseError(Exception): 
    def __init__(self, msg: str):
        pass

def require(b: bool, msg: Optional[str] = None) -> None:
    if msg is None:
        msg = "unknown"
    if not b:
        raise ParseError(msg=msg)

TT = TypeVar('TT', covariant=True)

T = TypeVar('T')
T2 = TypeVar('T2', covariant=True)
T3 = TypeVar('T3')

def require_eq(lhs: T, rhs: T, msg: Optional[str] = None) -> None:
    return require(lhs == rhs)

@dataclass(frozen=True)
class Parser(Generic[TT]):
    _func: Callable[[str], Tuple[TT, str]]

    def __call__(self, rest: str) -> Tuple[TT, str]:
        return self._func(rest)

    @staticmethod
    def pure(f: Callable[[str], T]) -> 'Parser[T]':
        def _f(rest: str, f: Callable[[str], T]=f) -> Tuple[T, str]:
            return (f(rest), rest)
        return Parser(_f)

    @staticmethod
    def constant(t: T) -> 'Parser[T]':
        return Parser.pure(lambda _: t)

    def fmap(self, f: Callable[[TT], T2]) -> 'Parser[T2]':
        def _f(rest: str, f: Callable[[TT], T2]=f) -> Tuple[T2, str]:
            val, _rest = self(rest)
            return f(val), _rest
        return Parser(_f)

    def bind(self, f: Callable[[TT], 'Parser[T2]']) -> 'Parser[T2]':
        def _f(rest: str, f: Callable[[TT], Parser[T2]]=f) -> Tuple[T2, str]:
            v, rest1 = self(rest)
            return f(v)(rest1)
        return Parser(_f)

    def __or__(self, other: 'LazyParser[T2]') -> 'Parser[Union[T, T2]]':
        return or_else(self, other)

    def __and__(self, other: 'LazyParser[T2]') -> 'Parser[Tuple[TT, T2]]':
        return and_then(self, other)

    def __rshift__(self, other: 'Parser[T2]') -> 'Parser[T2]':
        def _f(rest: str, other: Parser[T2]=other) -> Tuple[T2, str]:
            _, rest1 = self(rest)
            return other(rest1)
        return Parser(_f)

    def __lshift__(self, other: 'Parser[T2]') -> 'Parser[TT]':
        def _f(rest: str, other: Parser[T2]=other) -> Tuple[TT, str]:
            v, rest1 = self(rest)
            _, rest2 = other(rest1)
            return v, rest2
        return Parser(_f)


LazyParser = Union[Parser[T], Callable[[], Parser[T]]]

empty: Parser[bool] = Parser.pure(lambda rest: len(rest) == 0)
true: Parser[bool] = Parser.constant(True)
false: Parser[bool] = Parser.constant(False)
empty_iter: Parser[Iterable[Any]] = Parser.constant(tuple())

def _fail(rest: str) -> Tuple[Any, str]:
    require(False, "fail")
    assert False

fail = Parser(_fail)

def _anychar(rest: str) -> Tuple[str, str]:
    require(len(rest) > 0)
    return rest[0], rest[1:]

anychar: Parser[str] = Parser(_anychar)

def notchar(c: str) -> Parser[str]:
    return anychar.bind(lambda cc: Parser.constant(cc) if c == cc else fail)

def exactly(s: str) -> Parser[str]:
    def _f(rest: str, pattern: str = s) -> Tuple[str, str]:
        require(rest.startswith(s))
        return (s, rest[len(s):])
    return Parser(_f)

def char(c: str) -> Parser[str]:
    if len(c) != 1:
        raise ValueError(f'Invalid character of length {len(c)}: "{c}"')
    def _f(rest: str, c: str = c) -> Tuple[str, str]:
        require(len(rest) > 0 and rest[0] == c)
        return (rest[0], rest[1:])
    return Parser(_f)

def inchars(c: str) -> Parser[str]:
    return oneof(char(cc) for cc in c)

def notinchars(cs: str) -> Parser[str]:
    result = anychar
    for c in cs:
        result = notchar(c) >> anychar
    return result

def undo(p: Parser[T]) -> Parser[T]:
    def _f(rest: str) -> Tuple[T, str]:
        v, _ = p(rest) 
        return v, rest
    return Parser(_f)

def negate(p: Parser[T]) -> Parser[None]:
    def _f(rest: str) -> Tuple[None, str]:
        try:
            p(rest)
        except ParseError:
            return (None, rest)
        else:
            require(False)
    return Parser(_f)

def must_match(p: Parser[bool]) -> Parser[bool]:
    def _f(rest: str) -> Tuple[str, bool]:
        vv, rest1 = p(rest)
        require(vv)
        return (vv, rest1)
    return Parser(_f)

def discard(p: Parser[str]) -> Parser[str]:
    return p.fmap(lambda _: '')

def matches(p: Parser[T]) -> Parser[bool]:
    def _f(rest: str, _p: Parser[T] = p) -> Tuple[bool, str]:
        try:
            _p(rest)
            return (True, rest)
        except ParseError:
            return (False, rest)
    return Parser(_f)

def notmatches(p: Parser[T]) -> Parser[bool]:
    return matches(p).fmap(lambda b: not b)

def or_else(fst: Parser[T], snd: LazyParser[T2]) -> Parser[Union[T, T2]]:
    def _f(rest: str) -> Tuple[Union[T, T2], str]:
        val, _ = matches(fst)(rest)
        if val:
            return fst(rest)
        else:
            return eval_thunk(snd)(rest)
    return Parser(_f)

def maybe(p: Parser[T]) -> Parser[Optional[T]]:
    return or_else(p, Parser.constant(None))

def oneof(ps: Iterable[Parser[T]]) -> Parser[T]:
    result = fail
    for p in ps:
        result = or_else(p, result) 
    return result

def sequence(ps: Iterable[Parser[T]]) -> Parser[Iterable[T]]:
    result: Parser[Iterable[T]] = Parser.constant(tuple())
    for p in ps: 
        result = result.bind(lambda accum: p.fmap(lambda v: (*accum, v)))
    return result

def iter_of(p: Parser[T]) -> Parser[Iterable[T]]:
    return p.fmap(lambda v: tuple([v]))

def concat(p: Parser[Iterable[T]], p2: LazyParser[Iterable[T]]) -> Parser[Iterable[T]]:
    return (p & p2).fmap(lambda t: (*t[0], *t[1]))

def eval_thunk(p: LazyParser[T]) -> Parser[T]:
    if isinstance(p, Parser):
        return p
    else:
        evaled = p()
        assert isinstance(evaled, Parser)
        return evaled

def and_then(fst: Parser[T], snd: LazyParser[T2]) -> Parser[Tuple[T, T2]]:
    def _f(rest: str, fst: Parser[T] = fst, snd: LazyParser[T2] = snd) -> Tuple[Tuple[T, T2], str]:
        val1, rest1 = fst(rest)
        val2, rest2 = eval_thunk(snd)(rest1)
        return (val1, val2), rest2
    return Parser(_f)

def until(cond: Parser[T]) -> Parser[str]:
    def _f(rest: str, term_cond: Parser[T]=cond) -> Tuple[str, str]:
        result: List[str] = []
        while not matches(cond)(rest):
            result.append(rest[0])
            rest = rest[1:]
        return ''.join(result), rest
    return Parser(_f)

# def ite(cond: Parser[bool], tt: Parser[T2], ff: Parser[T3]) -> Parser[Union[T2, T3]]:
#     return cond.bind(lambda c: tt if c else ff)

def repeat_until(end: Parser[T2], reg: Parser[T]) -> Parser[Iterable[T]]:
    return (end >> empty_iter) | (lambda: concat(iter_of(reg), repeat_until(end, reg)))

def repeat(reg: Parser[T]) -> Parser[Iterable[T]]:
    return repeat_until(notmatches(reg), reg)

def one_or_more(p: Parser[T]) -> Parser[Iterable[T]]:
    return concat(iter_of(p), repeat(p))

def join(p: Parser[Iterable[str]]) -> Parser[str]:
    return p.fmap(lambda ss: ''.join(ss))

def chain(ps: Parser[Iterable[Iterable[T]]]) -> Parser[Iterable[T]]:
    return ps.fmap(itertools.chain.from_iterable)

@dataclass(frozen=True)
class LeafTree(Generic[T]):
    children: Tuple[Union[T, 'LeafTree[T]'], ...]

    @staticmethod
    def from_iter(it: Iterable[Union[T, 'LeafTree[T]']]) -> 'LeafTree[T]':
        return LeafTree(tuple(it))

def leaftree(_open: Parser[T2], child: Parser[T], _close: Parser[T3]) -> Parser[Union[T, LeafTree[T]]]:
    return or_else(
        _open >> repeat_until(
            _close,
            leaftree(_open, child, _close),
        ).fmap(lambda vs: LeafTree.from_iter(vs)),
        child
    )
        
         
