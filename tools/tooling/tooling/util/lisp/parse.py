from .ast import StringLiteral, NameLiteral, SExpr, AST
from .combinators import repeat_until, char, inchars, Parser, join, matches, anychar, one_or_more, leaftree, repeat, notinchars, LeafTree
from typing import Union, Iterable
# def parse_str_literal(s: str) -> StringLiteral:
#     assert s.startswith('"')
#     require('"' in s[1:])
#     result = []
#     idx = 1
#     while s[idx] != '\\':
#         result.append(idx)

escape_code: Parser[str] = (char('\\') | inchars('"')).fmap(lambda s: s[1])
string_piece: Parser[str] = join(repeat_until(matches(inchars(r'\"')), anychar))
string: Parser[str] = char('"') >> join(repeat_until(char('"'), escape_code | string_piece))
wschar: Parser[str] = inchars(' \n\t')
ws: Parser[str] = join(one_or_more(wschar))
openpar = char('(')
closepar = char(')')

name_literal: Parser[NameLiteral] = join(repeat_until(wschar | openpar | closepar, anychar)).fmap(NameLiteral)
string_literal: Parser[StringLiteral] = string.fmap(StringLiteral)

Token = Union[StringLiteral, NameLiteral]

token: Parser[Token] = string_literal | name_literal
tokens: Parser[Iterable[Token]] = repeat(repeat(wschar) >> token << repeat(wschar))

base_s_expr: Parser[SExpr] = (openpar >> tokens << closepar).fmap(SExpr.from_iter)

def to_ast(t: Union[Token, LeafTree[Token]]) -> AST:
    if isinstance(t, (StringLiteral, NameLiteral)):
        return t
    else:
        return SExpr.from_iter(to_ast(c) for c in t.children)

ast: Parser[AST] = leaftree(openpar, repeat(wschar) >> token << repeat(wschar), closepar).fmap(to_ast)
