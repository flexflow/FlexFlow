from .ast import StringLiteral, NameLiteral, SExpr, AST
from .combinators import (
    repeat_until, 
    char, 
    inchars, 
    Parser, 
    join, 
    undo, 
    anychar, 
    one_or_more, 
    leaftree, 
    repeat, 
    LeafTree,
    eof,
    repeat_one_or_more_until
)
from typing import Union

escape_code: Parser[str] = (char('\\') >> inchars('"'))
string_piece: Parser[str] = join(repeat_until(undo(inchars(r'\"')), anychar))
string: Parser[str] = char('"') >> join(repeat_until(char('"'), escape_code | string_piece))
wschar: Parser[str] = inchars(' \n\t').named('wschar')
ws: Parser[str] = join(one_or_more(wschar))
openpar = char('(').named('openpar')
closepar = char(')').named('closepar')

name_literal: Parser[NameLiteral] = join(repeat_one_or_more_until(undo(wschar | openpar | closepar | eof), anychar)).fmap(NameLiteral).named('name_literal')
string_literal: Parser[StringLiteral] = string.fmap(StringLiteral).named('string_literal')

Token = Union[StringLiteral, NameLiteral]

token: Parser[Token] = (string_literal | name_literal).named('token')
def to_ast(t: Union[Token, LeafTree[Token]]) -> AST:
    if isinstance(t, (StringLiteral, NameLiteral)):
        return t
    else:
        return SExpr.from_iter(to_ast(c) for c in t.children)

child: Parser[Token] = repeat(wschar) >> token << repeat(wschar)
ast: Parser[AST] = leaftree(openpar, child, closepar).fmap(to_ast)

def parse(s: str) -> AST:
    return (ast << eof)(s)[0]
