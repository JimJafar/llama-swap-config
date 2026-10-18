Implement `calc.py` in the current directory: a small, safe expression evaluator (no `eval`, no `ast`).

API:

- `evaluate(expr: str, variables: dict | None = None) -> int | float`
- `class CalcError(Exception)` with an attribute `pos`: the 0-based index in `expr` where the problem was found.

The language:

- Numbers: integers (`42`) and decimals (`3.5`, `.5`, `2.`). Integers stay `int` unless an operation produces a float.
- Operators, from lowest to highest precedence:
  - `+` `-` (left-associative)
  - `*` `/` `//` `%` (left-associative); `/` always gives a float, `//` and `%` follow Python
  - unary `-` and `+`
  - `**` (right-associative, binds tighter than unary minus on its left: `-2 ** 2` is `-4`, and `2 ** -1` is `0.5`)
- Parentheses.
- Names: letters, digits and `_`, not starting with a digit. A name is looked up in `variables`; the constants `pi` and `e` exist unless `variables` overrides them.
- Function calls: `abs(x)`, `min(a, b, ...)`, `max(a, b, ...)` (at least one argument), `sqrt(x)`, `round(x)` and `round(x, n)`.
- Spaces and tabs anywhere between tokens.

Errors raise `CalcError` (never another exception type), with `pos` at the offending character or token:

- an unexpected character, an unexpected token, or an unexpected end (pos = `len(expr)`)
- an unknown name or an unknown function (pos at the name)
- a wrong number of arguments (pos at the function name)
- division or modulo by zero (pos at the operator)
- `sqrt` of a negative number (pos at `sqrt`)

There is a small test file, `test_calc_basic.py`, you can run with `python3 -m unittest -v`. It is not complete: the implementation will be checked against a larger hidden test suite that follows the rules above.
