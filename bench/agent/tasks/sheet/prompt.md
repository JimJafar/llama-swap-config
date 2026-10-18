Implement `sheet.py` in the current directory: an in-memory spreadsheet engine.

API:

- `class Sheet` with:
  - `set(ref: str, text: str)`: set a cell's contents. `ref` is a cell reference like `A1`, `b12` or `AA3` (column letters, case-insensitive, then a row number from 1). `text` is either a number (`"42"`, `"-3.5"`), a formula starting with `=`, or anything else, which is a text value. An empty string clears the cell.
  - `get(ref: str)`: the cell's current value: an `int` or `float` for numbers, a `str` for text, `None` for an empty cell, or an error string (below).
  - `formula(ref: str) -> str`: the contents as set (`""` for an empty cell).
  - `cells() -> list[str]`: the references of all non-empty cells, normalised to upper case, sorted by row and then by column (`B1` before `A2`, `Z1` before `AA1`).
- Values update immediately: after `set`, every cell that depends on the changed cell, directly or indirectly, shows its new value.

Formulas (after the `=`):

- Numbers, cell references, `+ - * /`, unary minus and parentheses, with the usual precedence. `/` gives a float.
- Ranges like `A1:B3` (any two corners), only as function arguments.
- Functions `SUM`, `AVG`, `MIN`, `MAX`, `COUNT` taking any mix of ranges, references and expressions. `SUM`, `MIN`, `MAX` and `AVG` use only numeric cells in ranges and skip empty and text cells; `COUNT` counts the numeric cells. `AVG`, `MIN` and `MAX` of no numbers is `#DIV/0!` for `AVG` and `#VALUE!` for `MIN`/`MAX`. Function names are case-insensitive.
- An empty cell used directly in arithmetic counts as 0. A text cell used directly in arithmetic gives `#VALUE!`.

Errors are values, shown as these strings:

- `#DIV/0!` for division by zero
- `#VALUE!` for text in arithmetic, as above
- `#REF!` for a malformed formula or an unknown function
- `#CYCLE!` for every cell on a circular reference (a cell that depends on itself, directly or through other cells); cells that only depend on a cycle show `#CYCLE!` too
- An error in a cell a formula uses (directly, or inside a range) makes the formula's value that same error.

When a cycle is broken by changing one of its cells, the other cells recover their normal values.

Run your own tests as you go. The implementation will be checked against a hidden test suite that follows these rules.
