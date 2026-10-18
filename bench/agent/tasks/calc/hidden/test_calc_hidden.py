import math
import unittest

from calc import CalcError, evaluate


class Numbers(unittest.TestCase):
    def test_int_stays_int(self):
        r = evaluate("2 + 3")
        self.assertEqual(r, 5)
        self.assertIsInstance(r, int)

    def test_decimals(self):
        self.assertEqual(evaluate(".5 + 2."), 2.5)
        self.assertEqual(evaluate("3.25"), 3.25)

    def test_true_division_is_float(self):
        r = evaluate("6 / 3")
        self.assertEqual(r, 2.0)
        self.assertIsInstance(r, float)


class Operators(unittest.TestCase):
    def test_precedence(self):
        self.assertEqual(evaluate("2 + 3 * 4 - 5"), 9)
        self.assertEqual(evaluate("10 - 4 - 3"), 3)
        self.assertEqual(evaluate("2 * 3 % 4"), 2)

    def test_floor_div_and_mod_follow_python(self):
        self.assertEqual(evaluate("-7 // 2"), -4)
        self.assertEqual(evaluate("-7 % 3"), 2)
        self.assertEqual(evaluate("7.5 // 2"), 3.0)

    def test_power_right_assoc(self):
        self.assertEqual(evaluate("2 ** 3 ** 2"), 512)

    def test_power_vs_unary(self):
        self.assertEqual(evaluate("-2 ** 2"), -4)
        self.assertEqual(evaluate("2 ** -1"), 0.5)
        self.assertEqual(evaluate("(-2) ** 2"), 4)

    def test_unary_chain(self):
        self.assertEqual(evaluate("--3"), 3)
        self.assertEqual(evaluate("-+-3"), 3)
        self.assertEqual(evaluate("2 * -3"), -6)

    def test_whitespace(self):
        self.assertEqual(evaluate(" \t1\t+ 2 "), 3)


class Names(unittest.TestCase):
    def test_constants(self):
        self.assertAlmostEqual(evaluate("pi"), math.pi)
        self.assertAlmostEqual(evaluate("e"), math.e)

    def test_override_constant(self):
        self.assertEqual(evaluate("pi + 1", {"pi": 3}), 4)

    def test_underscore_names(self):
        self.assertEqual(evaluate("_a1 * b_2", {"_a1": 2, "b_2": 5}), 10)


class Functions(unittest.TestCase):
    def test_basic_functions(self):
        self.assertEqual(evaluate("abs(-3)"), 3)
        self.assertEqual(evaluate("min(4, 2, 8)"), 2)
        self.assertEqual(evaluate("max(1)"), 1)
        self.assertEqual(evaluate("sqrt(16)"), 4.0)

    def test_round(self):
        self.assertEqual(evaluate("round(2.567, 2)"), 2.57)
        self.assertEqual(evaluate("round(2.5)"), 2)

    def test_nested_calls(self):
        self.assertEqual(evaluate("max(abs(-5), min(3, 9) * 2)"), 6)

    def test_expression_arguments(self):
        self.assertEqual(evaluate("min(1 + 1, 2 ** 3, x)", {"x": -1}), -1)


class Errors(unittest.TestCase):
    def assertError(self, expr, pos, variables=None):
        with self.assertRaises(CalcError) as cm:
            evaluate(expr, variables)
        self.assertEqual(cm.exception.pos, pos, expr)

    def test_unexpected_character(self):
        self.assertError("1 + $", 4)

    def test_unexpected_token(self):
        self.assertError("1 + * 2", 4)
        self.assertError("(1 + 2))", 7)

    def test_unexpected_end(self):
        self.assertError("(1 + 2", 6)
        self.assertError("", 0)
        self.assertError("max(1,", 6)

    def test_unknown_name(self):
        self.assertError("1 + foo", 4)

    def test_unknown_function(self):
        self.assertError("2 * bar(1)", 4)

    def test_wrong_arg_count(self):
        self.assertError("sqrt(1, 2)", 0)
        self.assertError("1 + min()", 4)
        self.assertError("round(1, 2, 3)", 0)

    def test_division_by_zero(self):
        self.assertError("1 / 0", 2)
        self.assertError("5 // (2 - 2)", 2)
        self.assertError("5 % 0", 2)

    def test_sqrt_negative(self):
        self.assertError("1 + sqrt(-4)", 4)

    def test_only_calcerror(self):
        for expr in ["1e5", "1..2", "abs", "abs(", ")", "2 ** ** 2", "a.b"]:
            with self.assertRaises(CalcError, msg=expr):
                evaluate(expr, {"a": 1})


if __name__ == "__main__":
    unittest.main()
