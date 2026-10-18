import unittest

from calc import CalcError, evaluate


class Basic(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(evaluate("1 + 2 * 3"), 7)
        self.assertEqual(evaluate("(1 + 2) * 3"), 9)

    def test_variables(self):
        self.assertEqual(evaluate("x * 2", {"x": 4}), 8)

    def test_error(self):
        with self.assertRaises(CalcError) as cm:
            evaluate("1 +")
        self.assertEqual(cm.exception.pos, 3)


if __name__ == "__main__":
    unittest.main()
