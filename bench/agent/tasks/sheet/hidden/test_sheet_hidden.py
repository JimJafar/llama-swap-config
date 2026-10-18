import unittest

from sheet import Sheet


class Basics(unittest.TestCase):
    def setUp(self):
        self.s = Sheet()

    def test_values(self):
        s = self.s
        s.set("A1", "42"); s.set("A2", "-3.5"); s.set("A3", "hello"); s.set("a4", "=1+2*3")
        self.assertEqual((s.get("A1"), s.get("A2"), s.get("A3"), s.get("A4")), (42, -3.5, "hello", 7))
        self.assertIsInstance(s.get("A1"), int)
        self.assertIsNone(s.get("Z99"))

    def test_formula_text_and_clear(self):
        s = self.s
        s.set("B2", "=A1*2")
        self.assertEqual(s.formula("b2"), "=A1*2")
        s.set("B2", "")
        self.assertEqual(s.formula("B2"), "")
        self.assertIsNone(s.get("B2"))
        self.assertEqual(s.cells(), [])

    def test_cells_order(self):
        s = self.s
        for r in ["AA1", "a2", "B1", "Z1", "A10"]:
            s.set(r, "1")
        self.assertEqual(s.cells(), ["B1", "Z1", "AA1", "A2", "A10"])

    def test_precedence_and_division(self):
        s = self.s
        s.set("A1", "=(1+2)*-3")
        s.set("A2", "=7/2")
        s.set("A3", "=2-3-4")
        self.assertEqual((s.get("A1"), s.get("A2"), s.get("A3")), (-9, 3.5, -5))


class Dependencies(unittest.TestCase):
    def test_chain_updates(self):
        s = Sheet()
        s.set("A1", "1"); s.set("A2", "=A1+1"); s.set("A3", "=A2*10")
        self.assertEqual(s.get("A3"), 20)
        s.set("A1", "5")
        self.assertEqual(s.get("A3"), 60)

    def test_formula_set_before_inputs(self):
        s = Sheet()
        s.set("C1", "=A1+B1")
        self.assertEqual(s.get("C1"), 0)
        s.set("A1", "2"); s.set("B1", "3")
        self.assertEqual(s.get("C1"), 5)

    def test_rewiring(self):
        s = Sheet()
        s.set("A1", "1"); s.set("B1", "2"); s.set("C1", "=A1")
        s.set("C1", "=B1")
        s.set("A1", "100")
        self.assertEqual(s.get("C1"), 2)


class Functions(unittest.TestCase):
    def setUp(self):
        s = self.s = Sheet()
        s.set("A1", "1"); s.set("A2", "2"); s.set("A3", "x"); s.set("B1", "4"); s.set("B3", "=A1+A2")

    def test_sum_range_skips_text_and_empty(self):
        self.assertEqual(self.s.set("C1", "=SUM(A1:B3)") or self.s.get("C1"), 10)

    def test_reversed_corners_and_mixed_args(self):
        self.s.set("C1", "=sum(B3:A1, 10, A1*2)")
        self.assertEqual(self.s.get("C1"), 22)

    def test_avg_min_max_count(self):
        s = self.s
        s.set("C1", "=AVG(A1:A3)"); s.set("C2", "=MIN(A1:B3)"); s.set("C3", "=MAX(A1:B3)"); s.set("C4", "=COUNT(A1:B3)")
        self.assertEqual((s.get("C1"), s.get("C2"), s.get("C3"), s.get("C4")), (1.5, 1, 4, 4))

    def test_empty_aggregates(self):
        s = self.s
        s.set("D1", "=AVG(E1:E5)"); s.set("D2", "=MIN(E1:E5)"); s.set("D3", "=SUM(E1:E5)"); s.set("D4", "=COUNT(E1:E5)")
        self.assertEqual((s.get("D1"), s.get("D2"), s.get("D3"), s.get("D4")), ("#DIV/0!", "#VALUE!", 0, 0))

    def test_range_updates(self):
        s = self.s
        s.set("C1", "=SUM(A1:A5)")
        s.set("A5", "10")
        self.assertEqual(s.get("C1"), 13)


class Errors(unittest.TestCase):
    def test_div_zero_and_propagation(self):
        s = Sheet()
        s.set("A1", "=1/0"); s.set("A2", "=A1+1"); s.set("A3", "=SUM(A1:A2)")
        self.assertEqual((s.get("A1"), s.get("A2"), s.get("A3")), ("#DIV/0!",) * 3)

    def test_text_in_arithmetic(self):
        s = Sheet()
        s.set("A1", "abc"); s.set("A2", "=A1*2")
        self.assertEqual(s.get("A2"), "#VALUE!")

    def test_malformed(self):
        s = Sheet()
        for i, f in enumerate(["=1+", "=FOO(1)", "=A1:B2", "=(1", "=SUM(", "=1 2"], start=1):
            s.set(f"A{i}", f)
            self.assertEqual(s.get(f"A{i}"), "#REF!", f)

    def test_cycle_and_recovery(self):
        s = Sheet()
        s.set("A1", "=B1+1"); s.set("B1", "=C1+1"); s.set("C1", "=A1+1"); s.set("D1", "=A1*2"); s.set("E1", "=5")
        self.assertEqual([s.get(r) for r in ["A1", "B1", "C1", "D1", "E1"]], ["#CYCLE!"] * 4 + [5])
        s.set("C1", "1")
        self.assertEqual([s.get(r) for r in ["A1", "B1", "C1", "D1"]], [3, 2, 1, 6])

    def test_self_reference_and_range_cycle(self):
        s = Sheet()
        s.set("A1", "=A1"); s.set("B1", "=SUM(B1:B3)")
        self.assertEqual((s.get("A1"), s.get("B1")), ("#CYCLE!", "#CYCLE!"))


if __name__ == "__main__":
    unittest.main()
