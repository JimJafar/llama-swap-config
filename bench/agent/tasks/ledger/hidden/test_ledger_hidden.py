import unittest
from datetime import date
from decimal import Decimal

from ledger import Ledger, Transaction
from ledger.io import parse_amount, parse_date


def tx(d, acc, amt, memo=""):
    return Transaction(d, acc, Decimal(amt), memo)


class Hidden(unittest.TestCase):
    def test_dates(self):
        self.assertEqual(parse_date(" 31/12/2023 "), date(2023, 12, 31))
        self.assertEqual(parse_date("2023-12-31"), date(2023, 12, 31))

    def test_amount_forms(self):
        self.assertEqual(parse_amount("$1,000"), Decimal("1000.00"))
        self.assertEqual(parse_amount("(0.005)"), Decimal("-0.01"))
        self.assertEqual(parse_amount("2.675"), Decimal("2.68"))

    def test_between_ties_keep_insertion_order(self):
        l = Ledger()
        for memo, amt in [("b", "5"), ("a", "-5"), ("c", "1")]:
            l.add(tx(date(2024, 3, 1), "x", amt, memo))
        self.assertEqual([t.memo for t in l.between(date(2024, 3, 1), date(2024, 3, 1))], ["b", "a", "c"])

    def test_balance_exact_with_many_small_amounts(self):
        l = Ledger()
        for _ in range(1000):
            l.add(tx(date(2024, 1, 1), "x", "0.10"))
        l.add(tx(date(2024, 1, 2), "x", "0.005"))
        self.assertEqual(l.balance("x"), Decimal("100.01"))
        self.assertEqual(l.balance("x", until=date(2024, 1, 1)), Decimal("100.00"))

    def test_balance_is_decimal(self):
        l = Ledger()
        l.add(tx(date(2024, 1, 1), "x", "1.10"))
        self.assertIsInstance(l.balance("x"), Decimal)

    def test_tags_do_not_leak_between_adds(self):
        l = Ledger()
        tags = ["a"]
        l.add(tx(date(2024, 1, 1), "x", "1"), tags)
        l.add(tx(date(2024, 1, 2), "x", "1"))
        l.add(tx(date(2024, 1, 3), "x", "1"))
        self.assertEqual([t.tags for t in l.transactions], [("a",), (), ()])

    def test_empty_tag_string(self):
        import io, tempfile, os
        from ledger import read_csv
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("date,account,amount,memo,tags\n2024-01-01,x,1,a,\n2024-01-02,x,1,b, food ; ;bills\n")
        try:
            l = read_csv(f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual([t.tags for t in l.transactions], [(), ("food", "bills")])

    def test_monthly_out_positive_and_zero_amounts_ignored(self):
        l = Ledger()
        l.add(tx(date(2024, 5, 1), "x", "-3.00"))
        l.add(tx(date(2024, 5, 2), "x", "0"))
        l.add(tx(date(2024, 5, 3), "x", "4.00"))
        self.assertEqual(l.monthly_totals("x")[(2024, 5)], (Decimal("4.00"), Decimal("3.00")))


if __name__ == "__main__":
    unittest.main()
