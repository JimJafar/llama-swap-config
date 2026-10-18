import pathlib
import unittest
from datetime import date
from decimal import Decimal

from ledger import Ledger, Transaction, read_csv
from ledger.io import parse_amount, parse_date

SAMPLE = pathlib.Path(__file__).with_name("sample.csv")


class Parsing(unittest.TestCase):
    def test_uk_date_is_day_first(self):
        self.assertEqual(parse_date("05/01/2024"), date(2024, 1, 5))

    def test_amounts(self):
        self.assertEqual(parse_amount("1,200.00"), Decimal("1200.00"))
        self.assertEqual(parse_amount("(19.99)"), Decimal("-19.99"))
        self.assertEqual(parse_amount("£-12.345"), Decimal("-12.35"))


class Reports(unittest.TestCase):
    def setUp(self):
        self.ledger = read_csv(SAMPLE)

    def test_between_includes_end_day(self):
        got = self.ledger.between(date(2024, 1, 1), date(2024, 1, 31))
        self.assertEqual([t.memo for t in got], ["salary", "groceries", "transfer", "phone"])

    def test_balance(self):
        self.assertEqual(self.ledger.balance("current"), Decimal("1122.76"))

    def test_monthly_totals(self):
        m = self.ledger.monthly_totals("current")
        self.assertEqual(m[(2024, 1)], (Decimal("1200.00"), Decimal("65.09")))

    def test_untagged_rows_have_no_tags(self):
        fee = [t for t in self.ledger.transactions if t.memo == "fee"][0]
        self.assertEqual(fee.tags, ())

    def test_tagged(self):
        self.assertEqual([t.memo for t in self.ledger.tagged("food")], ["groceries", "coffee"])


if __name__ == "__main__":
    unittest.main()
