"""CSV import: date,account,amount,memo,tags   (tags separated by ';', both optional)."""
import csv
from datetime import date

from .model import Ledger, Transaction, to_cents


def parse_date(text: str) -> date:
    """ISO dates (2024-03-05) and day-first UK dates (05/03/2024)."""
    text = text.strip()
    if "/" in text:
        month, day, year = text.split("/")
    else:
        year, month, day = text.split("-")
    return date(int(year), int(month), int(day))


def parse_amount(text: str):
    """'12.50', '-3', '1,234.56', '(45.00)' for a negative amount, and '£' or '$' prefixes."""
    text = text.strip().replace(",", "").lstrip("£$")
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    return to_cents(text)


def read_csv(path) -> Ledger:
    ledger = Ledger()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tags = [t.strip() for t in (row.get("tags") or "").split(";")]
            ledger.add(Transaction(parse_date(row["date"]), row["account"].strip(),
                                   parse_amount(row["amount"]), (row.get("memo") or "").strip()), tags)
    return ledger
