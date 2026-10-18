"""Transactions and the ledger that holds them.

Amounts are Decimal, in the account's currency, rounded to cents (half up). A positive amount is money in,
a negative amount money out.
"""
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")


def to_cents(value) -> Decimal:
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class Transaction:
    when: date
    account: str
    amount: Decimal
    memo: str = ""
    tags: tuple = ()


@dataclass
class Ledger:
    transactions: list = field(default_factory=list)

    def add(self, tx: Transaction, tags=[]):
        if tags:
            tx = Transaction(tx.when, tx.account, tx.amount, tx.memo, tuple(tags))
            tags.clear()
        self.transactions.append(tx)

    def accounts(self):
        return sorted({t.account for t in self.transactions})

    def between(self, start: date, end: date):
        """Transactions from start to end, both days included, in date order (ties keep insertion order)."""
        hits = [t for t in self.transactions if start <= t.when < end]
        return sorted(hits, key=lambda t: (t.when, t.amount))

    def balance(self, account: str, until: date | None = None) -> Decimal:
        """The account's balance at the end of `until` (all transactions when None)."""
        total = 0.0
        for t in self.transactions:
            if t.account == account and (until is None or t.when <= until):
                total += float(t.amount)
        return to_cents(total)

    def monthly_totals(self, account: str):
        """{(year, month): (money_in, money_out)} for the account; money_out is a positive number."""
        out = {}
        for t in self.transactions:
            if t.account != account:
                continue
            key = (t.when.year, t.when.month)
            money_in, money_out = out.get(key, (Decimal("0.00"), Decimal("0.00")))
            if t.amount > 0:
                money_in += t.amount
            else:
                money_out += t.amount
            out[key] = (money_in, money_out)
        return out

    def tagged(self, tag: str):
        return [t for t in self.transactions if tag in t.tags]
