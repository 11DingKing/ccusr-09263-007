"""预约押金分录：收取/退还/抵扣金额守恒，重复退款不产生第二笔资金变动。"""
from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from service_09252_008.application.catalog_service import CatalogService
from service_09252_008.application.booking_service import BookingService
from service_09252_008.application.deposit_service import COLLECTION_DEPOSITS, DepositService
from service_09252_008.application.ports import ManualClock, UuidIdGenerator
from service_09252_008.domain.errors import NotFoundError, StateError, ValidationError
from service_09252_008.domain.models import DepositEntry, DepositEntryType, DepositLedger
from service_09252_008.persistence.sqlite_store import SQLiteStore
from tests.helpers import NOW, apply_payload, make_services, seed_catalog

DEPOSIT_AMOUNT = 10000  # 100.00 元（单位：分）


def _entry_types(view: dict) -> list[str]:
    return [e["type"] for e in view["entries"]]


def _amounts_by_type(view: dict) -> dict[str, int]:
    totals: dict[str, int] = {}
    for entry in view["entries"]:
        totals[entry["type"]] = totals.get(entry["type"], 0) + entry["amount_cents"]
    return totals


class DepositEntryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog, self.bookings, self.deposits, self.clock, self.store = make_services()
        self.ids = seed_catalog(self.catalog)
        applied = self.bookings.apply(apply_payload(self.ids, "k-dep-apply"))
        self.booking_id = applied["booking_id"]

    def _collect(self, amount: int = DEPOSIT_AMOUNT, key: str = "k-dep-collect") -> dict:
        return self.deposits.collect(
            self.booking_id, {"idempotency_key": key, "amount_cents": amount, "currency": "CNY"}
        )

    # ------------------------------------------------------------------
    # 守恒
    # ------------------------------------------------------------------

    def test_collect_writes_single_entry_and_returns_balance(self) -> None:
        view = self._collect()
        self.assertEqual(_entry_types(view), ["COLLECT"])
        self.assertEqual(view["collected_cents"], DEPOSIT_AMOUNT)
        self.assertEqual(view["refunded_cents"], 0)
        self.assertEqual(view["applied_cents"], 0)
        # Python 返回守恒后的余额
        self.assertEqual(view["balance_cents"], DEPOSIT_AMOUNT)
        self.assertEqual(view["state"], "HELD")
        self.assertEqual(len(self.store.query(COLLECTION_DEPOSITS)), 1)

    def test_full_refund_conserves(self) -> None:
        self._collect()
        view = self.deposits.refund(self.booking_id, {"idempotency_key": "k-dep-refund"})
        self.assertEqual(_entry_types(view), ["COLLECT", "REFUND"])
        self.assertEqual(view["refunded_cents"], DEPOSIT_AMOUNT)
        self.assertEqual(view["balance_cents"], 0)
        self.assertEqual(view["state"], "REFUNDED")
        totals = _amounts_by_type(view)
        # 金额合计守恒：收取 == 退还 + 抵扣
        self.assertEqual(totals["COLLECT"], totals.get("REFUND", 0) + totals.get("APPLY", 0))

    def test_partial_apply_then_refund_remainder_conserves(self) -> None:
        self._collect()
        applied = self.deposits.apply_deposit(
            self.booking_id, {"idempotency_key": "k-dep-apply-4000", "amount_cents": 4000}
        )
        self.assertEqual(applied["applied_cents"], 4000)
        self.assertEqual(applied["balance_cents"], 6000)
        self.assertEqual(applied["state"], "HELD")

        refunded = self.deposits.refund(self.booking_id, {"idempotency_key": "k-dep-refund-rest"})
        self.assertEqual(_entry_types(refunded), ["COLLECT", "APPLY", "REFUND"])
        self.assertEqual(refunded["refunded_cents"], 6000)
        self.assertEqual(refunded["balance_cents"], 0)
        self.assertEqual(refunded["state"], "PARTIALLY_APPLIED")
        totals = _amounts_by_type(refunded)
        self.assertEqual(totals["COLLECT"], totals["REFUND"] + totals["APPLY"])

    def test_full_apply_state_is_applied(self) -> None:
        self._collect()
        view = self.deposits.apply_deposit(self.booking_id, {"idempotency_key": "k-dep-apply-full"})
        self.assertEqual(_entry_types(view), ["COLLECT", "APPLY"])
        self.assertEqual(view["balance_cents"], 0)
        self.assertEqual(view["state"], "APPLIED")

    def test_refund_over_balance_is_rejected_and_rolls_back(self) -> None:
        self._collect()
        self.deposits.refund(self.booking_id, {"idempotency_key": "k-dep-refund-6000", "amount_cents": 6000})
        with self.assertRaises(ValidationError):
            self.deposits.refund(self.booking_id, {"idempotency_key": "k-dep-refund-over", "amount_cents": 6000})
        view = self.deposits.get_deposit(self.booking_id)
        # 超额退款未落任何分录，余额守恒仍为 4000
        self.assertEqual(_entry_types(view), ["COLLECT", "REFUND"])
        self.assertEqual(view["refunded_cents"], 6000)
        self.assertEqual(view["balance_cents"], 4000)

    def test_apply_over_balance_is_rejected(self) -> None:
        self._collect()
        with self.assertRaises(ValidationError):
            self.deposits.apply_deposit(
                self.booking_id, {"idempotency_key": "k-dep-apply-over", "amount_cents": DEPOSIT_AMOUNT + 1}
            )
        view = self.deposits.get_deposit(self.booking_id)
        self.assertEqual(_entry_types(view), ["COLLECT"])
        self.assertEqual(view["balance_cents"], DEPOSIT_AMOUNT)

    # ------------------------------------------------------------------
    # 重复退款：保持一笔记录
    # ------------------------------------------------------------------

    def test_duplicate_refund_same_key_replays_with_one_entry(self) -> None:
        self._collect()
        payload = {"idempotency_key": "k-dep-refund-dup"}
        first = self.deposits.refund(self.booking_id, payload)
        replay = self.deposits.refund(self.booking_id, payload)
        self.assertTrue(replay["idempotent_replay"])
        self.assertNotIn("idempotent_replay", first)
        view = self.deposits.get_deposit(self.booking_id)
        refund_entries = [e for e in view["entries"] if e["type"] == "REFUND"]
        self.assertEqual(len(refund_entries), 1)
        self.assertEqual(view["balance_cents"], 0)

    def test_duplicate_refund_new_key_is_blocked_by_state_guard(self) -> None:
        self._collect()
        self.deposits.refund(self.booking_id, {"idempotency_key": "k-dep-refund-first"})
        # 换新键重试第二笔退款：在押余额已结清，状态守卫拒绝
        with self.assertRaises(StateError):
            self.deposits.refund(self.booking_id, {"idempotency_key": "k-dep-refund-second"})
        view = self.deposits.get_deposit(self.booking_id)
        self.assertEqual(len([e for e in view["entries"] if e["type"] == "REFUND"]), 1)
        self.assertEqual(view["balance_cents"], 0)

    def test_duplicate_apply_after_settlement_is_blocked(self) -> None:
        self._collect()
        self.deposits.apply_deposit(self.booking_id, {"idempotency_key": "k-apply-first"})
        with self.assertRaises(StateError):
            self.deposits.apply_deposit(self.booking_id, {"idempotency_key": "k-apply-second"})
        view = self.deposits.get_deposit(self.booking_id)
        self.assertEqual(len([e for e in view["entries"] if e["type"] == "APPLY"]), 1)

    # ------------------------------------------------------------------
    # 输入与状态校验
    # ------------------------------------------------------------------

    def test_collect_requires_idempotency_key(self) -> None:
        with self.assertRaises(ValidationError):
            self.deposits.collect(self.booking_id, {"amount_cents": 1000})

    def test_refund_requires_idempotency_key(self) -> None:
        self._collect()
        with self.assertRaises(ValidationError):
            self.deposits.refund(self.booking_id, {})

    def test_collect_rejects_non_positive_amount(self) -> None:
        with self.assertRaises(ValidationError):
            self.deposits.collect(self.booking_id, {"idempotency_key": "k-bad", "amount_cents": 0})

    def test_collect_twice_is_rejected(self) -> None:
        self._collect()
        with self.assertRaises(StateError):
            self.deposits.collect(
                self.booking_id, {"idempotency_key": "k-dep-collect-again", "amount_cents": DEPOSIT_AMOUNT}
            )
        view = self.deposits.get_deposit(self.booking_id)
        self.assertEqual(len(view["entries"]), 1)

    def test_refund_without_ledger_is_not_found(self) -> None:
        with self.assertRaises(NotFoundError):
            self.deposits.refund(self.booking_id, {"idempotency_key": "k-no-ledger"})

    def test_collect_for_unknown_booking_is_not_found(self) -> None:
        with self.assertRaises(NotFoundError):
            self.deposits.collect("bkg_missing", {"idempotency_key": "k-x", "amount_cents": 100})


class DepositLedgerDomainTests(unittest.TestCase):
    def test_ledger_rejects_non_conserving_entries(self) -> None:
        from datetime import datetime, timezone

        now = datetime(2026, 9, 25, tzinfo=timezone.utc)
        ledger = DepositLedger(deposit_id="dep_1", booking_id="bkg_1", currency="CNY", created_at=now, updated_at=now)
        ledger.add(DepositEntry("ent_1", "dep_1", "bkg_1", DepositEntryType.COLLECT, 1000, now))
        ledger.add(DepositEntry("ent_2", "dep_1", "bkg_1", DepositEntryType.REFUND, 600, now))
        with self.assertRaises(ValueError):
            ledger.add(DepositEntry("ent_3", "dep_1", "bkg_1", DepositEntryType.APPLY, 500, now))
        # 违反守恒的分录未留在台账内
        self.assertEqual(len(ledger.entries), 2)
        self.assertEqual(ledger.balance_cents, 400)


class DepositSQLiteTests(unittest.TestCase):
    """SQLite 事务后端：重复退款保持一笔记录，分录重启后仍守恒。"""

    def _build(self, tmp: str):
        store = SQLiteStore(f"{tmp}/booking.db")
        clock = ManualClock(NOW)
        ids = UuidIdGenerator()
        catalog = CatalogService(store, clock, ids)
        bookings = BookingService(store, clock, ids)
        deposits = DepositService(store, clock, ids)
        return store, catalog, bookings, deposits

    def _seed_booking_with_deposit(self, catalog, bookings, deposits) -> str:
        ids = seed_catalog(catalog)
        booking_id = bookings.apply(apply_payload(ids, "k-sql-apply"))["booking_id"]
        deposits.collect(booking_id, {"idempotency_key": "k-sql-collect", "amount_cents": DEPOSIT_AMOUNT})
        return booking_id

    def test_entries_persist_and_stay_conserved_after_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, catalog, bookings, deposits = self._build(tmp)
            booking_id = self._seed_booking_with_deposit(catalog, bookings, deposits)
            deposits.refund(booking_id, {"idempotency_key": "k-sql-refund", "amount_cents": 4000})
            deposits.apply_deposit(booking_id, {"idempotency_key": "k-sql-apply-dep", "amount_cents": 6000})
            store.close()

            store2, _, _, deposits2 = self._build(tmp)
            view = deposits2.get_deposit(booking_id)
            self.assertEqual(view["collected_cents"], DEPOSIT_AMOUNT)
            self.assertEqual(view["refunded_cents"], 4000)
            self.assertEqual(view["applied_cents"], 6000)
            self.assertEqual(view["balance_cents"], 0)
            self.assertEqual(len(view["entries"]), 3)
            store2.close()

    def test_concurrent_duplicate_refunds_keep_one_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, catalog, bookings, deposits = self._build(tmp)
            booking_id = self._seed_booking_with_deposit(catalog, bookings, deposits)

            def refund(_: int) -> str:
                try:
                    deposits.refund(booking_id, {"idempotency_key": "k-sql-refund-dup"})
                    return "ok"
                except StateError:
                    return "rejected"

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(refund, range(8)))

            self.assertEqual(results.count("ok"), 8)  # 全部拿到首次结果（重放）
            records = store.query(COLLECTION_DEPOSITS)
            self.assertEqual(len(records), 1)
            refund_entries = [e for e in records[0]["entries"] if e["type"] == "REFUND"]
            self.assertEqual(len(refund_entries), 1)  # 只有一笔资金变动
            self.assertEqual(refund_entries[0]["amount_cents"], DEPOSIT_AMOUNT)
            ledger = DepositLedger.from_dict(records[0])
            self.assertEqual(ledger.balance_cents, 0)
            store.close()

    def test_concurrent_refunds_with_distinct_keys_keep_one_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, catalog, bookings, deposits = self._build(tmp)
            booking_id = self._seed_booking_with_deposit(catalog, bookings, deposits)

            def refund(index: int) -> str:
                try:
                    deposits.refund(booking_id, {"idempotency_key": f"k-sql-refund-{index}"})
                    return "ok"
                except StateError:
                    return "rejected"

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(refund, range(8)))

            self.assertEqual(results.count("ok"), 1)  # 只有一笔退款成功
            self.assertEqual(results.count("rejected"), 7)
            records = store.query(COLLECTION_DEPOSITS)
            refund_entries = [e for e in records[0]["entries"] if e["type"] == "REFUND"]
            self.assertEqual(len(refund_entries), 1)
            self.assertEqual(DepositLedger.from_dict(records[0]).balance_cents, 0)
            store.close()


if __name__ == "__main__":
    unittest.main()
