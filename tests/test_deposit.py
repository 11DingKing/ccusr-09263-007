"""预约押金分录：收取/退还/抵扣的金额守恒与重复退款幂等。"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from service_09252_008.application.booking_service import BookingService
from service_09252_008.application.catalog_service import CatalogService
from service_09252_008.application.deposit_service import (
    COLLECTION_DEPOSIT_ENTRIES,
    DepositService,
)
from service_09252_008.application.ports import ManualClock, UuidIdGenerator
from service_09252_008.domain.errors import BusinessRuleError, NotFoundError, StateError, ValidationError
from service_09252_008.interfaces.http_api import create_server
from service_09252_008.persistence.sqlite_store import SQLiteStore
from tests.helpers import NOW, apply_payload, make_services, seed_catalog

DEPOSIT_CENTS = 10000


def _assert_conserved(testcase: unittest.TestCase, view: dict) -> None:
    """金额合计守恒：收取 == 退还 + 抵扣 + 在押；现金 == 收取 - 退还。"""
    testcase.assertEqual(
        view["collected_cents"],
        view["refunded_cents"] + view["deducted_cents"] + view["held_cents"],
    )
    testcase.assertEqual(
        view["cash_cents"], view["collected_cents"] - view["refunded_cents"]
    )
    testcase.assertGreaterEqual(view["held_cents"], 0)
    testcase.assertGreaterEqual(view["cash_cents"], 0)


class DepositTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog, self.bookings, self.deposits, self.clock, self.store = make_services()
        self.ids = seed_catalog(self.catalog)
        applied = self.bookings.apply(apply_payload(self.ids, "k-deposit-apply"))
        self.booking_id = applied["booking_id"]

    def test_collect_posts_entry_and_returns_conserved_balance(self) -> None:
        view = self.deposits.collect(
            self.booking_id, {"idempotency_key": "k-dep-collect", "amount_cents": DEPOSIT_CENTS}
        )
        self.assertEqual(view["state"], "OPEN")
        self.assertEqual(view["collected_cents"], DEPOSIT_CENTS)
        self.assertEqual(view["held_cents"], DEPOSIT_CENTS)
        self.assertEqual(view["cash_cents"], DEPOSIT_CENTS)
        self.assertEqual(view["refunded_cents"], 0)
        self.assertEqual(view["deducted_cents"], 0)
        _assert_conserved(self, view)
        # 只有一条收取分录
        entries = self.store.query(COLLECTION_DEPOSIT_ENTRIES, booking_id=self.booking_id)
        self.assertEqual([e["type"] for e in entries], ["COLLECT"])

    def test_collect_twice_is_rejected(self) -> None:
        self.deposits.collect(
            self.booking_id, {"idempotency_key": "k-dep-collect", "amount_cents": DEPOSIT_CENTS}
        )
        with self.assertRaises(BusinessRuleError):
            self.deposits.collect(
                self.booking_id, {"idempotency_key": "k-dep-collect-2", "amount_cents": 3000}
            )
        # 没有产生第二笔资金变动
        self.assertEqual(len(self.store.query(COLLECTION_DEPOSIT_ENTRIES)), 1)

    def test_refund_before_collect_is_invalid_state(self) -> None:
        with self.assertRaises(StateError):
            self.deposits.refund(self.booking_id, {"idempotency_key": "k-r-early"})

    def test_partial_then_full_refund_keeps_conservation(self) -> None:
        self.deposits.collect(
            self.booking_id, {"idempotency_key": "k-dep-collect", "amount_cents": DEPOSIT_CENTS}
        )
        first = self.deposits.refund(
            self.booking_id, {"idempotency_key": "k-r1", "amount_cents": 4000}
        )
        self.assertEqual(first["refunded_cents"], 4000)
        self.assertEqual(first["cash_cents"], 6000)
        self.assertEqual(first["held_cents"], 6000)
        _assert_conserved(self, first)

        # 缺省金额 = 退还全部在押余额
        second = self.deposits.refund(self.booking_id, {"idempotency_key": "k-r2"})
        self.assertEqual(second["refunded_cents"], DEPOSIT_CENTS)
        self.assertEqual(second["cash_cents"], 0)
        self.assertEqual(second["held_cents"], 0)
        self.assertEqual(second["state"], "REFUNDED")
        _assert_conserved(self, second)

    def test_refund_cannot_exceed_held_deposit(self) -> None:
        self.deposits.collect(
            self.booking_id, {"idempotency_key": "k-dep-collect", "amount_cents": DEPOSIT_CENTS}
        )
        with self.assertRaises(BusinessRuleError):
            self.deposits.refund(
                self.booking_id, {"idempotency_key": "k-r-over", "amount_cents": DEPOSIT_CENTS + 1}
            )

    def test_duplicate_refund_replay_keeps_single_entry(self) -> None:
        self.deposits.collect(
            self.booking_id, {"idempotency_key": "k-dep-collect", "amount_cents": DEPOSIT_CENTS}
        )
        first = self.deposits.refund(
            self.booking_id, {"idempotency_key": "k-refund", "amount_cents": 4000}
        )
        replay = self.deposits.refund(
            self.booking_id, {"idempotency_key": "k-refund", "amount_cents": 4000}
        )
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(replay["last_entry"]["entry_id"], first["last_entry"]["entry_id"])
        self.assertEqual(replay["refunded_cents"], 4000)
        self.assertEqual(replay["cash_cents"], 6000)
        _assert_conserved(self, replay)
        # 重复退款没有产生第二笔资金变动：仍然只有收取 + 一笔退还
        entries = self.store.query(COLLECTION_DEPOSIT_ENTRIES, booking_id=self.booking_id)
        self.assertEqual([e["type"] for e in entries], ["COLLECT", "REFUND"])
        self.assertEqual(len(entries), 2)

    def test_second_refund_after_full_settlement_without_key_is_rejected(self) -> None:
        self.deposits.collect(
            self.booking_id, {"idempotency_key": "k-dep-collect", "amount_cents": DEPOSIT_CENTS}
        )
        self.deposits.refund(self.booking_id, {"idempotency_key": "k-refund"})
        # 无键的再次退款（重复请求）不能在在押为 0 时再产生资金变动
        with self.assertRaises(BusinessRuleError):
            self.deposits.refund(self.booking_id, {})
        self.assertEqual(len(self.store.query(COLLECTION_DEPOSIT_ENTRIES)), 2)

    def test_deduct_does_not_move_cash_and_conserves(self) -> None:
        self.deposits.collect(
            self.booking_id, {"idempotency_key": "k-dep-collect", "amount_cents": DEPOSIT_CENTS}
        )
        view = self.deposits.deduct(
            self.booking_id, {"idempotency_key": "k-deduct", "amount_cents": 3500, "reason": "课损抵扣"}
        )
        self.assertEqual(view["deducted_cents"], 3500)
        self.assertEqual(view["held_cents"], 6500)
        # 抵扣只是押金负债转收入，现金不变
        self.assertEqual(view["cash_cents"], DEPOSIT_CENTS)
        _assert_conserved(self, view)

    def test_refund_and_deduct_split_conserves_to_collected(self) -> None:
        self.deposits.collect(
            self.booking_id, {"idempotency_key": "k-dep-collect", "amount_cents": DEPOSIT_CENTS}
        )
        self.deposits.deduct(self.booking_id, {"idempotency_key": "k-d1", "amount_cents": 3000})
        self.deposits.refund(self.booking_id, {"idempotency_key": "k-r1", "amount_cents": 7000})
        view = self.deposits.get_deposit(self.booking_id)
        self.assertEqual(view["state"], "SETTLED")
        self.assertEqual(view["held_cents"], 0)
        self.assertEqual(view["cash_cents"], 3000)  # 10000 - 7000 退还
        self.assertEqual(
            view["collected_cents"], view["refunded_cents"] + view["deducted_cents"]
        )

    def test_invalid_amount_rejected(self) -> None:
        for bad in (0, -1, "100", 1.5, True, None):
            with self.assertRaises(ValidationError):
                self.deposits.collect(
                    self.booking_id,
                    {"idempotency_key": f"k-bad-{bad!r}", "amount_cents": bad},
                )

    def test_collect_unknown_booking_is_not_found(self) -> None:
        with self.assertRaises(NotFoundError):
            self.deposits.collect("bkg_missing", {"amount_cents": 100})


class DepositSQLiteTests(unittest.TestCase):
    """SQLite 事务更新：落盘持久、重开库后重复退款仍只对应一笔记录。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = f"{self._tmp.name}/booking.db"
        self.store = SQLiteStore(self.db_path)
        clock = ManualClock(NOW)
        ids = UuidIdGenerator()
        self.catalog = CatalogService(self.store, clock, ids)
        self.bookings = BookingService(self.store, clock, ids)
        self.clock = clock
        self.ids_gen = ids

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def _deposits(self) -> DepositService:
        return DepositService(self.store, self.clock, self.ids_gen)

    def _seed_booking(self) -> str:
        ids = seed_catalog(self.catalog)
        return self.bookings.apply(apply_payload(ids, "k-sqlite-apply"))["booking_id"]

    def test_entries_persist_and_replay_after_reopen(self) -> None:
        booking_id = self._seed_booking()
        deposits = self._deposits()
        deposits.collect(
            booking_id, {"idempotency_key": "k-sql-collect", "amount_cents": DEPOSIT_CENTS}
        )
        deposits.refund(
            booking_id, {"idempotency_key": "k-sql-refund", "amount_cents": 4000}
        )
        self.assertEqual(len(self.store.query(COLLECTION_DEPOSIT_ENTRIES)), 2)

        # 模拟服务重启：新连接、新服务实例打开同一个库文件
        self.store.close()
        self.store = SQLiteStore(self.db_path)
        reopened = self._deposits()
        replay = reopened.refund(
            booking_id, {"idempotency_key": "k-sql-refund", "amount_cents": 4000}
        )
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(replay["refunded_cents"], 4000)
        self.assertEqual(replay["cash_cents"], 6000)
        # 重开库后的重复退款仍只对应一笔退还记录
        self.assertEqual(len(self.store.query(COLLECTION_DEPOSIT_ENTRIES)), 2)

    def test_concurrent_duplicate_refunds_produce_one_entry(self) -> None:
        booking_id = self._seed_booking()
        deposits = self._deposits()
        deposits.collect(
            booking_id, {"idempotency_key": "k-sql-collect", "amount_cents": DEPOSIT_CENTS}
        )

        def try_refund(_: int) -> dict:
            return deposits.refund(
                booking_id,
                {"idempotency_key": "k-sql-refund-race"},  # 缺省退全额
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(try_refund, range(8)))
        winners = [r for r in results if not r.get("idempotent_replay")]
        replays = [r for r in results if r.get("idempotent_replay")]
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(replays), 7)
        for view in results:
            self.assertEqual(view["cash_cents"], 0)
            self.assertEqual(view["held_cents"], 0)
            _assert_conserved(self, view)
        # 并发重复退款只产生一笔退还分录
        refund_entries = [
            e for e in self.store.query(COLLECTION_DEPOSIT_ENTRIES) if e["type"] == "REFUND"
        ]
        self.assertEqual(len(refund_entries), 1)
        self.assertEqual(refund_entries[0]["amount_cents"], DEPOSIT_CENTS)


class DepositHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        catalog, bookings, deposits, clock, store = make_services()
        ids = seed_catalog(catalog)
        cls.booking_id = bookings.apply(apply_payload(ids, "k-http-deposit-apply"))["booking_id"]
        cls.server = create_server("127.0.0.1", 0, catalog, bookings, deposits)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _request(self, method: str, path: str, body: dict | None, headers: dict | None = None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method, headers=headers or {}
        )
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_collect_refund_deduct_over_http_with_replay(self) -> None:
        status, collected = self._request(
            "POST",
            f"/bookings/{self.booking_id}/deposit/collect",
            {"amount_cents": 10000},
            headers={"Idempotency-Key": "k-http-collect"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(collected["held_cents"], 10000)

        status, refunded = self._request(
            "POST",
            f"/bookings/{self.booking_id}/deposit/refund",
            {"amount_cents": 4000},
            headers={"Idempotency-Key": "k-http-refund"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(refunded["cash_cents"], 6000)

        # 同键重复退款重放，不产生第二笔
        status, replay = self._request(
            "POST",
            f"/bookings/{self.booking_id}/deposit/refund",
            {"amount_cents": 4000},
            headers={"Idempotency-Key": "k-http-refund"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(replay["idempotent_replay"])

        status, ledger = self._request(
            "GET", f"/bookings/{self.booking_id}/deposit", None
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(ledger["entries"]), 2)
        self.assertEqual(
            ledger["collected_cents"],
            ledger["refunded_cents"] + ledger["deducted_cents"] + ledger["held_cents"],
        )


if __name__ == "__main__":
    unittest.main()
