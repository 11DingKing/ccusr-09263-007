"""预约押金服务：收取、退还、抵扣三类资金分录。

财务核对约定：
- 押金拆成 ``COLLECT``（收取）、``REFUND``（退还）、``APPLY``（抵扣）三类分录；
- 金额合计守恒：收取合计 == 退还合计 + 抵扣合计，在押余额恒为非负；
- 每次“读-判-写”都在单个 ``store.transaction()`` 内完成，
  内存后端为全局排他锁，SQLite 后端为 ``BEGIN IMMEDIATE``；
- 退还/抵扣金额超过在押余额时直接报错，事务回滚，不落任何分录；
- 重复退款有双重防护：相同幂等键重放只返回首次结果；即便用新键重试，
  在押余额结清后的状态守卫也会拒绝，均不会产生第二笔资金变动。
"""
from __future__ import annotations

from typing import Any

from ..domain.errors import NotFoundError, StateError, ValidationError
from ..domain.models import (
    DomainEvent,
    DepositEntry,
    DepositEntryType,
    DepositLedger,
)
from ..persistence.store import Store
from .booking_service import COLLECTION_BOOKINGS, COLLECTION_EVENTS
from .idempotency import run_idempotent
from .ports import Clock, IdGenerator

COLLECTION_DEPOSITS = "deposit_ledgers"

DEFAULT_CURRENCY = "CNY"


class DepositService:
    """预约押金用例编排。"""

    def __init__(self, store: Store, clock: Clock, ids: IdGenerator) -> None:
        self._store = store
        self._clock = clock
        self._ids = ids

    # ------------------------------------------------------------------
    # 基础设施
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, booking_id: str, payload: dict[str, Any]) -> None:
        event = DomainEvent(
            event_id=self._ids.new_id("evt"),
            type=event_type,
            booking_id=booking_id,
            payload=payload,
            created_at=self._clock.now(),
        )
        self._store.put(COLLECTION_EVENTS, event.event_id, event.to_dict())

    def _require_booking(self, booking_id: str) -> None:
        if self._store.get(COLLECTION_BOOKINGS, booking_id) is None:
            raise NotFoundError(f"booking not found: {booking_id}", details={"booking_id": booking_id})

    def _load_ledger(self, booking_id: str) -> DepositLedger:
        records = self._store.query(COLLECTION_DEPOSITS, booking_id=booking_id)
        if not records:
            raise NotFoundError(
                f"deposit ledger not found for booking: {booking_id}", details={"booking_id": booking_id}
            )
        return DepositLedger.from_dict(records[0])

    def _save_ledger(self, ledger: DepositLedger) -> None:
        ledger.updated_at = self._clock.now()
        self._store.put(COLLECTION_DEPOSITS, ledger.deposit_id, ledger.to_dict())

    @staticmethod
    def _amount(request: dict[str, Any], *, fallback: int | None = None) -> int:
        raw = request.get("amount_cents", fallback)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValidationError("field amount_cents must be a positive integer", details={"field": "amount_cents"})
        return raw

    def _view(self, ledger: DepositLedger) -> dict[str, Any]:
        view = ledger.to_dict()
        view["state"] = ledger.state.value
        view["collected_cents"] = ledger.collected_cents
        view["refunded_cents"] = ledger.refunded_cents
        view["applied_cents"] = ledger.applied_cents
        view["balance_cents"] = ledger.balance_cents
        return view

    # ------------------------------------------------------------------
    # 收取
    # ------------------------------------------------------------------

    def collect(self, booking_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """收取押金：建立台账并写入一笔 COLLECT 分录（需幂等键）。"""
        key = request.get("idempotency_key")
        return run_idempotent(
            self._store,
            self._clock,
            "deposit_collect",
            key,
            {"booking_id": booking_id, **request},
            lambda: self._collect(booking_id, request),
            required=True,
        )

    def _collect(self, booking_id: str, request: dict[str, Any]) -> dict[str, Any]:
        amount = self._amount(request)
        currency = request.get("currency", DEFAULT_CURRENCY)
        if not isinstance(currency, str) or not currency.strip():
            raise ValidationError("field currency must be a non-empty string")
        with self._store.transaction():
            self._require_booking(booking_id)
            if self._store.query(COLLECTION_DEPOSITS, booking_id=booking_id):
                raise StateError(
                    "deposit was already collected for this booking",
                    details={"booking_id": booking_id},
                )
            now = self._clock.now()
            ledger = DepositLedger(
                deposit_id=self._ids.new_id("dep"),
                booking_id=booking_id,
                currency=currency.strip(),
                created_at=now,
                updated_at=now,
            )
            ledger.add(
                DepositEntry(
                    entry_id=self._ids.new_id("ent"),
                    deposit_id=ledger.deposit_id,
                    booking_id=booking_id,
                    type=DepositEntryType.COLLECT,
                    amount_cents=amount,
                    created_at=now,
                    reason=str(request.get("reason", "")),
                )
            )
            self._save_ledger(ledger)
            self._emit(
                "deposit_collected",
                booking_id,
                {"deposit_id": ledger.deposit_id, "amount_cents": amount, "currency": ledger.currency},
            )
            return self._view(ledger)

    # ------------------------------------------------------------------
    # 退还
    # ------------------------------------------------------------------

    def refund(self, booking_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        """退还押金：写入一笔 REFUND 分录；缺省金额为全额退还（需幂等键）。

        重复请求不会产生第二笔资金变动：相同幂等键重放只返回首次结果；
        即便用新键重试，在押余额结清后的状态守卫也会拒绝。
        """
        request = request or {}
        key = request.get("idempotency_key")
        return run_idempotent(
            self._store,
            self._clock,
            "deposit_refund",
            key,
            {"booking_id": booking_id, **request},
            lambda: self._refund(booking_id, request),
            required=True,
        )

    def _refund(self, booking_id: str, request: dict[str, Any]) -> dict[str, Any]:
        with self._store.transaction():
            ledger = self._load_ledger(booking_id)
            if ledger.balance_cents <= 0:
                raise StateError(
                    "deposit has no held balance to refund",
                    details={"booking_id": booking_id, "state": ledger.state.value},
                )
            amount = self._amount(request, fallback=ledger.balance_cents)
            if amount > ledger.balance_cents:
                raise ValidationError(
                    "refund amount exceeds held deposit balance",
                    details={"amount_cents": amount, "balance_cents": ledger.balance_cents},
                )
            now = self._clock.now()
            ledger.add(
                DepositEntry(
                    entry_id=self._ids.new_id("ent"),
                    deposit_id=ledger.deposit_id,
                    booking_id=booking_id,
                    type=DepositEntryType.REFUND,
                    amount_cents=amount,
                    created_at=now,
                    reason=str(request.get("reason", "")),
                )
            )
            self._save_ledger(ledger)
            self._emit(
                "deposit_refunded",
                booking_id,
                {"amount_cents": amount, "balance_cents": ledger.balance_cents},
            )
            return self._view(ledger)

    # ------------------------------------------------------------------
    # 抵扣
    # ------------------------------------------------------------------

    def apply_deposit(self, booking_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """抵扣押金：写入一笔 APPLY 分录，将在押押金结转为收入；缺省金额为全额（需幂等键）。"""
        request = request or {}
        key = request.get("idempotency_key")
        return run_idempotent(
            self._store,
            self._clock,
            "deposit_apply",
            key,
            {"booking_id": booking_id, **request},
            lambda: self._apply(booking_id, request),
            required=True,
        )

    def _apply(self, booking_id: str, request: dict[str, Any]) -> dict[str, Any]:
        with self._store.transaction():
            ledger = self._load_ledger(booking_id)
            if ledger.balance_cents <= 0:
                raise StateError(
                    "deposit has no held balance to apply",
                    details={"booking_id": booking_id, "state": ledger.state.value},
                )
            amount = self._amount(request, fallback=ledger.balance_cents)
            if amount > ledger.balance_cents:
                raise ValidationError(
                    "apply amount exceeds held deposit balance",
                    details={"amount_cents": amount, "balance_cents": ledger.balance_cents},
                )
            now = self._clock.now()
            ledger.add(
                DepositEntry(
                    entry_id=self._ids.new_id("ent"),
                    deposit_id=ledger.deposit_id,
                    booking_id=booking_id,
                    type=DepositEntryType.APPLY,
                    amount_cents=amount,
                    created_at=now,
                    reason=str(request.get("reason", "")),
                )
            )
            self._save_ledger(ledger)
            self._emit(
                "deposit_applied",
                booking_id,
                {"amount_cents": amount, "balance_cents": ledger.balance_cents},
            )
            return self._view(ledger)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_deposit(self, booking_id: str) -> dict[str, Any]:
        with self._store.transaction():
            return self._view(self._load_ledger(booking_id))

    def list_deposits(self) -> list[dict[str, Any]]:
        with self._store.transaction():
            ledgers = [DepositLedger.from_dict(r) for r in self._store.query(COLLECTION_DEPOSITS)]
        ledgers.sort(key=lambda l: (l.created_at, l.deposit_id))
        return [self._view(l) for l in ledgers]
