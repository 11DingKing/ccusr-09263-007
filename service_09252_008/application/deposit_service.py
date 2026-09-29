"""预约押金服务：收取、退还、抵扣三类资金分录与金额守恒。

账务模型（单位：整数分）::

    collected_cents == refunded_cents + deducted_cents + held_cents
    cash_cents      == collected_cents - refunded_cents

- 收取（COLLECT）：现金与押金负债同增；
- 退还（REFUND）：现金与押金负债同减；
- 抵扣（DEDUCT）：押金负债转为平台收入，现金不变。

每个用例在单个 ``store.transaction()`` 内完成“查重-判余-过账”，
SQLite 后端以 ``BEGIN IMMEDIATE`` 串行化写者；重复的退款请求命中去重键后
直接重放首次结果，不会产生第二笔资金变动。
"""
from __future__ import annotations

from typing import Any

from ..domain.errors import BusinessRuleError, NotFoundError, StateError, ValidationError
from ..domain.models import DepositEntry, DepositEntryType, DepositLedger, DomainEvent, dt_to_str
from ..domain.rules import deposit_dedup_key
from ..persistence.store import Store
from .booking_service import COLLECTION_BOOKINGS, COLLECTION_EVENTS
from .ports import Clock, IdGenerator

COLLECTION_DEPOSIT_LEDGERS = "deposit_ledgers"
COLLECTION_DEPOSIT_ENTRIES = "deposit_entries"
COLLECTION_DEPOSIT_DEDUP = "deposit_entry_dedup"

DEFAULT_CURRENCY = "CNY"


class DepositService:
    """预约押金用例：收取、退还、抵扣、查询。"""

    def __init__(self, store: Store, clock: Clock, ids: IdGenerator) -> None:
        self._store = store
        self._clock = clock
        self._ids = ids

    # ------------------------------------------------------------------
    # 读取辅助
    # ------------------------------------------------------------------

    def _booking_exists(self, booking_id: str) -> None:
        if self._store.get(COLLECTION_BOOKINGS, booking_id) is None:
            raise NotFoundError(
                f"booking not found: {booking_id}", details={"booking_id": booking_id}
            )

    def _load_ledger(self, booking_id: str) -> DepositLedger | None:
        rows = self._store.query(COLLECTION_DEPOSIT_LEDGERS, booking_id=booking_id)
        return DepositLedger.from_dict(rows[0]) if rows else None

    def _require_ledger(self, booking_id: str) -> DepositLedger:
        ledger = self._load_ledger(booking_id)
        if ledger is None:
            raise StateError(
                "no deposit has been collected for this booking",
                details={"booking_id": booking_id},
            )
        return ledger

    def _entries_of(self, booking_id: str) -> list[DepositEntry]:
        rows = self._store.query(COLLECTION_DEPOSIT_ENTRIES, booking_id=booking_id)
        entries = [DepositEntry.from_dict(r) for r in rows]
        entries.sort(key=lambda e: (e.created_at, e.entry_id))
        return entries

    def _emit(self, event_type: str, booking_id: str, payload: dict[str, Any]) -> None:
        event = DomainEvent(
            event_id=self._ids.new_id("evt"),
            type=event_type,
            booking_id=booking_id,
            payload=payload,
            created_at=self._clock.now(),
        )
        self._store.put(COLLECTION_EVENTS, event.event_id, event.to_dict())

    # ------------------------------------------------------------------
    # 收取
    # ------------------------------------------------------------------

    def collect(self, booking_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        request = request or {}
        amount = self._parse_amount(request, "amount_cents")
        currency = self._parse_currency(request)
        idem_key = request.get("idempotency_key")
        with self._store.transaction():
            self._booking_exists(booking_id)
            return self._post_entry(
                booking_id,
                DepositEntryType.COLLECT,
                amount,
                idem_key=idem_key,
                currency=currency,
                reason=request.get("reason"),
                create_ledger=True,
            )

    # ------------------------------------------------------------------
    # 退还
    # ------------------------------------------------------------------

    def refund(self, booking_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        request = request or {}
        idem_key = request.get("idempotency_key")
        with self._store.transaction():
            ledger = self._require_ledger(booking_id)
            # 缺省退全部在押金额；显式金额也不得超过在押金额
            amount = (
                ledger.held_cents
                if request.get("amount_cents") is None
                else self._parse_amount(request, "amount_cents")
            )
            return self._post_entry(
                booking_id,
                DepositEntryType.REFUND,
                amount,
                idem_key=idem_key,
                currency=ledger.currency,
                reason=request.get("reason"),
                ledger=ledger,
            )

    # ------------------------------------------------------------------
    # 抵扣
    # ------------------------------------------------------------------

    def deduct(self, booking_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        request = request or {}
        idem_key = request.get("idempotency_key")
        with self._store.transaction():
            ledger = self._require_ledger(booking_id)
            amount = (
                ledger.held_cents
                if request.get("amount_cents") is None
                else self._parse_amount(request, "amount_cents")
            )
            return self._post_entry(
                booking_id,
                DepositEntryType.DEDUCT,
                amount,
                idem_key=idem_key,
                currency=ledger.currency,
                reason=request.get("reason"),
                ledger=ledger,
            )

    # ------------------------------------------------------------------
    # 过账核心：事务内查重、判余、守恒、落库
    # ------------------------------------------------------------------

    def _post_entry(
        self,
        booking_id: str,
        entry_type: DepositEntryType,
        amount: int,
        *,
        idem_key: str | None,
        currency: str,
        reason: str | None,
        create_ledger: bool = False,
        ledger: DepositLedger | None = None,
    ) -> dict[str, Any]:
        now = self._clock.now()
        dedup = deposit_dedup_key(booking_id, entry_type.value, idem_key) if idem_key is not None else None

        # 1) 去重：同键重复请求重放首次结果，不产生第二笔资金变动。
        #    无键请求不做静默重放，改由下方“已收取/在押不足”规则显式拒绝。
        if dedup is not None:
            existing = self._store.get(COLLECTION_DEPOSIT_DEDUP, dedup)
            if existing is not None:
                replayed = DepositEntry.from_dict(existing["entry"])
                return self._view(self._load_ledger(booking_id), replayed, idempotent_replay=True)

        # 2) 取/建台账
        if create_ledger:
            if ledger is None:
                ledger = self._load_ledger(booking_id)
            if ledger is not None:
                # 同键重放已在第 1 步返回；走到这里说明一个预约试图收第二笔押金
                raise BusinessRuleError(
                    "deposit has already been collected for this booking",
                    details={"booking_id": booking_id},
                )
            ledger = DepositLedger(
                ledger_id=self._ids.new_id("dpl"),
                booking_id=booking_id,
                currency=currency,
                created_at=now,
                updated_at=now,
            )
        else:
            assert ledger is not None

        # 3) 判余：退还/抵扣不得超过在押金额（在押为 0 视为重复结清请求，拒绝）
        if entry_type is not DepositEntryType.COLLECT:
            if ledger.held_cents <= 0:
                raise BusinessRuleError(
                    "no held deposit left for this booking",
                    details={"booking_id": booking_id, "state": ledger.state.value},
                )
            if amount > ledger.held_cents:
                raise BusinessRuleError(
                    "amount exceeds held deposit",
                    details={
                        "booking_id": booking_id,
                        "amount_cents": amount,
                        "held_cents": ledger.held_cents,
                    },
                )

        # 4) 过账并验证守恒（守恒式内置在台账结转中，此处防御性断言）
        entry = DepositEntry(
            entry_id=self._ids.new_id("dpe"),
            booking_id=booking_id,
            type=entry_type,
            amount_cents=amount,
            created_at=now,
            idempotency_key=idem_key,
            reason=reason,
        )
        ledger.apply(entry)
        if not ledger.is_consistent():
            # 理论不可达：任意单笔过账都保持守恒，触发即说明数据被破坏
            raise BusinessRuleError(
                "deposit conservation violated after posting",
                details={"booking_id": booking_id, "ledger_id": ledger.ledger_id},
            )

        # 5) 落库：台账、分录、去重键（若有）同事务提交
        self._store.put(COLLECTION_DEPOSIT_LEDGERS, ledger.ledger_id, ledger.to_dict())
        self._store.put(COLLECTION_DEPOSIT_ENTRIES, entry.entry_id, entry.to_dict())
        if dedup is not None:
            self._store.put(
                COLLECTION_DEPOSIT_DEDUP,
                dedup,
                {
                    "dedup_key": dedup,
                    "booking_id": booking_id,
                    "entry_type": entry_type.value,
                    "idempotency_key": idem_key,
                    "entry": entry.to_dict(),
                    "created_at": dt_to_str(entry.created_at),
                },
            )
        self._emit(
            f"deposit_{entry_type.value.lower()}",
            booking_id,
            {
                "entry_id": entry.entry_id,
                "amount_cents": amount,
                "cash_cents": ledger.cash_cents,
                "held_cents": ledger.held_cents,
            },
        )
        return self._view(ledger, entry, idempotent_replay=False)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_deposit(self, booking_id: str) -> dict[str, Any]:
        self._booking_exists(booking_id)
        ledger = self._load_ledger(booking_id)
        if ledger is None:
            raise NotFoundError(
                f"no deposit ledger for booking: {booking_id}", details={"booking_id": booking_id}
            )
        return self._view(ledger, None, idempotent_replay=False)

    # ------------------------------------------------------------------
    # 输入校验与视图
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_amount(request: dict[str, Any], field: str) -> int:
        raw = request.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValidationError(
                f"field {field} must be a positive integer amount in cents",
                details={"field": field},
            )
        return raw

    @staticmethod
    def _parse_currency(request: dict[str, Any]) -> str:
        currency = request.get("currency", DEFAULT_CURRENCY)
        if not isinstance(currency, str) or not currency.strip():
            raise ValidationError("field currency must be a non-empty string")
        return currency.strip().upper()

    def _view(
        self,
        ledger: DepositLedger | None,
        last_entry: DepositEntry | None,
        *,
        idempotent_replay: bool,
    ) -> dict[str, Any]:
        if ledger is None:
            raise NotFoundError("deposit ledger missing")
        view: dict[str, Any] = {
            "ledger_id": ledger.ledger_id,
            "booking_id": ledger.booking_id,
            "currency": ledger.currency,
            "state": ledger.state.value,
            "collected_cents": ledger.collected_cents,
            "refunded_cents": ledger.refunded_cents,
            "deducted_cents": ledger.deducted_cents,
            "held_cents": ledger.held_cents,
            "cash_cents": ledger.cash_cents,
            "entries": [e.to_dict() for e in self._entries_of(ledger.booking_id)],
            "last_entry": last_entry.to_dict() if last_entry is not None else None,
        }
        if idempotent_replay:
            view["idempotent_replay"] = True
        return view
