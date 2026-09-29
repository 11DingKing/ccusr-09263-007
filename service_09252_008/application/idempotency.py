"""共享的幂等执行支撑。

变更类用例支持幂等键：相同键重放返回首次结果（``idempotent_replay: true``），
不同载荷复用键则冲突。幂等记录与业务写入在同一存储事务内提交，
因此重放绝不会产生第二笔资金变动。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from ..domain.errors import IdempotencyConflict, ValidationError
from ..domain.models import dt_to_str
from ..persistence.store import Store
from .ports import Clock

COLLECTION_IDEMPOTENCY = "idempotency_keys"


def canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def run_idempotent(
    store: Store,
    clock: Clock,
    endpoint: str,
    key: str | None,
    payload: dict[str, Any],
    fn: Callable[[], dict[str, Any]],
    *,
    required: bool,
) -> dict[str, Any]:
    """在单事务内完成“查重放/判冲突/执行/落键”。

    存储事务为排他事务（内存后端加全局锁、SQLite 后端 ``BEGIN IMMEDIATE``），
    两个携带相同键的并发请求只有一个能进入执行段，另一个只会读到首次结果。
    """
    if key is None:
        if required:
            raise ValidationError("idempotency_key is required for this operation")
        with store.transaction():
            return fn()
    fingerprint = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
    with store.transaction():
        existing = store.get(COLLECTION_IDEMPOTENCY, key)
        if existing is not None:
            if existing["endpoint"] != endpoint or existing["request_hash"] != fingerprint:
                raise IdempotencyConflict(
                    "idempotency key was already used with a different request",
                    details={"key": key, "endpoint": endpoint},
                )
            return {**existing["response"], "idempotent_replay": True}
        result = fn()
        store.put(
            COLLECTION_IDEMPOTENCY,
            key,
            {
                "key": key,
                "endpoint": endpoint,
                "request_hash": fingerprint,
                "response": result,
                "created_at": dt_to_str(clock.now()),
            },
        )
        return result
