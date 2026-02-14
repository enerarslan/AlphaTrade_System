"""Unit tests for audit storage and training audit events."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_trading_system.monitoring.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditSeverity,
    DatabaseAuditStorage,
    FileAuditStorage,
)


def _event(
    event_id: str,
    timestamp: datetime,
    event_type: AuditEventType = AuditEventType.MODEL_TRAINING_STARTED,
    previous_hash: str | None = None,
) -> AuditEvent:
    return AuditEvent(
        event_id=event_id,
        event_type=event_type,
        timestamp=timestamp,
        severity=AuditSeverity.INFO,
        source="unit-test",
        user_id="tester",
        session_id="session-1",
        action=event_type.value,
        details={"model": "xgboost"},
        outcome="success",
        previous_hash=previous_hash,
    )


def test_database_audit_storage_requires_connection_string() -> None:
    with pytest.raises(ValueError, match="connection string"):
        DatabaseAuditStorage("")


def test_database_audit_storage_roundtrip_and_chain_correction() -> None:
    storage = DatabaseAuditStorage("sqlite+pysqlite:///:memory:")
    t0 = datetime.now(timezone.utc)

    first = _event("evt-1", t0, AuditEventType.MODEL_TRAINING_STARTED)
    storage.store(first)

    second = _event(
        "evt-2",
        t0 + timedelta(seconds=1),
        AuditEventType.MODEL_TRAINING_COMPLETED,
        previous_hash="incorrect",
    )
    storage.store(second)

    events = storage.retrieve(limit=10)
    assert len(events) == 2
    assert events[0].event_id == "evt-1"
    assert events[1].event_id == "evt-2"
    assert events[1].previous_hash == events[0].event_hash
    assert storage.get_last_hash() == events[-1].event_hash

    completed = storage.retrieve(event_types=[AuditEventType.MODEL_TRAINING_COMPLETED])
    assert len(completed) == 1
    assert completed[0].event_id == "evt-2"


def test_audit_logger_model_training_events(tmp_path) -> None:
    storage = FileAuditStorage(storage_dir=tmp_path)
    logger = AuditLogger(storage=storage, source="training_tests")

    started = logger.log_model_training_started(
        model_name="xgboost",
        model_version="xgb_v1",
        config={"cv_method": "purged_kfold"},
        symbols=["AAPL", "MSFT"],
    )
    assert started.event_type == AuditEventType.MODEL_TRAINING_STARTED
    assert started.outcome == "success"
    assert started.details["model_name"] == "xgboost"

    completed = logger.log_model_training_completed(
        model_name="xgboost",
        model_version="xgb_v1",
        metrics={"mean_accuracy": 0.61},
        model_path="models/xgb_v1.pkl",
        duration_seconds=12.5,
        success=False,
        error="validation gates failed",
    )
    assert completed.event_type == AuditEventType.MODEL_TRAINING_COMPLETED
    assert completed.outcome == "failure"
    assert completed.severity == AuditSeverity.WARNING
    assert completed.details["metrics"]["mean_accuracy"] == 0.61
