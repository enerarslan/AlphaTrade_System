"""
JPMORGAN FIX: Audit Logging Module.

Provides comprehensive audit logging for regulatory compliance:
- Immutable audit trail for all trading decisions
- Order lifecycle tracking
- Model inference logging
- Risk decision logging
- User action logging
- Tamper-evident log storage

Designed to meet MiFID II, SEC, and FINRA requirements.

Author: AlphaTrade System
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Trading events
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_MODIFIED = "order_modified"

    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_MODIFIED = "position_modified"

    # Risk events
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    KILL_SWITCH_RESET = "kill_switch_reset"

    # Model events
    MODEL_PREDICTION = "model_prediction"
    MODEL_TRAINING_STARTED = "model_training_started"
    MODEL_TRAINING_COMPLETED = "model_training_completed"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_RETIRED = "model_retired"

    # Signal events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_REJECTED = "signal_rejected"
    SIGNAL_EXECUTED = "signal_executed"

    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIGURATION_CHANGED = "configuration_changed"
    ERROR_OCCURRED = "error_occurred"

    # User events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_ACTION = "user_action"
    AUTHORIZATION_GRANTED = "authorization_granted"
    AUTHORIZATION_DENIED = "authorization_denied"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Immutable audit event record."""

    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    severity: AuditSeverity
    source: str  # Component that generated the event
    user_id: str | None
    session_id: str | None
    action: str
    details: dict[str, Any]
    outcome: str  # "success", "failure", "pending"
    previous_hash: str | None = None  # For chain integrity
    event_hash: str | None = None

    def __post_init__(self):
        """Calculate event hash after initialization."""
        if self.event_hash is None:
            self.event_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the event."""
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "source": self.source,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "action": self.action,
            "details": self.details,
            "outcome": self.outcome,
            "previous_hash": self.previous_hash,
        }
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify the event hash is valid."""
        expected_hash = self._calculate_hash()
        return self.event_hash == expected_hash

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "source": self.source,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "action": self.action,
            "details": self.details,
            "outcome": self.outcome,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=AuditEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            severity=AuditSeverity(data["severity"]),
            source=data["source"],
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            action=data["action"],
            details=data.get("details", {}),
            outcome=data["outcome"],
            previous_hash=data.get("previous_hash"),
            event_hash=data.get("event_hash"),
        )


class AuditStorage(ABC):
    """Abstract base class for audit log storage."""

    @abstractmethod
    def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        pass

    @abstractmethod
    def retrieve(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        limit: int = 1000,
    ) -> list[AuditEvent]:
        """Retrieve audit events with optional filters."""
        pass

    @abstractmethod
    def get_last_hash(self) -> str | None:
        """Get hash of the last stored event."""
        pass


class FileAuditStorage(AuditStorage):
    """
    File-based audit storage with chain integrity.

    Stores events in append-only JSON files with hash chaining
    for tamper detection.
    """

    def __init__(
        self,
        storage_dir: Path | str,
        max_file_size_mb: int = 100,
        rotate_daily: bool = True,
    ):
        """Initialize file storage.

        Args:
            storage_dir: Directory for audit logs.
            max_file_size_mb: Maximum file size before rotation.
            rotate_daily: Whether to rotate logs daily.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.rotate_daily = rotate_daily

        self._lock = threading.Lock()
        self._last_hash: str | None = None
        self._current_file: Path | None = None
        self._event_count = 0

        # Load last hash from existing logs
        self._load_last_hash()

    def _load_last_hash(self) -> None:
        """Load the last event hash from existing logs."""
        log_files = sorted(self.storage_dir.glob("audit_*.jsonl"), reverse=True)
        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_event = json.loads(lines[-1])
                        self._last_hash = last_event.get("event_hash")
                        return
            except Exception:
                continue

    def _get_current_file(self) -> Path:
        """Get the current log file, rotating if necessary."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")

        if self._current_file is None:
            self._current_file = self.storage_dir / f"audit_{today}_001.jsonl"

        # Check if rotation needed
        if self._current_file.exists():
            if self._current_file.stat().st_size >= self.max_file_size:
                # Size-based rotation
                base = self._current_file.stem
                parts = base.rsplit("_", 1)
                num = int(parts[1]) + 1
                self._current_file = self.storage_dir / f"{parts[0]}_{num:03d}.jsonl"

            elif self.rotate_daily and today not in self._current_file.name:
                # Daily rotation
                self._current_file = self.storage_dir / f"audit_{today}_001.jsonl"

        return self._current_file

    def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        with self._lock:
            log_file = self._get_current_file()

            # Ensure chain integrity
            if event.previous_hash != self._last_hash:
                event = AuditEvent(
                    event_id=event.event_id,
                    event_type=event.event_type,
                    timestamp=event.timestamp,
                    severity=event.severity,
                    source=event.source,
                    user_id=event.user_id,
                    session_id=event.session_id,
                    action=event.action,
                    details=event.details,
                    outcome=event.outcome,
                    previous_hash=self._last_hash,
                )

            # Append to file
            with open(log_file, "a") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")

            self._last_hash = event.event_hash
            self._event_count += 1

    def retrieve(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        limit: int = 1000,
        verify_on_read: bool = False,
    ) -> list[AuditEvent]:
        """
        Retrieve audit events with optional filters.

        MAJOR FIX: Added verify_on_read parameter for hash chain verification
        during retrieval to detect tampering.

        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            event_types: Filter by event types
            limit: Maximum events to return
            verify_on_read: If True, verify each event's hash and chain

        Returns:
            List of audit events

        Raises:
            AuditIntegrityError: If verify_on_read=True and integrity check fails
        """
        events = []
        log_files = sorted(self.storage_dir.glob("audit_*.jsonl"))

        previous_hash: str | None = None
        integrity_errors: list[str] = []

        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        if len(events) >= limit:
                            break

                        event_data = json.loads(line)
                        event = AuditEvent.from_dict(event_data)

                        # MAJOR FIX: Verify integrity on read if requested
                        if verify_on_read:
                            # Verify event hash
                            if not event.verify_integrity():
                                integrity_errors.append(
                                    f"HASH MISMATCH: Event {event.event_id} at {event.timestamp} "
                                    f"has invalid hash (possible tampering)"
                                )

                            # Verify chain link
                            if previous_hash is not None and event.previous_hash != previous_hash:
                                integrity_errors.append(
                                    f"CHAIN BREAK: Event {event.event_id} expected previous_hash "
                                    f"{previous_hash[:16]}... but got {(event.previous_hash or 'None')[:16]}..."
                                )

                            previous_hash = event.event_hash

                        # Apply filters
                        if start_time and event.timestamp < start_time:
                            continue
                        if end_time and event.timestamp > end_time:
                            continue
                        if event_types and event.event_type not in event_types:
                            continue

                        events.append(event)

            except Exception as e:
                logger.warning(f"Error reading audit log {log_file}: {e}")

        # MAJOR FIX: Raise error if integrity verification failed
        if verify_on_read and integrity_errors:
            from quant_trading_system.core.exceptions import TradingSystemError

            error_summary = "; ".join(integrity_errors[:5])  # First 5 errors
            if len(integrity_errors) > 5:
                error_summary += f" ... and {len(integrity_errors) - 5} more errors"

            logger.critical(f"AUDIT LOG INTEGRITY VIOLATION: {error_summary}")
            raise TradingSystemError(
                f"Audit log integrity verification failed: {len(integrity_errors)} errors detected. "
                f"{error_summary}"
            )

        return events[:limit]

    def retrieve_verified(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        limit: int = 1000,
    ) -> tuple[list[AuditEvent], bool, list[str]]:
        """
        MAJOR FIX: Retrieve audit events with integrity verification.

        This method always verifies the hash chain but returns errors
        instead of raising exceptions, allowing callers to decide how to handle.

        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            event_types: Filter by event types
            limit: Maximum events to return

        Returns:
            Tuple of (events, is_valid, error_messages)
        """
        events = []
        log_files = sorted(self.storage_dir.glob("audit_*.jsonl"))

        previous_hash: str | None = None
        integrity_errors: list[str] = []

        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        if len(events) >= limit:
                            break

                        event_data = json.loads(line)
                        event = AuditEvent.from_dict(event_data)

                        # Verify event hash
                        if not event.verify_integrity():
                            integrity_errors.append(
                                f"HASH MISMATCH: Event {event.event_id} at {event.timestamp}"
                            )

                        # Verify chain link
                        if previous_hash is not None and event.previous_hash != previous_hash:
                            integrity_errors.append(
                                f"CHAIN BREAK at event {event.event_id}"
                            )

                        previous_hash = event.event_hash

                        # Apply filters
                        if start_time and event.timestamp < start_time:
                            continue
                        if end_time and event.timestamp > end_time:
                            continue
                        if event_types and event.event_type not in event_types:
                            continue

                        events.append(event)

            except Exception as e:
                integrity_errors.append(f"Read error in {log_file}: {e}")
                logger.warning(f"Error reading audit log {log_file}: {e}")

        is_valid = len(integrity_errors) == 0
        if not is_valid:
            logger.warning(f"Audit log integrity issues: {len(integrity_errors)} errors found")

        return events[:limit], is_valid, integrity_errors

    def get_last_hash(self) -> str | None:
        """Get hash of the last stored event."""
        return self._last_hash

    def verify_chain_integrity(self) -> tuple[bool, list[str]]:
        """Verify integrity of the entire audit chain.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []
        previous_hash = None

        for event in self.retrieve(limit=100000):
            # Verify event hash
            if not event.verify_integrity():
                errors.append(
                    f"Event {event.event_id} hash mismatch at {event.timestamp}"
                )

            # Verify chain link
            if event.previous_hash != previous_hash:
                errors.append(
                    f"Chain break at event {event.event_id}: "
                    f"expected {previous_hash}, got {event.previous_hash}"
                )

            previous_hash = event.event_hash

        return len(errors) == 0, errors


class DatabaseAuditStorage(AuditStorage):
    """
    Database-based audit storage.

    FIX: Changed from silent placeholder to explicit NotImplementedError.
    Use FileAuditStorage for production until database implementation is complete.
    """

    def __init__(self, connection_string: str):
        """Initialize database storage.

        Raises:
            NotImplementedError: Database audit storage is not yet implemented.
        """
        # FIX: Fail explicitly instead of silently doing nothing
        raise NotImplementedError(
            "DatabaseAuditStorage is not yet implemented. "
            "Use FileAuditStorage for production audit logging. "
            "Example: AuditLogger(FileAuditStorage('/var/log/alphatrade/audit'))"
        )

    def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        raise NotImplementedError("DatabaseAuditStorage not implemented")

    def retrieve(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        limit: int = 1000,
    ) -> list[AuditEvent]:
        """Retrieve audit events."""
        raise NotImplementedError("DatabaseAuditStorage not implemented")

    def get_last_hash(self) -> str | None:
        """Get last hash."""
        raise NotImplementedError("DatabaseAuditStorage not implemented")


class AuditLogger:
    """
    Main audit logger for the trading system.

    Provides high-level methods for logging different types of
    trading events with automatic context enrichment.
    """

    def __init__(
        self,
        storage: AuditStorage,
        source: str = "trading_system",
        auto_session: bool = True,
    ):
        """Initialize audit logger.

        Args:
            storage: Audit storage backend.
            source: Default source identifier.
            auto_session: Auto-generate session ID.
        """
        self.storage = storage
        self.source = source
        self.session_id = str(uuid.uuid4()) if auto_session else None
        self._user_id: str | None = None

        # Callbacks for event notifications
        self._callbacks: list[Callable[[AuditEvent], None]] = []

    def set_user(self, user_id: str | None) -> None:
        """Set the current user ID."""
        self._user_id = user_id

    def add_callback(self, callback: Callable[[AuditEvent], None]) -> None:
        """Add callback for event notifications."""
        self._callbacks.append(callback)

    def _log_event(
        self,
        event_type: AuditEventType,
        action: str,
        details: dict[str, Any],
        severity: AuditSeverity = AuditSeverity.INFO,
        outcome: str = "success",
        source: str | None = None,
    ) -> AuditEvent:
        """Log an audit event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            source=source or self.source,
            user_id=self._user_id,
            session_id=self.session_id,
            action=action,
            details=details,
            outcome=outcome,
            previous_hash=self.storage.get_last_hash(),
        )

        self.storage.store(event)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Audit callback failed: {e}")

        return event

    # Trading events

    def log_order_created(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: float | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log order creation."""
        return self._log_event(
            event_type=AuditEventType.ORDER_CREATED,
            action=f"Create {side} order for {symbol}",
            details={
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
                **kwargs,
            },
        )

    def log_order_submitted(
        self,
        order_id: str,
        broker_order_id: str | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log order submission to broker."""
        return self._log_event(
            event_type=AuditEventType.ORDER_SUBMITTED,
            action=f"Submit order {order_id} to broker",
            details={
                "order_id": order_id,
                "broker_order_id": broker_order_id,
                **kwargs,
            },
        )

    def log_order_filled(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: float,
        commission: float = 0.0,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log order fill."""
        return self._log_event(
            event_type=AuditEventType.ORDER_FILLED,
            action=f"Order {order_id} filled",
            details={
                "order_id": order_id,
                "fill_price": fill_price,
                "fill_quantity": fill_quantity,
                "commission": commission,
                **kwargs,
            },
        )

    def log_order_cancelled(
        self,
        order_id: str,
        reason: str = "",
        **kwargs: Any,
    ) -> AuditEvent:
        """Log order cancellation."""
        return self._log_event(
            event_type=AuditEventType.ORDER_CANCELLED,
            action=f"Cancel order {order_id}",
            details={
                "order_id": order_id,
                "reason": reason,
                **kwargs,
            },
        )

    def log_order_rejected(
        self,
        order_id: str,
        reason: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log order rejection."""
        return self._log_event(
            event_type=AuditEventType.ORDER_REJECTED,
            action=f"Order {order_id} rejected",
            details={
                "order_id": order_id,
                "reason": reason,
                **kwargs,
            },
            severity=AuditSeverity.WARNING,
            outcome="failure",
        )

    # Risk events

    def log_risk_check(
        self,
        check_name: str,
        passed: bool,
        details: dict[str, Any],
        order_id: str | None = None,
    ) -> AuditEvent:
        """Log risk check result."""
        return self._log_event(
            event_type=(
                AuditEventType.RISK_CHECK_PASSED
                if passed
                else AuditEventType.RISK_CHECK_FAILED
            ),
            action=f"Risk check: {check_name}",
            details={
                "check_name": check_name,
                "passed": passed,
                "order_id": order_id,
                **details,
            },
            severity=AuditSeverity.INFO if passed else AuditSeverity.WARNING,
            outcome="success" if passed else "failure",
        )

    def log_risk_limit_breach(
        self,
        limit_name: str,
        current_value: float,
        limit_value: float,
        action_taken: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log risk limit breach."""
        return self._log_event(
            event_type=AuditEventType.RISK_LIMIT_BREACH,
            action=f"Risk limit breach: {limit_name}",
            details={
                "limit_name": limit_name,
                "current_value": current_value,
                "limit_value": limit_value,
                "action_taken": action_taken,
                **kwargs,
            },
            severity=AuditSeverity.CRITICAL,
            outcome="failure",
        )

    def log_kill_switch_activated(
        self,
        reason: str,
        triggered_by: str,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log kill switch activation."""
        return self._log_event(
            event_type=AuditEventType.KILL_SWITCH_ACTIVATED,
            action="Kill switch activated",
            details={
                "reason": reason,
                "triggered_by": triggered_by,
                **kwargs,
            },
            severity=AuditSeverity.CRITICAL,
        )

    def log_kill_switch_reset(
        self,
        reset_by: str,
        authorization_code: str | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log kill switch reset."""
        return self._log_event(
            event_type=AuditEventType.KILL_SWITCH_RESET,
            action="Kill switch reset",
            details={
                "reset_by": reset_by,
                "authorization_code": authorization_code,
                **kwargs,
            },
            severity=AuditSeverity.WARNING,
        )

    # Model events

    def log_model_prediction(
        self,
        model_name: str,
        model_version: str,
        prediction: Any,
        input_features: dict[str, Any] | None = None,
        confidence: float | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log model prediction."""
        return self._log_event(
            event_type=AuditEventType.MODEL_PREDICTION,
            action=f"Model prediction: {model_name}",
            details={
                "model_name": model_name,
                "model_version": model_version,
                "prediction": prediction,
                "input_features": input_features,
                "confidence": confidence,
                **kwargs,
            },
            source=f"model:{model_name}",
        )

    def log_model_deployed(
        self,
        model_name: str,
        model_version: str,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log model deployment."""
        return self._log_event(
            event_type=AuditEventType.MODEL_DEPLOYED,
            action=f"Deploy model: {model_name} v{model_version}",
            details={
                "model_name": model_name,
                "model_version": model_version,
                "metrics": metrics,
                **kwargs,
            },
        )

    # Signal events

    def log_signal_generated(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        strength: float,
        source_model: str | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log signal generation."""
        return self._log_event(
            event_type=AuditEventType.SIGNAL_GENERATED,
            action=f"Generate signal for {symbol}",
            details={
                "signal_id": signal_id,
                "symbol": symbol,
                "direction": direction,
                "strength": strength,
                "source_model": source_model,
                **kwargs,
            },
        )

    # System events

    def log_system_startup(
        self,
        version: str,
        configuration: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log system startup."""
        return self._log_event(
            event_type=AuditEventType.SYSTEM_STARTUP,
            action="System startup",
            details={
                "version": version,
                "configuration": configuration,
                **kwargs,
            },
        )

    def log_system_shutdown(
        self,
        reason: str = "normal",
        **kwargs: Any,
    ) -> AuditEvent:
        """Log system shutdown."""
        return self._log_event(
            event_type=AuditEventType.SYSTEM_SHUTDOWN,
            action="System shutdown",
            details={
                "reason": reason,
                **kwargs,
            },
        )

    def log_configuration_changed(
        self,
        config_key: str,
        old_value: Any,
        new_value: Any,
        changed_by: str | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log configuration change."""
        return self._log_event(
            event_type=AuditEventType.CONFIGURATION_CHANGED,
            action=f"Configuration changed: {config_key}",
            details={
                "config_key": config_key,
                "old_value": old_value,
                "new_value": new_value,
                "changed_by": changed_by,
                **kwargs,
            },
            severity=AuditSeverity.WARNING,
        )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        stack_trace: str | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log error occurrence."""
        return self._log_event(
            event_type=AuditEventType.ERROR_OCCURRED,
            action=f"Error: {error_type}",
            details={
                "error_type": error_type,
                "error_message": error_message,
                "stack_trace": stack_trace,
                "context": context,
                **kwargs,
            },
            severity=AuditSeverity.ERROR,
            outcome="failure",
        )

    # User events

    def log_user_login(
        self,
        user_id: str,
        ip_address: str | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log user login."""
        self._user_id = user_id
        return self._log_event(
            event_type=AuditEventType.USER_LOGIN,
            action=f"User login: {user_id}",
            details={
                "ip_address": ip_address,
                **kwargs,
            },
        )

    def log_user_action(
        self,
        action: str,
        resource: str,
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuditEvent:
        """Log user action."""
        return self._log_event(
            event_type=AuditEventType.USER_ACTION,
            action=f"User action: {action} on {resource}",
            details={
                "resource": resource,
                **(details or {}),
                **kwargs,
            },
        )


class AuditQuery:
    """Query builder for audit log retrieval."""

    def __init__(self, storage: AuditStorage):
        """Initialize query builder."""
        self.storage = storage
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None
        self._event_types: list[AuditEventType] | None = None
        self._limit: int = 1000

    def since(self, start_time: datetime) -> "AuditQuery":
        """Filter events after start time."""
        self._start_time = start_time
        return self

    def until(self, end_time: datetime) -> "AuditQuery":
        """Filter events before end time."""
        self._end_time = end_time
        return self

    def of_type(self, *event_types: AuditEventType) -> "AuditQuery":
        """Filter by event types."""
        self._event_types = list(event_types)
        return self

    def limit(self, n: int) -> "AuditQuery":
        """Limit number of results."""
        self._limit = n
        return self

    def execute(self) -> list[AuditEvent]:
        """Execute the query."""
        return self.storage.retrieve(
            start_time=self._start_time,
            end_time=self._end_time,
            event_types=self._event_types,
            limit=self._limit,
        )


def create_audit_logger(
    storage_dir: Path | str | None = None,
    source: str = "trading_system",
) -> AuditLogger:
    """Factory function to create audit logger.

    Args:
        storage_dir: Directory for audit logs (default: ./audit_logs).
        source: Source identifier for events.

    Returns:
        Configured AuditLogger instance.
    """
    storage_dir = storage_dir or Path("audit_logs")
    storage = FileAuditStorage(storage_dir=storage_dir)
    return AuditLogger(storage=storage, source=source)
