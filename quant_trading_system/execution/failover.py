"""
Broker failover orchestration for execution resilience.

Provides an Alpaca-compatible broker facade with:
- Active/passive broker failover
- Per-broker health tracking and cooldown quarantine
- Order-to-broker routing memory for safe follow-up operations
- Best-effort callback fan-in for trade update streams
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, TypeVar

from quant_trading_system.core.exceptions import (
    BrokerConnectionError,
    InsufficientFundsError,
    OrderCancellationError,
    OrderSubmissionError,
)
from quant_trading_system.monitoring.audit import AuditEventType

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class BrokerEndpoint:
    """Static broker endpoint metadata."""

    name: str
    client: Any
    priority: int = 0
    enabled: bool = True
    is_primary: bool = False


@dataclass
class BrokerFailoverPolicy:
    """Failover thresholds and recovery policy."""

    max_consecutive_failures: int = 3
    recovery_cooldown_seconds: int = 120
    failover_on_order_submission_error: bool = False


@dataclass
class BrokerRuntimeState:
    """Mutable runtime health state for one broker."""

    consecutive_failures: int = 0
    last_failure_at: datetime | None = None
    last_success_at: datetime | None = None
    quarantined_until: datetime | None = None
    last_error: str | None = None

    def is_quarantined(self) -> bool:
        """Return true if broker is still in cooldown quarantine window."""
        if self.quarantined_until is None:
            return False
        return datetime.now(timezone.utc) < self.quarantined_until


class FailoverBrokerClient:
    """
    Broker wrapper with active/passive failover semantics.

    The wrapper mirrors the subset of AlpacaClient methods consumed by the
    execution and trading layers. It can operate with one or many broker
    clients as long as each broker exposes the same async methods.
    """

    def __init__(
        self,
        endpoints: list[BrokerEndpoint],
        policy: BrokerFailoverPolicy | None = None,
        metrics_collector: Any | None = None,
        audit_logger: Any | None = None,
    ) -> None:
        if not endpoints:
            raise ValueError("FailoverBrokerClient requires at least one broker endpoint")

        ordered = sorted(
            [endpoint for endpoint in endpoints if endpoint.enabled],
            key=lambda endpoint: endpoint.priority,
        )
        if not ordered:
            raise ValueError("No enabled broker endpoints available")

        self._endpoints = ordered
        self._policy = policy or BrokerFailoverPolicy()
        self._metrics_collector = metrics_collector
        self._audit_logger = audit_logger
        self._lock = threading.RLock()

        # If none marked primary, the first endpoint is the primary.
        if not any(endpoint.is_primary for endpoint in self._endpoints):
            self._endpoints[0].is_primary = True

        self._runtime: dict[str, BrokerRuntimeState] = {
            endpoint.name: BrokerRuntimeState() for endpoint in self._endpoints
        }
        self._active_index = 0

        # Routing memory to ensure follow-up operations hit the original broker.
        self._client_order_to_broker: dict[str, str] = {}
        self._broker_order_to_broker: dict[str, str] = {}

        self._trade_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._register_upstream_trade_callbacks()

        logger.info(
            "FailoverBrokerClient initialized with brokers=%s",
            [endpoint.name for endpoint in self._endpoints],
        )

    @property
    def active_broker_name(self) -> str:
        """Current active broker endpoint name."""
        with self._lock:
            return self._endpoints[self._active_index].name

    def get_failover_status(self) -> dict[str, Any]:
        """Return active/primary status and broker runtime health snapshots."""
        with self._lock:
            primary_name = next(
                endpoint.name for endpoint in self._endpoints if endpoint.is_primary
            )
            return {
                "active_broker": self.active_broker_name,
                "primary_broker": primary_name,
                "brokers": {
                    endpoint.name: {
                        "is_primary": endpoint.is_primary,
                        "enabled": endpoint.enabled,
                        "consecutive_failures": self._runtime[endpoint.name].consecutive_failures,
                        "last_failure_at": self._iso(self._runtime[endpoint.name].last_failure_at),
                        "last_success_at": self._iso(self._runtime[endpoint.name].last_success_at),
                        "quarantined_until": self._iso(self._runtime[endpoint.name].quarantined_until),
                        "last_error": self._runtime[endpoint.name].last_error,
                    }
                    for endpoint in self._endpoints
                },
            }

    async def submit_order(self, **kwargs: Any) -> Any:
        """Submit order and fail over on transient broker faults."""
        order, broker_name = await self._call_with_failover(
            operation="submit_order",
            invoker=lambda client: client.submit_order(**kwargs),
            allow_failover=True,
        )
        self._remember_order_mapping(order=order, broker_name=broker_name, kwargs=kwargs)
        return order

    async def get_order(self, order_id: str, by_client_id: bool = False) -> Any:
        """Fetch order details, preferring the original broker if mapped."""
        preferred = self._resolve_order_broker(order_id, by_client_id=by_client_id)
        order, broker_name = await self._call_with_failover(
            operation="get_order",
            invoker=lambda client: client.get_order(order_id, by_client_id=by_client_id),
            allow_failover=preferred is None,
            preferred_broker=preferred,
        )
        self._remember_order_mapping(order=order, broker_name=broker_name, kwargs={})
        return order

    async def cancel_order(self, order_id: str) -> None:
        """Cancel order on its owning broker (or active broker if unknown)."""
        preferred = self._resolve_order_broker(order_id, by_client_id=False)
        await self._call_with_failover(
            operation="cancel_order",
            invoker=lambda client: client.cancel_order(order_id),
            allow_failover=preferred is None,
            preferred_broker=preferred,
        )

    async def replace_order(
        self,
        order_id: str,
        qty: Any = None,
        limit_price: Any = None,
        stop_price: Any = None,
        time_in_force: Any = None,
        client_order_id: str | None = None,
    ) -> Any:
        """Replace order on its owning broker."""
        preferred = self._resolve_order_broker(order_id, by_client_id=False)
        order, broker_name = await self._call_with_failover(
            operation="replace_order",
            invoker=lambda client: client.replace_order(
                order_id=order_id,
                qty=qty,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                client_order_id=client_order_id,
            ),
            allow_failover=preferred is None,
            preferred_broker=preferred,
        )
        self._remember_order_mapping(order=order, broker_name=broker_name, kwargs={})
        return order

    async def get_orders(self, **kwargs: Any) -> Any:
        """List orders from the active healthy broker."""
        orders, _ = await self._call_with_failover(
            operation="get_orders",
            invoker=lambda client: client.get_orders(**kwargs),
            allow_failover=True,
        )
        return orders

    async def get_account(self) -> Any:
        """Get account snapshot from active healthy broker."""
        account, _ = await self._call_with_failover(
            operation="get_account",
            invoker=lambda client: client.get_account(),
            allow_failover=True,
        )
        return account

    async def get_positions(self) -> Any:
        """Get open positions from active healthy broker."""
        positions, _ = await self._call_with_failover(
            operation="get_positions",
            invoker=lambda client: client.get_positions(),
            allow_failover=True,
        )
        return positions

    async def list_positions(self) -> Any:
        """Backward-compatible alias for callers expecting list_positions()."""
        return await self.get_positions()

    async def connect(self) -> None:
        """Best-effort broker connection warmup."""
        for endpoint in self._endpoints:
            connect = getattr(endpoint.client, "connect", None)
            if connect is None:
                continue
            try:
                await connect()
                self._mark_success(endpoint.name)
            except Exception as exc:
                self._mark_failure(endpoint.name, exc, operation="connect")

    async def disconnect(self) -> None:
        """Best-effort broker connection teardown."""
        for endpoint in self._endpoints:
            disconnect = getattr(endpoint.client, "disconnect", None)
            if disconnect is None:
                continue
            try:
                await disconnect()
            except Exception as exc:
                logger.warning("Broker disconnect failed (%s): %s", endpoint.name, exc)

    async def get_clock(self) -> Any:
        """Get market clock from active healthy broker."""
        clock, _ = await self._call_with_failover(
            operation="get_clock",
            invoker=lambda client: client.get_clock(),
            allow_failover=True,
        )
        return clock

    async def is_market_open(self) -> bool:
        """Get market open state from active healthy broker."""
        is_open, _ = await self._call_with_failover(
            operation="is_market_open",
            invoker=lambda client: client.is_market_open(),
            allow_failover=True,
        )
        return bool(is_open)

    def on_trade_update(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register callback for aggregated trade updates."""
        self._trade_callbacks.append(callback)

    async def start_trade_stream(self) -> None:
        """Start trade streams on all brokers (best effort)."""
        for endpoint in self._endpoints:
            start = getattr(endpoint.client, "start_trade_stream", None)
            if start is None:
                continue
            try:
                asyncio.create_task(start())
            except Exception as exc:
                logger.warning("Failed to start trade stream (%s): %s", endpoint.name, exc)

    def stop_trade_stream(self) -> None:
        """Stop trade streams on all brokers (best effort)."""
        for endpoint in self._endpoints:
            stop = getattr(endpoint.client, "stop_trade_stream", None)
            if stop is None:
                continue
            try:
                stop()
            except Exception as exc:
                logger.warning("Failed to stop trade stream (%s): %s", endpoint.name, exc)

    async def _call_with_failover(
        self,
        operation: str,
        invoker: Callable[[Any], Awaitable[T]],
        allow_failover: bool,
        preferred_broker: str | None = None,
    ) -> tuple[T, str]:
        """Execute operation against brokers with optional failover retries."""
        errors: list[str] = []
        last_exception: Exception | None = None

        for endpoint in self._iter_candidate_endpoints(preferred_broker=preferred_broker):
            try:
                result = await invoker(endpoint.client)
                self._mark_success(endpoint.name)
                self._promote_active(endpoint.name, reason=f"{operation}_success")
                return result, endpoint.name
            except Exception as exc:
                last_exception = exc
                errors.append(f"{endpoint.name}: {exc}")

                transient = self._is_transient_failure(operation, exc)
                self._mark_failure(endpoint.name, exc, operation=operation)

                if not allow_failover or not transient:
                    raise

        raise BrokerConnectionError(
            f"All broker endpoints failed for {operation}. Errors: {' | '.join(errors)}",
            broker=self.active_broker_name,
            details={"operation": operation, "last_error": str(last_exception)},
        )

    def _iter_candidate_endpoints(self, preferred_broker: str | None = None) -> list[BrokerEndpoint]:
        """Build ordered endpoint list with active broker first and quarantine awareness."""
        with self._lock:
            candidates: list[BrokerEndpoint] = []

            if preferred_broker is not None:
                preferred = self._get_endpoint(preferred_broker)
                if preferred is not None:
                    candidates.append(preferred)

            active = self._endpoints[self._active_index]
            if active not in candidates:
                candidates.append(active)

            for endpoint in self._endpoints:
                if endpoint not in candidates:
                    candidates.append(endpoint)

            healthy_candidates = [
                endpoint for endpoint in candidates if not self._runtime[endpoint.name].is_quarantined()
            ]
            if healthy_candidates:
                return healthy_candidates
            return candidates

    def _is_transient_failure(self, operation: str, exc: Exception) -> bool:
        """Decide whether this failure should trigger failover."""
        if isinstance(exc, InsufficientFundsError):
            return False

        if isinstance(exc, OrderCancellationError):
            return False

        if isinstance(exc, OrderSubmissionError):
            return self._policy.failover_on_order_submission_error

        if isinstance(exc, BrokerConnectionError):
            return True

        if isinstance(exc, (TimeoutError, ConnectionError, OSError, asyncio.TimeoutError)):
            return True

        if "timeout" in str(exc).lower():
            return True

        # Be conservative for unknown failures in state-mutating operations.
        return operation in {"get_account", "get_orders", "get_positions", "get_clock", "is_market_open"}

    def _mark_success(self, broker_name: str) -> None:
        """Mark successful broker interaction and clear failure streak."""
        with self._lock:
            state = self._runtime[broker_name]
            state.consecutive_failures = 0
            state.last_success_at = datetime.now(timezone.utc)
            state.last_error = None
            if state.quarantined_until and state.quarantined_until <= datetime.now(timezone.utc):
                state.quarantined_until = None

    def _mark_failure(self, broker_name: str, exc: Exception, operation: str) -> None:
        """Record broker failure and quarantine if threshold exceeded."""
        with self._lock:
            state = self._runtime[broker_name]
            state.consecutive_failures += 1
            state.last_failure_at = datetime.now(timezone.utc)
            state.last_error = str(exc)

            if state.consecutive_failures >= self._policy.max_consecutive_failures:
                state.quarantined_until = datetime.now(timezone.utc) + timedelta(
                    seconds=self._policy.recovery_cooldown_seconds
                )
                self._promote_next_healthy(reason=f"{operation}_failure")

        self._record_metric("record_system_error", f"broker_{broker_name}_{operation}")
        self._audit(
            AuditEventType.ERROR_OCCURRED,
            action="broker_operation_failed",
            outcome="failure",
            details={
                "broker": broker_name,
                "operation": operation,
                "error": str(exc),
                "active_broker": self.active_broker_name,
            },
        )

    def _promote_next_healthy(self, reason: str) -> None:
        """Switch active broker to next non-quarantined endpoint if available."""
        current_name = self._endpoints[self._active_index].name
        for index, endpoint in enumerate(self._endpoints):
            if index == self._active_index:
                continue
            if not endpoint.enabled:
                continue
            if self._runtime[endpoint.name].is_quarantined():
                continue
            self._active_index = index
            logger.warning(
                "Broker failover activated: %s -> %s (%s)",
                current_name,
                endpoint.name,
                reason,
            )
            self._record_metric("record_system_error", "broker_failover")
            self._audit(
                AuditEventType.CONFIGURATION_CHANGED,
                action="broker_failover",
                outcome="success",
                details={"from": current_name, "to": endpoint.name, "reason": reason},
            )
            return

    def _promote_active(self, broker_name: str, reason: str) -> None:
        """Promote successful broker as active endpoint."""
        with self._lock:
            for index, endpoint in enumerate(self._endpoints):
                if endpoint.name == broker_name:
                    if index != self._active_index:
                        previous = self._endpoints[self._active_index].name
                        self._active_index = index
                        logger.info(
                            "Broker active endpoint updated: %s -> %s (%s)",
                            previous,
                            broker_name,
                            reason,
                        )
                    return

    def _register_upstream_trade_callbacks(self) -> None:
        """Subscribe to all upstream broker trade streams."""
        for endpoint in self._endpoints:
            registrar = getattr(endpoint.client, "on_trade_update", None)
            if registrar is None:
                continue
            registrar(self._build_trade_update_handler(endpoint.name))

    def _build_trade_update_handler(self, broker_name: str) -> Callable[[dict[str, Any]], None]:
        """Create callback wrapper that captures broker origin for updates."""

        def _handler(data: dict[str, Any]) -> None:
            self._remember_order_mapping_from_stream(broker_name, data)
            for callback in self._trade_callbacks:
                try:
                    callback(data)
                except Exception as exc:
                    logger.error("Trade callback error (%s): %s", broker_name, exc)

        return _handler

    def _remember_order_mapping_from_stream(self, broker_name: str, data: dict[str, Any]) -> None:
        """Persist broker routing keys from trade stream payload."""
        order_data = data.get("order", {})
        if not isinstance(order_data, dict):
            return

        broker_order_id = order_data.get("id")
        client_order_id = order_data.get("client_order_id")

        with self._lock:
            if isinstance(broker_order_id, str) and broker_order_id:
                self._broker_order_to_broker[broker_order_id] = broker_name
            if isinstance(client_order_id, str) and client_order_id:
                self._client_order_to_broker[client_order_id] = broker_name

    def _remember_order_mapping(self, order: Any, broker_name: str, kwargs: dict[str, Any]) -> None:
        """Persist broker routing keys from API responses."""
        broker_order_id = getattr(order, "order_id", None)
        client_order_id = getattr(order, "client_order_id", None)
        if client_order_id is None:
            client_order_id = kwargs.get("client_order_id")

        with self._lock:
            if isinstance(broker_order_id, str) and broker_order_id:
                self._broker_order_to_broker[broker_order_id] = broker_name
            if isinstance(client_order_id, str) and client_order_id:
                self._client_order_to_broker[client_order_id] = broker_name

    def _resolve_order_broker(self, order_id: str, by_client_id: bool) -> str | None:
        """Resolve owning broker for an order identifier."""
        with self._lock:
            if by_client_id:
                return self._client_order_to_broker.get(order_id)
            return self._broker_order_to_broker.get(order_id)

    def _get_endpoint(self, broker_name: str) -> BrokerEndpoint | None:
        """Lookup endpoint by name."""
        for endpoint in self._endpoints:
            if endpoint.name == broker_name:
                return endpoint
        return None

    def _record_metric(self, method_name: str, *args: Any) -> None:
        """Best-effort metrics hooks."""
        if self._metrics_collector is None:
            return
        recorder = getattr(self._metrics_collector, method_name, None)
        if recorder is None:
            return
        try:
            recorder(*args)
        except Exception:
            return

    def _audit(
        self,
        event_type: AuditEventType,
        action: str,
        outcome: str,
        details: dict[str, Any],
    ) -> None:
        """Best-effort audit hooks."""
        if self._audit_logger is None:
            return
        try:
            self._audit_logger.log(event_type, {"action": action, "outcome": outcome, **details})
        except Exception:
            return

    @staticmethod
    def _iso(value: datetime | None) -> str | None:
        """Serialize datetime to ISO-8601 string."""
        if value is None:
            return None
        return value.isoformat()
