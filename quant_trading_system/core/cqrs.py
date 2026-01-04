"""
CQRS (Command Query Responsibility Segregation) pattern implementation.

Separates read and write operations for better scalability and performance:
- Commands: Modify state (create order, update position)
- Queries: Read state (get portfolio, list orders)

Benefits:
- Optimized read/write paths
- Better scalability (scale reads independently)
- Clearer separation of concerns
- Event sourcing compatibility

Usage:
    # Define a command
    @dataclass
    class PlaceOrderCommand(Command):
        symbol: str
        side: OrderSide
        quantity: Decimal

    # Define command handler
    class PlaceOrderHandler(CommandHandler[PlaceOrderCommand]):
        async def handle(self, command: PlaceOrderCommand) -> CommandResult:
            # Execute the order placement
            ...

    # Define a query
    @dataclass
    class GetPortfolioQuery(Query):
        pass

    # Define query handler
    class GetPortfolioHandler(QueryHandler[GetPortfolioQuery, Portfolio]):
        async def handle(self, query: GetPortfolioQuery) -> Portfolio:
            # Return portfolio data
            ...

    # Use the bus
    command_bus = CommandBus()
    query_bus = QueryBus()

    result = await command_bus.dispatch(PlaceOrderCommand(...))
    portfolio = await query_bus.dispatch(GetPortfolioQuery())
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Generic, Type, TypeVar
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# BASE TYPES
# =============================================================================


class MessageType(str, Enum):
    """Type of message in CQRS system."""

    COMMAND = "command"
    QUERY = "query"
    EVENT = "event"


@dataclass(kw_only=True)
class Message:
    """Base class for all CQRS messages."""

    message_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message_type(self) -> MessageType:
        """Return the type of message."""
        return MessageType.COMMAND


# =============================================================================
# COMMANDS
# =============================================================================


@dataclass
class Command(Message):
    """
    Base class for commands.

    Commands represent intent to change system state.
    They should be named as imperative verbs (PlaceOrder, CancelOrder, etc.)
    """

    @property
    def message_type(self) -> MessageType:
        return MessageType.COMMAND


class CommandStatus(str, Enum):
    """Status of command execution."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class CommandResult:
    """Result of command execution."""

    command_id: UUID
    status: CommandStatus
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == CommandStatus.COMPLETED

    @property
    def is_failure(self) -> bool:
        return self.status in (CommandStatus.FAILED, CommandStatus.REJECTED)


C = TypeVar("C", bound=Command)


class CommandHandler(ABC, Generic[C]):
    """
    Base class for command handlers.

    Each command type should have exactly one handler.
    """

    @abstractmethod
    async def handle(self, command: C) -> CommandResult:
        """Handle the command and return result."""
        pass


class CommandMiddleware(ABC):
    """Middleware for command processing pipeline."""

    @abstractmethod
    async def process(
        self,
        command: Command,
        next_handler: Callable[[Command], Any],
    ) -> CommandResult:
        """Process command, optionally calling next handler."""
        pass


class LoggingCommandMiddleware(CommandMiddleware):
    """Log command execution."""

    async def process(
        self,
        command: Command,
        next_handler: Callable[[Command], Any],
    ) -> CommandResult:
        start_time = time.time()
        logger.info(f"Executing command: {type(command).__name__} id={command.message_id}")

        try:
            result = await next_handler(command)
            duration = (time.time() - start_time) * 1000
            logger.info(
                f"Command completed: {type(command).__name__} "
                f"status={result.status.value} duration={duration:.2f}ms"
            )
            return result
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(
                f"Command failed: {type(command).__name__} "
                f"error={e} duration={duration:.2f}ms"
            )
            raise


class ValidationCommandMiddleware(CommandMiddleware):
    """Validate commands before execution."""

    def __init__(self, validators: dict[Type[Command], Callable[[Command], bool]] | None = None):
        self._validators = validators or {}

    def register_validator(
        self,
        command_type: Type[Command],
        validator: Callable[[Command], bool],
    ) -> None:
        """Register a validator for a command type."""
        self._validators[command_type] = validator

    async def process(
        self,
        command: Command,
        next_handler: Callable[[Command], Any],
    ) -> CommandResult:
        command_type = type(command)
        if command_type in self._validators:
            validator = self._validators[command_type]
            if not validator(command):
                return CommandResult(
                    command_id=command.message_id,
                    status=CommandStatus.REJECTED,
                    error="Command validation failed",
                )

        return await next_handler(command)


class CommandBus:
    """
    Dispatches commands to their handlers.

    Features:
    - Handler registration
    - Middleware pipeline
    - Async execution
    - Error handling
    """

    def __init__(self) -> None:
        self._handlers: dict[Type[Command], CommandHandler] = {}
        self._middleware: list[CommandMiddleware] = []
        self._lock = threading.RLock()

    def register_handler(
        self,
        command_type: Type[C],
        handler: CommandHandler[C],
    ) -> None:
        """Register a handler for a command type."""
        with self._lock:
            if command_type in self._handlers:
                logger.warning(f"Replacing handler for {command_type.__name__}")
            self._handlers[command_type] = handler

    def add_middleware(self, middleware: CommandMiddleware) -> None:
        """Add middleware to the processing pipeline."""
        with self._lock:
            self._middleware.append(middleware)

    async def dispatch(self, command: Command) -> CommandResult:
        """Dispatch a command to its handler."""
        command_type = type(command)
        start_time = time.time()

        with self._lock:
            handler = self._handlers.get(command_type)
            middleware = list(self._middleware)

        if handler is None:
            return CommandResult(
                command_id=command.message_id,
                status=CommandStatus.REJECTED,
                error=f"No handler registered for {command_type.__name__}",
            )

        # Build middleware chain
        async def execute(cmd: Command) -> CommandResult:
            try:
                result = await handler.handle(cmd)
                result.execution_time_ms = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                return CommandResult(
                    command_id=cmd.message_id,
                    status=CommandStatus.FAILED,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

        # Apply middleware in reverse order
        next_handler = execute
        for mw in reversed(middleware):
            current_mw = mw
            prev_handler = next_handler

            async def wrapped(cmd: Command, mw=current_mw, handler=prev_handler) -> CommandResult:
                return await mw.process(cmd, handler)

            next_handler = wrapped

        return await next_handler(command)

    def get_handler(self, command_type: Type[Command]) -> CommandHandler | None:
        """Get handler for a command type."""
        with self._lock:
            return self._handlers.get(command_type)


# =============================================================================
# QUERIES
# =============================================================================


@dataclass
class Query(Message):
    """
    Base class for queries.

    Queries represent requests to read data without side effects.
    They should be named as questions (GetPortfolio, ListOrders, etc.)
    """

    @property
    def message_type(self) -> MessageType:
        return MessageType.QUERY


Q = TypeVar("Q", bound=Query)
R = TypeVar("R")


class QueryHandler(ABC, Generic[Q, R]):
    """
    Base class for query handlers.

    Returns the result type R for query type Q.
    """

    @abstractmethod
    async def handle(self, query: Q) -> R:
        """Handle the query and return result."""
        pass


class QueryMiddleware(ABC):
    """Middleware for query processing pipeline."""

    @abstractmethod
    async def process(
        self,
        query: Query,
        next_handler: Callable[[Query], Any],
    ) -> Any:
        """Process query, optionally calling next handler."""
        pass


class CachingQueryMiddleware(QueryMiddleware):
    """Cache query results."""

    def __init__(self, ttl_seconds: float = 60.0, max_size: int = 1000):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._lock = threading.RLock()

    def _make_key(self, query: Query) -> str:
        """Create cache key from query."""
        query_type = type(query).__name__
        # Create deterministic key from query attributes
        attrs = {k: v for k, v in query.__dict__.items() if not k.startswith('_')}
        return f"{query_type}:{hash(str(sorted(attrs.items())))}"

    async def process(
        self,
        query: Query,
        next_handler: Callable[[Query], Any],
    ) -> Any:
        key = self._make_key(query)

        with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    logger.debug(f"Cache hit for query: {type(query).__name__}")
                    return result
                else:
                    del self._cache[key]

        # Execute query
        result = await next_handler(query)

        # Cache result
        with self._lock:
            if len(self._cache) >= self._max_size:
                # Evict oldest entries
                oldest = sorted(self._cache.items(), key=lambda x: x[1][1])[:100]
                for k, _ in oldest:
                    del self._cache[k]
            self._cache[key] = (result, time.time())

        return result


class QueryBus:
    """
    Dispatches queries to their handlers.

    Features:
    - Handler registration
    - Middleware pipeline (caching, logging)
    - Async execution
    """

    def __init__(self) -> None:
        self._handlers: dict[Type[Query], QueryHandler] = {}
        self._middleware: list[QueryMiddleware] = []
        self._lock = threading.RLock()

    def register_handler(
        self,
        query_type: Type[Q],
        handler: QueryHandler[Q, R],
    ) -> None:
        """Register a handler for a query type."""
        with self._lock:
            if query_type in self._handlers:
                logger.warning(f"Replacing handler for {query_type.__name__}")
            self._handlers[query_type] = handler

    def add_middleware(self, middleware: QueryMiddleware) -> None:
        """Add middleware to the processing pipeline."""
        with self._lock:
            self._middleware.append(middleware)

    async def dispatch(self, query: Q) -> R:
        """Dispatch a query to its handler."""
        query_type = type(query)

        with self._lock:
            handler = self._handlers.get(query_type)
            middleware = list(self._middleware)

        if handler is None:
            raise ValueError(f"No handler registered for {query_type.__name__}")

        # Build middleware chain
        async def execute(q: Query) -> Any:
            return await handler.handle(q)

        next_handler = execute
        for mw in reversed(middleware):
            current_mw = mw
            prev_handler = next_handler

            async def wrapped(q: Query, mw=current_mw, handler=prev_handler) -> Any:
                return await mw.process(q, handler)

            next_handler = wrapped

        return await next_handler(query)

    def get_handler(self, query_type: Type[Query]) -> QueryHandler | None:
        """Get handler for a query type."""
        with self._lock:
            return self._handlers.get(query_type)


# =============================================================================
# READ MODEL (for optimized queries)
# =============================================================================


class ReadModel(ABC):
    """
    Base class for read models.

    Read models are denormalized views optimized for specific queries.
    They are updated by event handlers when the write model changes.
    """

    @abstractmethod
    async def rebuild(self) -> None:
        """Rebuild the read model from events."""
        pass


class InMemoryReadModel(ReadModel):
    """In-memory read model for fast queries."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def get_all(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    async def rebuild(self) -> None:
        """Clear and rebuild from events - override in subclass."""
        with self._lock:
            self._data.clear()


# =============================================================================
# GLOBAL BUS REGISTRY
# =============================================================================


class CQRSBusRegistry:
    """
    Singleton registry for command and query buses.

    Provides global access to CQRS infrastructure.
    """

    _instance: CQRSBusRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> CQRSBusRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._command_bus = CommandBus()
        self._query_bus = QueryBus()

        # Add default middleware
        self._command_bus.add_middleware(LoggingCommandMiddleware())
        self._query_bus.add_middleware(CachingQueryMiddleware())

        self._initialized = True

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    @property
    def command_bus(self) -> CommandBus:
        return self._command_bus

    @property
    def query_bus(self) -> QueryBus:
        return self._query_bus


# Convenience functions


def get_command_bus() -> CommandBus:
    """Get the global command bus."""
    return CQRSBusRegistry().command_bus


def get_query_bus() -> QueryBus:
    """Get the global query bus."""
    return CQRSBusRegistry().query_bus
