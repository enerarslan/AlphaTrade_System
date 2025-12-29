"""
Data access patterns and repository implementations.

Provides a clean interface for database operations, abstracting
the SQLAlchemy session management and query building.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Generic, TypeVar
from uuid import UUID

from sqlalchemy import and_, desc, select
from sqlalchemy.orm import Session

from quant_trading_system.core.exceptions import DataNotFoundError

from .connection import DatabaseManager, get_db_manager
from .models import (
    Alert,
    Base,
    DailyPerformance,
    Feature,
    ModelPrediction,
    OHLCVBar,
    Order,
    Position,
    PositionHistory,
    Signal,
    SystemLog,
    Trade,
    TradeLog,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations.

    Provides a generic interface for database operations that can
    be extended by specific model repositories.
    """

    model: type[T]

    def __init__(self, db_manager: DatabaseManager | None = None) -> None:
        """Initialize repository.

        Args:
            db_manager: Database manager instance.
        """
        self._db = db_manager or get_db_manager()

    def create(self, session: Session, **kwargs: Any) -> T:
        """Create a new record.

        Args:
            session: Database session.
            **kwargs: Model field values.

        Returns:
            Created model instance.
        """
        instance = self.model(**kwargs)
        session.add(instance)
        session.flush()
        return instance

    def get_by_id(self, session: Session, id_value: Any) -> T | None:
        """Get a record by primary key.

        Args:
            session: Database session.
            id_value: Primary key value.

        Returns:
            Model instance or None.
        """
        return session.get(self.model, id_value)

    def get_all(self, session: Session, limit: int = 100) -> list[T]:
        """Get all records with optional limit.

        Args:
            session: Database session.
            limit: Maximum number of records.

        Returns:
            List of model instances.
        """
        stmt = select(self.model).limit(limit)
        return list(session.scalars(stmt).all())

    def update(self, session: Session, instance: T, **kwargs: Any) -> T:
        """Update a record.

        Args:
            session: Database session.
            instance: Model instance to update.
            **kwargs: Fields to update.

        Returns:
            Updated model instance.
        """
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        session.flush()
        return instance

    def delete(self, session: Session, instance: T) -> None:
        """Delete a record.

        Args:
            session: Database session.
            instance: Model instance to delete.
        """
        session.delete(instance)
        session.flush()


class OHLCVRepository(BaseRepository[OHLCVBar]):
    """Repository for OHLCV bar data operations."""

    model = OHLCVBar

    def get_bars(
        self,
        session: Session,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[OHLCVBar]:
        """Get OHLCV bars for a symbol within a time range.

        Args:
            session: Database session.
            symbol: Stock symbol.
            start_time: Start of time range.
            end_time: End of time range.
            limit: Maximum bars to return.

        Returns:
            List of OHLCV bars ordered by timestamp.
        """
        stmt = select(OHLCVBar).where(OHLCVBar.symbol == symbol.upper())

        if start_time:
            stmt = stmt.where(OHLCVBar.timestamp >= start_time)
        if end_time:
            stmt = stmt.where(OHLCVBar.timestamp <= end_time)

        stmt = stmt.order_by(OHLCVBar.timestamp).limit(limit)
        return list(session.scalars(stmt).all())

    def get_latest_bar(self, session: Session, symbol: str) -> OHLCVBar | None:
        """Get the most recent bar for a symbol.

        Args:
            session: Database session.
            symbol: Stock symbol.

        Returns:
            Most recent OHLCV bar or None.
        """
        stmt = (
            select(OHLCVBar)
            .where(OHLCVBar.symbol == symbol.upper())
            .order_by(desc(OHLCVBar.timestamp))
            .limit(1)
        )
        return session.scalars(stmt).first()

    def bulk_insert(self, session: Session, bars: list[dict[str, Any]]) -> int:
        """Bulk insert OHLCV bars.

        Args:
            session: Database session.
            bars: List of bar data dictionaries.

        Returns:
            Number of bars inserted.
        """
        instances = [OHLCVBar(**bar) for bar in bars]
        session.add_all(instances)
        session.flush()
        return len(instances)

    def get_symbols(self, session: Session) -> list[str]:
        """Get list of all symbols with data.

        Args:
            session: Database session.

        Returns:
            List of unique symbols.
        """
        stmt = select(OHLCVBar.symbol).distinct()
        return list(session.scalars(stmt).all())


class FeatureRepository(BaseRepository[Feature]):
    """Repository for feature data operations."""

    model = Feature

    def get_features(
        self,
        session: Session,
        symbol: str,
        feature_names: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Feature]:
        """Get features for a symbol.

        Args:
            session: Database session.
            symbol: Stock symbol.
            feature_names: Optional list of feature names to filter.
            start_time: Start of time range.
            end_time: End of time range.

        Returns:
            List of feature records.
        """
        stmt = select(Feature).where(Feature.symbol == symbol.upper())

        if feature_names:
            stmt = stmt.where(Feature.feature_name.in_(feature_names))
        if start_time:
            stmt = stmt.where(Feature.timestamp >= start_time)
        if end_time:
            stmt = stmt.where(Feature.timestamp <= end_time)

        stmt = stmt.order_by(Feature.timestamp, Feature.feature_name)
        return list(session.scalars(stmt).all())

    def get_latest_features(
        self,
        session: Session,
        symbol: str,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Get latest feature values for a symbol.

        Args:
            session: Database session.
            symbol: Stock symbol.
            feature_names: Optional list of feature names.

        Returns:
            Dictionary of feature name to value.
        """
        # Get the latest timestamp for this symbol
        subq = (
            select(Feature.timestamp)
            .where(Feature.symbol == symbol.upper())
            .order_by(desc(Feature.timestamp))
            .limit(1)
            .scalar_subquery()
        )

        stmt = select(Feature).where(
            and_(
                Feature.symbol == symbol.upper(),
                Feature.timestamp == subq,
            )
        )

        if feature_names:
            stmt = stmt.where(Feature.feature_name.in_(feature_names))

        features = session.scalars(stmt).all()
        return {f.feature_name: f.value for f in features}

    def bulk_insert(self, session: Session, features: list[dict[str, Any]]) -> int:
        """Bulk insert features.

        Args:
            session: Database session.
            features: List of feature data dictionaries.

        Returns:
            Number of features inserted.
        """
        instances = [Feature(**f) for f in features]
        session.add_all(instances)
        session.flush()
        return len(instances)


class OrderRepository(BaseRepository[Order]):
    """Repository for order operations."""

    model = Order

    def get_by_client_id(self, session: Session, client_order_id: str) -> Order | None:
        """Get order by client order ID.

        Args:
            session: Database session.
            client_order_id: Client-specified order ID.

        Returns:
            Order or None.
        """
        stmt = select(Order).where(Order.client_order_id == client_order_id)
        return session.scalars(stmt).first()

    def get_open_orders(self, session: Session, symbol: str | None = None) -> list[Order]:
        """Get all open orders.

        Args:
            session: Database session.
            symbol: Optional symbol filter.

        Returns:
            List of open orders.
        """
        open_statuses = ["PENDING", "SUBMITTED", "ACCEPTED", "PARTIAL_FILLED"]
        stmt = select(Order).where(Order.status.in_(open_statuses))

        if symbol:
            stmt = stmt.where(Order.symbol == symbol.upper())

        stmt = stmt.order_by(desc(Order.created_at))
        return list(session.scalars(stmt).all())

    def get_orders_by_status(
        self,
        session: Session,
        status: str,
        limit: int = 100,
    ) -> list[Order]:
        """Get orders by status.

        Args:
            session: Database session.
            status: Order status to filter by.
            limit: Maximum orders to return.

        Returns:
            List of orders.
        """
        stmt = (
            select(Order)
            .where(Order.status == status)
            .order_by(desc(Order.created_at))
            .limit(limit)
        )
        return list(session.scalars(stmt).all())

    def get_orders_by_symbol(
        self,
        session: Session,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[Order]:
        """Get orders for a symbol.

        Args:
            session: Database session.
            symbol: Stock symbol.
            start_time: Start of time range.
            end_time: End of time range.
            limit: Maximum orders to return.

        Returns:
            List of orders.
        """
        stmt = select(Order).where(Order.symbol == symbol.upper())

        if start_time:
            stmt = stmt.where(Order.created_at >= start_time)
        if end_time:
            stmt = stmt.where(Order.created_at <= end_time)

        stmt = stmt.order_by(desc(Order.created_at)).limit(limit)
        return list(session.scalars(stmt).all())


class TradeRepository(BaseRepository[Trade]):
    """Repository for trade operations."""

    model = Trade

    def get_trades_by_order(self, session: Session, order_id: str) -> list[Trade]:
        """Get all trades for an order.

        Args:
            session: Database session.
            order_id: Order ID.

        Returns:
            List of trades.
        """
        stmt = (
            select(Trade)
            .where(Trade.order_id == order_id)
            .order_by(Trade.executed_at)
        )
        return list(session.scalars(stmt).all())

    def get_trades_by_symbol(
        self,
        session: Session,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Trade]:
        """Get trades for a symbol.

        Args:
            session: Database session.
            symbol: Stock symbol.
            start_time: Start of time range.
            end_time: End of time range.

        Returns:
            List of trades.
        """
        stmt = select(Trade).where(Trade.symbol == symbol.upper())

        if start_time:
            stmt = stmt.where(Trade.executed_at >= start_time)
        if end_time:
            stmt = stmt.where(Trade.executed_at <= end_time)

        stmt = stmt.order_by(Trade.executed_at)
        return list(session.scalars(stmt).all())


class PositionRepository(BaseRepository[Position]):
    """Repository for position operations."""

    model = Position

    def get_position(self, session: Session, symbol: str) -> Position | None:
        """Get position for a symbol.

        Args:
            session: Database session.
            symbol: Stock symbol.

        Returns:
            Position or None.
        """
        return session.get(Position, symbol.upper())

    def get_all_positions(self, session: Session) -> list[Position]:
        """Get all current positions.

        Args:
            session: Database session.

        Returns:
            List of positions.
        """
        stmt = select(Position).where(Position.quantity != 0)
        return list(session.scalars(stmt).all())

    def update_position(
        self,
        session: Session,
        symbol: str,
        quantity: Decimal,
        avg_entry_price: Decimal,
        cost_basis: Decimal,
    ) -> Position:
        """Update or create a position.

        Args:
            session: Database session.
            symbol: Stock symbol.
            quantity: New quantity.
            avg_entry_price: Average entry price.
            cost_basis: Total cost basis.

        Returns:
            Updated or created position.
        """
        position = self.get_position(session, symbol)
        if position:
            position.quantity = quantity
            position.avg_entry_price = avg_entry_price
            position.cost_basis = cost_basis
            session.flush()
            return position
        else:
            return self.create(
                session,
                symbol=symbol.upper(),
                quantity=quantity,
                avg_entry_price=avg_entry_price,
                cost_basis=cost_basis,
            )


class SignalRepository(BaseRepository[Signal]):
    """Repository for signal operations."""

    model = Signal

    def get_signals(
        self,
        session: Session,
        symbol: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model_source: str | None = None,
        limit: int = 100,
    ) -> list[Signal]:
        """Get signals with optional filters.

        Args:
            session: Database session.
            symbol: Optional symbol filter.
            start_time: Start of time range.
            end_time: End of time range.
            model_source: Optional model source filter.
            limit: Maximum signals to return.

        Returns:
            List of signals.
        """
        stmt = select(Signal)

        if symbol:
            stmt = stmt.where(Signal.symbol == symbol.upper())
        if start_time:
            stmt = stmt.where(Signal.timestamp >= start_time)
        if end_time:
            stmt = stmt.where(Signal.timestamp <= end_time)
        if model_source:
            stmt = stmt.where(Signal.model_source == model_source)

        stmt = stmt.order_by(desc(Signal.timestamp)).limit(limit)
        return list(session.scalars(stmt).all())

    def get_latest_signal(
        self,
        session: Session,
        symbol: str,
        model_source: str | None = None,
    ) -> Signal | None:
        """Get the most recent signal for a symbol.

        Args:
            session: Database session.
            symbol: Stock symbol.
            model_source: Optional model source filter.

        Returns:
            Most recent signal or None.
        """
        stmt = (
            select(Signal)
            .where(Signal.symbol == symbol.upper())
            .order_by(desc(Signal.timestamp))
            .limit(1)
        )

        if model_source:
            stmt = stmt.where(Signal.model_source == model_source)

        return session.scalars(stmt).first()


class DailyPerformanceRepository(BaseRepository[DailyPerformance]):
    """Repository for daily performance operations."""

    model = DailyPerformance

    def get_performance_range(
        self,
        session: Session,
        start_date: datetime,
        end_date: datetime,
    ) -> list[DailyPerformance]:
        """Get daily performance for a date range.

        Args:
            session: Database session.
            start_date: Start date.
            end_date: End date.

        Returns:
            List of daily performance records.
        """
        stmt = (
            select(DailyPerformance)
            .where(
                and_(
                    DailyPerformance.date >= start_date.date(),
                    DailyPerformance.date <= end_date.date(),
                )
            )
            .order_by(DailyPerformance.date)
        )
        return list(session.scalars(stmt).all())


class AlertRepository(BaseRepository[Alert]):
    """Repository for alert operations."""

    model = Alert

    def get_unacknowledged(self, session: Session, limit: int = 50) -> list[Alert]:
        """Get unacknowledged alerts.

        Args:
            session: Database session.
            limit: Maximum alerts to return.

        Returns:
            List of unacknowledged alerts.
        """
        stmt = (
            select(Alert)
            .where(Alert.acknowledged == False)
            .order_by(desc(Alert.timestamp))
            .limit(limit)
        )
        return list(session.scalars(stmt).all())

    def acknowledge(
        self,
        session: Session,
        alert_id: str,
        acknowledged_by: str,
    ) -> Alert | None:
        """Acknowledge an alert.

        Args:
            session: Database session.
            alert_id: Alert ID.
            acknowledged_by: User who acknowledged.

        Returns:
            Updated alert or None.
        """
        alert = session.get(Alert, alert_id)
        if alert:
            alert.acknowledged = True
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            session.flush()
        return alert


# Repository instances (singletons)
_ohlcv_repo: OHLCVRepository | None = None
_feature_repo: FeatureRepository | None = None
_order_repo: OrderRepository | None = None
_trade_repo: TradeRepository | None = None
_position_repo: PositionRepository | None = None
_signal_repo: SignalRepository | None = None


def get_ohlcv_repository() -> OHLCVRepository:
    """Get OHLCV repository instance."""
    global _ohlcv_repo
    if _ohlcv_repo is None:
        _ohlcv_repo = OHLCVRepository()
    return _ohlcv_repo


def get_feature_repository() -> FeatureRepository:
    """Get feature repository instance."""
    global _feature_repo
    if _feature_repo is None:
        _feature_repo = FeatureRepository()
    return _feature_repo


def get_order_repository() -> OrderRepository:
    """Get order repository instance."""
    global _order_repo
    if _order_repo is None:
        _order_repo = OrderRepository()
    return _order_repo


def get_trade_repository() -> TradeRepository:
    """Get trade repository instance."""
    global _trade_repo
    if _trade_repo is None:
        _trade_repo = TradeRepository()
    return _trade_repo


def get_position_repository() -> PositionRepository:
    """Get position repository instance."""
    global _position_repo
    if _position_repo is None:
        _position_repo = PositionRepository()
    return _position_repo


def get_signal_repository() -> SignalRepository:
    """Get signal repository instance."""
    global _signal_repo
    if _signal_repo is None:
        _signal_repo = SignalRepository()
    return _signal_repo
