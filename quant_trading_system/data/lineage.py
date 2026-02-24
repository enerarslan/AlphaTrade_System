"""
JPMORGAN FIX: Data Lineage Tracking Module.

Implements institutional-grade data lineage tracking for:
1. Data source provenance
2. Transformation tracking
3. Feature dependency graphs
4. Model training data snapshots
5. Regulatory audit trails

This is critical for:
- Model debugging and reproducibility
- Regulatory compliance (MiFID II, Dodd-Frank)
- Data quality root cause analysis
- Feature impact analysis

Reference:
    Financial Industry Data Governance Standards
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class LineageEventType(str, Enum):
    """Types of lineage events."""

    DATA_INGESTION = "data_ingestion"
    DATA_TRANSFORMATION = "data_transformation"
    FEATURE_COMPUTATION = "feature_computation"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    DATA_VALIDATION = "data_validation"
    DATA_EXPORT = "data_export"
    MANUAL_CORRECTION = "manual_correction"


class DataQualityFlag(str, Enum):
    """Data quality flags for lineage tracking."""

    CLEAN = "clean"
    MISSING_VALUES = "missing_values"
    OUTLIERS_DETECTED = "outliers_detected"
    STALE_DATA = "stale_data"
    INTERPOLATED = "interpolated"
    MANUALLY_CORRECTED = "manually_corrected"
    VALIDATED = "validated"
    SUSPICIOUS = "suspicious"


@dataclass
class DataSource:
    """Represents a data source in the lineage graph."""

    source_id: str
    source_type: str  # e.g., "alpaca_api", "csv_file", "database"
    source_name: str  # e.g., "AAPL_15min_2024-08-01.csv"
    source_uri: str  # Full path or URI
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["created_at"] = self.created_at.isoformat()
        return result


@dataclass
class LineageNode:
    """
    A node in the data lineage graph.

    Represents a specific version of data at a point in the pipeline.
    """

    node_id: str
    data_type: str  # e.g., "raw_ohlcv", "features", "predictions"
    symbol: str | None = None
    timestamp_range: tuple[datetime, datetime] | None = None
    row_count: int = 0
    column_count: int = 0
    data_hash: str | None = None
    quality_flags: list[DataQualityFlag] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "node_id": self.node_id,
            "data_type": self.data_type,
            "symbol": self.symbol,
            "timestamp_range": None,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "data_hash": self.data_hash,
            "quality_flags": [f.value for f in self.quality_flags],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
        if self.timestamp_range:
            result["timestamp_range"] = [
                self.timestamp_range[0].isoformat(),
                self.timestamp_range[1].isoformat(),
            ]
        return result


@dataclass
class LineageEdge:
    """
    An edge in the data lineage graph.

    Represents a transformation or operation between nodes.
    """

    edge_id: str
    source_node_id: str
    target_node_id: str
    event_type: LineageEventType
    operation: str  # e.g., "feature_engineering", "normalization"
    parameters: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "event_type": self.event_type.value,
            "operation": self.operation,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
        }


@dataclass
class LineageEvent:
    """
    A lineage event for audit logging.

    Captures all details of a data operation for regulatory compliance.
    """

    event_id: str
    event_type: LineageEventType
    timestamp: datetime
    user_id: str
    source_nodes: list[str]
    target_nodes: list[str]
    operation: str
    parameters: dict[str, Any]
    status: str  # "success", "failure", "partial"
    error_message: str | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "source_nodes": self.source_nodes,
            "target_nodes": self.target_nodes,
            "operation": self.operation,
            "parameters": self.parameters,
            "status": self.status,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class DataLineageTracker:
    """
    JPMORGAN FIX: Central data lineage tracking system.

    Provides:
    1. Full data provenance tracking
    2. Transformation audit trail
    3. Feature dependency analysis
    4. Model training data snapshots
    5. Regulatory compliance logging

    Usage:
        tracker = DataLineageTracker()

        # Register data source
        source = tracker.register_source(
            source_type="csv_file",
            source_name="AAPL_15min.csv",
            source_uri="/data/raw/AAPL_15min.csv"
        )

        # Create data node
        raw_node = tracker.create_node(
            data_type="raw_ohlcv",
            symbol="AAPL",
            row_count=1000,
            data_hash=compute_hash(df)
        )

        # Record transformation
        feature_node = tracker.create_node(...)
        tracker.record_transformation(
            source_node=raw_node,
            target_node=feature_node,
            operation="feature_engineering",
            parameters={"indicators": ["RSI", "MACD"]}
        )

        # Query lineage
        upstream = tracker.get_upstream_lineage(feature_node.node_id)
    """

    def __init__(self, storage_path: str | Path | None = None):
        """
        Initialize lineage tracker.

        Args:
            storage_path: Path to persist lineage data (None for in-memory only)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self._sources: dict[str, DataSource] = {}
        self._nodes: dict[str, LineageNode] = {}
        self._edges: dict[str, LineageEdge] = {}
        self._events: list[LineageEvent] = []

        # Indexes for fast lookups
        self._node_to_sources: dict[str, set[str]] = {}  # node_id -> source_ids
        self._node_upstream: dict[str, set[str]] = {}  # node_id -> upstream node_ids
        self._node_downstream: dict[str, set[str]] = {}  # node_id -> downstream node_ids

        # Load existing data if storage path provided
        if self.storage_path:
            self._load_from_storage()

    def register_source(
        self,
        source_type: str,
        source_name: str,
        source_uri: str,
        metadata: dict[str, Any] | None = None,
    ) -> DataSource:
        """
        Register a new data source.

        Args:
            source_type: Type of source (e.g., "alpaca_api", "csv_file")
            source_name: Human-readable name
            source_uri: Full path or URI
            metadata: Additional metadata

        Returns:
            Created DataSource object
        """
        source_id = f"src_{uuid4().hex[:12]}"
        source = DataSource(
            source_id=source_id,
            source_type=source_type,
            source_name=source_name,
            source_uri=source_uri,
            metadata=metadata or {},
        )
        self._sources[source_id] = source

        logger.info(f"Registered data source: {source_name} ({source_type})")

        if self.storage_path:
            self._save_to_storage()

        return source

    def create_node(
        self,
        data_type: str,
        symbol: str | None = None,
        timestamp_range: tuple[datetime, datetime] | None = None,
        row_count: int = 0,
        column_count: int = 0,
        data_hash: str | None = None,
        quality_flags: list[DataQualityFlag] | None = None,
        source_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LineageNode:
        """
        Create a new lineage node.

        Args:
            data_type: Type of data (e.g., "raw_ohlcv", "features")
            symbol: Trading symbol if applicable
            timestamp_range: Time range of data
            row_count: Number of rows
            column_count: Number of columns
            data_hash: Hash of data content for integrity
            quality_flags: Data quality flags
            source_ids: Associated data source IDs
            metadata: Additional metadata

        Returns:
            Created LineageNode object
        """
        node_id = f"node_{uuid4().hex[:12]}"
        node = LineageNode(
            node_id=node_id,
            data_type=data_type,
            symbol=symbol,
            timestamp_range=timestamp_range,
            row_count=row_count,
            column_count=column_count,
            data_hash=data_hash,
            quality_flags=quality_flags or [],
            metadata=metadata or {},
        )
        self._nodes[node_id] = node

        # Link to sources
        if source_ids:
            self._node_to_sources[node_id] = set(source_ids)

        # Initialize relationship indexes
        if node_id not in self._node_upstream:
            self._node_upstream[node_id] = set()
        if node_id not in self._node_downstream:
            self._node_downstream[node_id] = set()

        logger.debug(f"Created lineage node: {node_id} ({data_type})")

        if self.storage_path:
            self._save_to_storage()

        return node

    def record_transformation(
        self,
        source_node: LineageNode | str,
        target_node: LineageNode | str,
        operation: str,
        event_type: LineageEventType = LineageEventType.DATA_TRANSFORMATION,
        parameters: dict[str, Any] | None = None,
        created_by: str = "system",
    ) -> LineageEdge:
        """
        Record a transformation between nodes.

        Args:
            source_node: Source node or node_id
            target_node: Target node or node_id
            operation: Name of operation
            event_type: Type of lineage event
            parameters: Operation parameters
            created_by: User or system that created this

        Returns:
            Created LineageEdge object
        """
        source_id = source_node.node_id if isinstance(source_node, LineageNode) else source_node
        target_id = target_node.node_id if isinstance(target_node, LineageNode) else target_node

        edge_id = f"edge_{uuid4().hex[:12]}"
        edge = LineageEdge(
            edge_id=edge_id,
            source_node_id=source_id,
            target_node_id=target_id,
            event_type=event_type,
            operation=operation,
            parameters=parameters or {},
            created_by=created_by,
        )
        self._edges[edge_id] = edge

        # Update relationship indexes
        if target_id not in self._node_upstream:
            self._node_upstream[target_id] = set()
        self._node_upstream[target_id].add(source_id)

        if source_id not in self._node_downstream:
            self._node_downstream[source_id] = set()
        self._node_downstream[source_id].add(target_id)

        logger.debug(f"Recorded transformation: {source_id} -> {target_id} ({operation})")

        if self.storage_path:
            self._save_to_storage()

        return edge

    def record_event(
        self,
        event_type: LineageEventType,
        operation: str,
        source_nodes: list[str],
        target_nodes: list[str],
        user_id: str = "system",
        parameters: dict[str, Any] | None = None,
        status: str = "success",
        error_message: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LineageEvent:
        """
        Record an audit event.

        Args:
            event_type: Type of event
            operation: Operation name
            source_nodes: Source node IDs
            target_nodes: Target node IDs
            user_id: User or system ID
            parameters: Operation parameters
            status: Event status
            error_message: Error message if failed
            duration_ms: Operation duration in milliseconds
            metadata: Additional metadata

        Returns:
            Created LineageEvent object
        """
        event = LineageEvent(
            event_id=f"evt_{uuid4().hex[:12]}",
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            operation=operation,
            parameters=parameters or {},
            status=status,
            error_message=error_message,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self._events.append(event)

        if status == "failure":
            logger.warning(f"Lineage event failed: {operation} - {error_message}")
        else:
            logger.debug(f"Recorded lineage event: {operation} ({status})")

        if self.storage_path:
            self._save_to_storage()

        return event

    def get_upstream_lineage(
        self,
        node_id: str,
        max_depth: int = 10,
    ) -> list[LineageNode]:
        """
        Get all upstream nodes (data sources) for a node.

        Args:
            node_id: Target node ID
            max_depth: Maximum traversal depth

        Returns:
            List of upstream nodes in order from closest to furthest
        """
        visited = set()
        result = []
        queue = [(node_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)

            for upstream_id in self._node_upstream.get(current_id, []):
                if upstream_id in self._nodes:
                    result.append(self._nodes[upstream_id])
                    queue.append((upstream_id, depth + 1))

        return result

    def get_downstream_lineage(
        self,
        node_id: str,
        max_depth: int = 10,
    ) -> list[LineageNode]:
        """
        Get all downstream nodes (derived data) for a node.

        Args:
            node_id: Source node ID
            max_depth: Maximum traversal depth

        Returns:
            List of downstream nodes
        """
        visited = set()
        result = []
        queue = [(node_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)

            for downstream_id in self._node_downstream.get(current_id, []):
                if downstream_id in self._nodes:
                    result.append(self._nodes[downstream_id])
                    queue.append((downstream_id, depth + 1))

        return result

    def get_data_sources(self, node_id: str) -> list[DataSource]:
        """Get all data sources for a node."""
        # Direct sources
        source_ids = self._node_to_sources.get(node_id, set())

        # Inherited sources from upstream
        for upstream_node in self.get_upstream_lineage(node_id):
            source_ids.update(self._node_to_sources.get(upstream_node.node_id, set()))

        return [self._sources[sid] for sid in source_ids if sid in self._sources]

    def get_events_for_node(
        self,
        node_id: str,
        event_types: list[LineageEventType] | None = None,
    ) -> list[LineageEvent]:
        """Get all events related to a node."""
        events = []
        for event in self._events:
            if node_id in event.source_nodes or node_id in event.target_nodes:
                if event_types is None or event.event_type in event_types:
                    events.append(event)
        return sorted(events, key=lambda e: e.timestamp)

    def export_lineage_graph(self, format: str = "dict") -> dict[str, Any]:
        """
        Export the entire lineage graph.

        Args:
            format: Export format ("dict" or "json")

        Returns:
            Dictionary representation of the lineage graph
        """
        graph = {
            "sources": {sid: s.to_dict() for sid, s in self._sources.items()},
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "edges": {eid: e.to_dict() for eid, e in self._edges.items()},
            "events": [e.to_dict() for e in self._events[-1000:]],  # Last 1000 events
            "metadata": {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "event_count": len(self._events),
            },
        }
        return graph

    def _save_to_storage(self) -> None:
        """Save lineage data to storage."""
        if not self.storage_path:
            return

        try:
            graph = self.export_lineage_graph()
            lineage_file = self.storage_path / "lineage.json"
            with open(lineage_file, "w") as f:
                json.dump(graph, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save lineage data: {e}")

    def _load_from_storage(self) -> None:
        """Load lineage data from storage."""
        if not self.storage_path:
            return

        lineage_file = self.storage_path / "lineage.json"
        if not lineage_file.exists():
            return

        try:
            with open(lineage_file, "r") as f:
                graph = json.load(f)

            # Restore sources
            for sid, sdata in graph.get("sources", {}).items():
                self._sources[sid] = DataSource(
                    source_id=sdata["source_id"],
                    source_type=sdata["source_type"],
                    source_name=sdata["source_name"],
                    source_uri=sdata["source_uri"],
                    created_at=datetime.fromisoformat(sdata["created_at"]),
                    metadata=sdata.get("metadata", {}),
                )

            # Restore nodes
            for nid, ndata in graph.get("nodes", {}).items():
                ts_range = None
                if ndata.get("timestamp_range"):
                    ts_range = (
                        datetime.fromisoformat(ndata["timestamp_range"][0]),
                        datetime.fromisoformat(ndata["timestamp_range"][1]),
                    )
                self._nodes[nid] = LineageNode(
                    node_id=ndata["node_id"],
                    data_type=ndata["data_type"],
                    symbol=ndata.get("symbol"),
                    timestamp_range=ts_range,
                    row_count=ndata.get("row_count", 0),
                    column_count=ndata.get("column_count", 0),
                    data_hash=ndata.get("data_hash"),
                    quality_flags=[DataQualityFlag(f) for f in ndata.get("quality_flags", [])],
                    created_at=datetime.fromisoformat(ndata["created_at"]),
                    metadata=ndata.get("metadata", {}),
                )

            # Restore edges and rebuild indexes
            for eid, edata in graph.get("edges", {}).items():
                edge = LineageEdge(
                    edge_id=edata["edge_id"],
                    source_node_id=edata["source_node_id"],
                    target_node_id=edata["target_node_id"],
                    event_type=LineageEventType(edata["event_type"]),
                    operation=edata["operation"],
                    parameters=edata.get("parameters", {}),
                    created_at=datetime.fromisoformat(edata["created_at"]),
                    created_by=edata.get("created_by", "system"),
                )
                self._edges[eid] = edge

                # Rebuild indexes
                source_id = edge.source_node_id
                target_id = edge.target_node_id

                if target_id not in self._node_upstream:
                    self._node_upstream[target_id] = set()
                self._node_upstream[target_id].add(source_id)

                if source_id not in self._node_downstream:
                    self._node_downstream[source_id] = set()
                self._node_downstream[source_id].add(target_id)

            logger.info(
                f"Loaded lineage data: {len(self._nodes)} nodes, "
                f"{len(self._edges)} edges, {len(self._sources)} sources"
            )

        except Exception as e:
            logger.error(f"Failed to load lineage data: {e}")


def compute_data_hash(data: Any, algorithm: str = "sha256") -> str:
    """
    Compute a hash of data for integrity tracking.

    Args:
        data: Data to hash (DataFrame, ndarray, or bytes)
        algorithm: Hash algorithm to use

    Returns:
        Hex digest of the hash
    """
    import numpy as np
    import pandas as pd

    hasher = hashlib.new(algorithm)

    if isinstance(data, pd.DataFrame):
        # Hash column names and dtypes
        hasher.update(str(list(data.columns)).encode())
        hasher.update(str(list(data.dtypes)).encode())
        # Hash values
        hasher.update(data.values.tobytes())
    elif isinstance(data, np.ndarray):
        hasher.update(data.tobytes())
    elif isinstance(data, bytes):
        hasher.update(data)
    else:
        hasher.update(str(data).encode())

    return hasher.hexdigest()


# =============================================================================
# JPMorgan-level Enhancement: Persistent Storage Options
# =============================================================================


class LineageStorageBackend:
    """Abstract base for lineage storage backends."""

    def save_source(self, source: DataSource) -> None:
        """Save a data source."""
        raise NotImplementedError

    def save_node(self, node: LineageNode) -> None:
        """Save a lineage node."""
        raise NotImplementedError

    def save_edge(self, edge: LineageEdge) -> None:
        """Save a lineage edge."""
        raise NotImplementedError

    def save_event(self, event: LineageEvent) -> None:
        """Save a lineage event."""
        raise NotImplementedError

    def load_all(self) -> dict[str, Any]:
        """Load all lineage data."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the storage backend."""
        pass


class SQLiteLineageStorage(LineageStorageBackend):
    """SQLite-based persistent storage for lineage data.

    JPMorgan-level enhancement: Provides ACID-compliant persistent
    storage for data lineage with proper concurrent access support.
    """

    def __init__(self, db_path: str | Path):
        """Initialize SQLite storage.

        Args:
            db_path: Path to the SQLite database file.
        """
        import sqlite3

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = __import__("threading").RLock()

        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        with self._lock:
            cursor = self._conn.cursor()

            # Sources table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_sources (
                    source_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            # Nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lineage_nodes (
                    node_id TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    symbol TEXT,
                    timestamp_start TEXT,
                    timestamp_end TEXT,
                    row_count INTEGER DEFAULT 0,
                    column_count INTEGER DEFAULT 0,
                    data_hash TEXT,
                    quality_flags TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            # Edges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lineage_edges (
                    edge_id TEXT PRIMARY KEY,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    parameters TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT DEFAULT 'system',
                    FOREIGN KEY (source_node_id) REFERENCES lineage_nodes(node_id),
                    FOREIGN KEY (target_node_id) REFERENCES lineage_nodes(node_id)
                )
            """)

            # Events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lineage_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    source_nodes TEXT,
                    target_nodes TEXT,
                    operation TEXT NOT NULL,
                    parameters TEXT,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    duration_ms REAL,
                    metadata TEXT
                )
            """)

            # Node-source associations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS node_sources (
                    node_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    PRIMARY KEY (node_id, source_id),
                    FOREIGN KEY (node_id) REFERENCES lineage_nodes(node_id),
                    FOREIGN KEY (source_id) REFERENCES data_sources(source_id)
                )
            """)

            # Indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_edges_source
                ON lineage_edges(source_node_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_edges_target
                ON lineage_edges(target_node_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON lineage_events(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_nodes_symbol
                ON lineage_nodes(symbol)
            """)

            self._conn.commit()

    def save_source(self, source: DataSource) -> None:
        """Save a data source to the database."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO data_sources
                (source_id, source_type, source_name, source_uri, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                source.source_id,
                source.source_type,
                source.source_name,
                source.source_uri,
                source.created_at.isoformat(),
                json.dumps(source.metadata),
            ))
            self._conn.commit()

    def save_node(self, node: LineageNode, source_ids: list[str] | None = None) -> None:
        """Save a lineage node to the database."""
        with self._lock:
            cursor = self._conn.cursor()

            ts_start = None
            ts_end = None
            if node.timestamp_range:
                ts_start = node.timestamp_range[0].isoformat()
                ts_end = node.timestamp_range[1].isoformat()

            cursor.execute("""
                INSERT OR REPLACE INTO lineage_nodes
                (node_id, data_type, symbol, timestamp_start, timestamp_end,
                 row_count, column_count, data_hash, quality_flags, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node.node_id,
                node.data_type,
                node.symbol,
                ts_start,
                ts_end,
                node.row_count,
                node.column_count,
                node.data_hash,
                json.dumps([f.value for f in node.quality_flags]),
                node.created_at.isoformat(),
                json.dumps(node.metadata),
            ))

            # Save node-source associations
            if source_ids:
                for source_id in source_ids:
                    cursor.execute("""
                        INSERT OR IGNORE INTO node_sources (node_id, source_id)
                        VALUES (?, ?)
                    """, (node.node_id, source_id))

            self._conn.commit()

    def save_edge(self, edge: LineageEdge) -> None:
        """Save a lineage edge to the database."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO lineage_edges
                (edge_id, source_node_id, target_node_id, event_type,
                 operation, parameters, created_at, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                edge.edge_id,
                edge.source_node_id,
                edge.target_node_id,
                edge.event_type.value,
                edge.operation,
                json.dumps(edge.parameters),
                edge.created_at.isoformat(),
                edge.created_by,
            ))
            self._conn.commit()

    def save_event(self, event: LineageEvent) -> None:
        """Save a lineage event to the database."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT INTO lineage_events
                (event_id, event_type, timestamp, user_id, source_nodes,
                 target_nodes, operation, parameters, status, error_message,
                 duration_ms, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                event.user_id,
                json.dumps(event.source_nodes),
                json.dumps(event.target_nodes),
                event.operation,
                json.dumps(event.parameters),
                event.status,
                event.error_message,
                event.duration_ms,
                json.dumps(event.metadata),
            ))
            self._conn.commit()

    def load_all(self) -> dict[str, Any]:
        """Load all lineage data from the database."""
        with self._lock:
            cursor = self._conn.cursor()

            # Load sources
            sources = {}
            cursor.execute("SELECT * FROM data_sources")
            for row in cursor.fetchall():
                sources[row["source_id"]] = DataSource(
                    source_id=row["source_id"],
                    source_type=row["source_type"],
                    source_name=row["source_name"],
                    source_uri=row["source_uri"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )

            # Load nodes
            nodes = {}
            cursor.execute("SELECT * FROM lineage_nodes")
            for row in cursor.fetchall():
                ts_range = None
                if row["timestamp_start"] and row["timestamp_end"]:
                    ts_range = (
                        datetime.fromisoformat(row["timestamp_start"]),
                        datetime.fromisoformat(row["timestamp_end"]),
                    )
                nodes[row["node_id"]] = LineageNode(
                    node_id=row["node_id"],
                    data_type=row["data_type"],
                    symbol=row["symbol"],
                    timestamp_range=ts_range,
                    row_count=row["row_count"] or 0,
                    column_count=row["column_count"] or 0,
                    data_hash=row["data_hash"],
                    quality_flags=[
                        DataQualityFlag(f)
                        for f in json.loads(row["quality_flags"] or "[]")
                    ],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )

            # Load node-source associations
            node_to_sources = {}
            cursor.execute("SELECT * FROM node_sources")
            for row in cursor.fetchall():
                if row["node_id"] not in node_to_sources:
                    node_to_sources[row["node_id"]] = set()
                node_to_sources[row["node_id"]].add(row["source_id"])

            # Load edges
            edges = {}
            node_upstream = {}
            node_downstream = {}
            cursor.execute("SELECT * FROM lineage_edges")
            for row in cursor.fetchall():
                edge = LineageEdge(
                    edge_id=row["edge_id"],
                    source_node_id=row["source_node_id"],
                    target_node_id=row["target_node_id"],
                    event_type=LineageEventType(row["event_type"]),
                    operation=row["operation"],
                    parameters=json.loads(row["parameters"]) if row["parameters"] else {},
                    created_at=datetime.fromisoformat(row["created_at"]),
                    created_by=row["created_by"] or "system",
                )
                edges[row["edge_id"]] = edge

                # Build indexes
                source_id = edge.source_node_id
                target_id = edge.target_node_id

                if target_id not in node_upstream:
                    node_upstream[target_id] = set()
                node_upstream[target_id].add(source_id)

                if source_id not in node_downstream:
                    node_downstream[source_id] = set()
                node_downstream[source_id].add(target_id)

            # Load events (last 1000)
            events = []
            cursor.execute("""
                SELECT * FROM lineage_events
                ORDER BY timestamp DESC LIMIT 1000
            """)
            for row in cursor.fetchall():
                events.append(LineageEvent(
                    event_id=row["event_id"],
                    event_type=LineageEventType(row["event_type"]),
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    user_id=row["user_id"],
                    source_nodes=json.loads(row["source_nodes"]) if row["source_nodes"] else [],
                    target_nodes=json.loads(row["target_nodes"]) if row["target_nodes"] else [],
                    operation=row["operation"],
                    parameters=json.loads(row["parameters"]) if row["parameters"] else {},
                    status=row["status"],
                    error_message=row["error_message"],
                    duration_ms=row["duration_ms"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                ))
            events.reverse()  # Oldest first

            return {
                "sources": sources,
                "nodes": nodes,
                "edges": edges,
                "events": events,
                "node_to_sources": node_to_sources,
                "node_upstream": node_upstream,
                "node_downstream": node_downstream,
            }

    def query_events(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        event_types: list[LineageEventType] | None = None,
        limit: int = 1000,
    ) -> list[LineageEvent]:
        """Query events with filtering.

        Args:
            since: Only return events after this time.
            until: Only return events before this time.
            event_types: Only return these event types.
            limit: Maximum number of events to return.

        Returns:
            List of matching LineageEvent records.
        """
        with self._lock:
            cursor = self._conn.cursor()

            query = "SELECT * FROM lineage_events WHERE 1=1"
            params = []

            if since:
                query += " AND timestamp >= ?"
                params.append(since.isoformat())
            if until:
                query += " AND timestamp <= ?"
                params.append(until.isoformat())
            if event_types:
                placeholders = ",".join("?" * len(event_types))
                query += f" AND event_type IN ({placeholders})"
                params.extend(et.value for et in event_types)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            events = []
            for row in cursor.fetchall():
                events.append(LineageEvent(
                    event_id=row["event_id"],
                    event_type=LineageEventType(row["event_type"]),
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    user_id=row["user_id"],
                    source_nodes=json.loads(row["source_nodes"]) if row["source_nodes"] else [],
                    target_nodes=json.loads(row["target_nodes"]) if row["target_nodes"] else [],
                    operation=row["operation"],
                    parameters=json.loads(row["parameters"]) if row["parameters"] else {},
                    status=row["status"],
                    error_message=row["error_message"],
                    duration_ms=row["duration_ms"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                ))

            events.reverse()
            return events

    def get_statistics(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics.
        """
        with self._lock:
            cursor = self._conn.cursor()

            stats = {}

            cursor.execute("SELECT COUNT(*) as count FROM data_sources")
            stats["source_count"] = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM lineage_nodes")
            stats["node_count"] = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM lineage_edges")
            stats["edge_count"] = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM lineage_events")
            stats["event_count"] = cursor.fetchone()["count"]

            cursor.execute("""
                SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest
                FROM lineage_events
            """)
            row = cursor.fetchone()
            if row["oldest"]:
                stats["oldest_event"] = row["oldest"]
                stats["newest_event"] = row["newest"]

            stats["db_size_bytes"] = self.db_path.stat().st_size

            return stats

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()


class PersistentDataLineageTracker(DataLineageTracker):
    """Enhanced DataLineageTracker with robust persistent storage.

    JPMorgan-level enhancement: Extends the base tracker with
    SQLite-backed persistent storage for institutional use.
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        use_sqlite: bool = True,
    ):
        """Initialize with persistent storage.

        Args:
            storage_path: Path to storage (directory for JSON, file for SQLite).
            use_sqlite: Use SQLite backend (default) vs JSON file.
        """
        self.use_sqlite = use_sqlite
        self._sqlite_backend: SQLiteLineageStorage | None = None

        if storage_path and use_sqlite:
            db_path = Path(storage_path)
            if db_path.is_dir():
                db_path = db_path / "lineage.db"
            self._sqlite_backend = SQLiteLineageStorage(db_path)

        # Initialize parent (will handle JSON storage if use_sqlite=False)
        super().__init__(storage_path if not use_sqlite else None)

        # Load from SQLite if available
        if self._sqlite_backend:
            self._load_from_sqlite()

    def _load_from_sqlite(self) -> None:
        """Load lineage data from SQLite storage."""
        if not self._sqlite_backend:
            return

        try:
            data = self._sqlite_backend.load_all()

            self._sources = data["sources"]
            self._nodes = data["nodes"]
            self._edges = data["edges"]
            self._events = data["events"]
            self._node_to_sources = data["node_to_sources"]
            self._node_upstream = data["node_upstream"]
            self._node_downstream = data["node_downstream"]

            logger.info(
                f"Loaded lineage data from SQLite: {len(self._nodes)} nodes, "
                f"{len(self._edges)} edges, {len(self._sources)} sources"
            )

        except Exception as e:
            logger.error(f"Failed to load lineage data from SQLite: {e}")

    def register_source(
        self,
        source_type: str,
        source_name: str,
        source_uri: str,
        metadata: dict[str, Any] | None = None,
    ) -> DataSource:
        """Register a data source with persistent storage."""
        source = super().register_source(source_type, source_name, source_uri, metadata)

        if self._sqlite_backend:
            self._sqlite_backend.save_source(source)

        return source

    def create_node(
        self,
        data_type: str,
        symbol: str | None = None,
        timestamp_range: tuple[datetime, datetime] | None = None,
        row_count: int = 0,
        column_count: int = 0,
        data_hash: str | None = None,
        quality_flags: list[DataQualityFlag] | None = None,
        source_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LineageNode:
        """Create a lineage node with persistent storage."""
        node = super().create_node(
            data_type, symbol, timestamp_range, row_count, column_count,
            data_hash, quality_flags, source_ids, metadata
        )

        if self._sqlite_backend:
            self._sqlite_backend.save_node(node, source_ids)

        return node

    def record_transformation(
        self,
        source_node: LineageNode | str,
        target_node: LineageNode | str,
        operation: str,
        event_type: LineageEventType = LineageEventType.DATA_TRANSFORMATION,
        parameters: dict[str, Any] | None = None,
        created_by: str = "system",
    ) -> LineageEdge:
        """Record a transformation with persistent storage."""
        edge = super().record_transformation(
            source_node, target_node, operation, event_type, parameters, created_by
        )

        if self._sqlite_backend:
            self._sqlite_backend.save_edge(edge)

        return edge

    def record_event(
        self,
        event_type: LineageEventType,
        operation: str,
        source_nodes: list[str],
        target_nodes: list[str],
        user_id: str = "system",
        parameters: dict[str, Any] | None = None,
        status: str = "success",
        error_message: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LineageEvent:
        """Record an event with persistent storage."""
        event = super().record_event(
            event_type, operation, source_nodes, target_nodes, user_id,
            parameters, status, error_message, duration_ms, metadata
        )

        if self._sqlite_backend:
            self._sqlite_backend.save_event(event)

        return event

    def query_events(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        event_types: list[LineageEventType] | None = None,
        limit: int = 1000,
    ) -> list[LineageEvent]:
        """Query events with filtering (optimized for SQLite).

        Args:
            since: Only return events after this time.
            until: Only return events before this time.
            event_types: Only return these event types.
            limit: Maximum number of events to return.

        Returns:
            List of matching LineageEvent records.
        """
        if self._sqlite_backend:
            return self._sqlite_backend.query_events(since, until, event_types, limit)

        # Fall back to in-memory filtering
        events = self._events.copy()
        if since:
            events = [e for e in events if e.timestamp >= since]
        if until:
            events = [e for e in events if e.timestamp <= until]
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        return events[-limit:]

    def get_storage_statistics(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics.
        """
        if self._sqlite_backend:
            return self._sqlite_backend.get_statistics()

        return {
            "source_count": len(self._sources),
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "event_count": len(self._events),
            "storage_type": "memory",
        }

    def close(self) -> None:
        """Close the storage backend."""
        if self._sqlite_backend:
            self._sqlite_backend.close()


# Singleton instance for global access
_global_tracker: DataLineageTracker | None = None


def get_lineage_tracker(
    storage_path: str | Path | None = None,
    use_sqlite: bool = True,
) -> DataLineageTracker:
    """
    Get the global lineage tracker instance.

    JPMorgan-level enhancement: Now supports SQLite-backed persistent storage.

    Args:
        storage_path: Path to persist lineage data.
        use_sqlite: Use SQLite backend (True) or JSON file (False).

    Returns:
        DataLineageTracker instance with persistent storage.
    """
    global _global_tracker
    if _global_tracker is None:
        if storage_path and use_sqlite:
            _global_tracker = PersistentDataLineageTracker(storage_path, use_sqlite=True)
        else:
            _global_tracker = DataLineageTracker(storage_path)
    return _global_tracker
