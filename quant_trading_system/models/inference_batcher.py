"""
GPU Inference Batching for High-Throughput Model Predictions.

Implements dynamic batching to maximize GPU utilization during inference.
Multiple prediction requests are collected and batched together for
more efficient GPU processing.

Features:
- Automatic request batching with configurable timeouts
- Priority queue for latency-sensitive predictions
- Async/await compatible for non-blocking inference
- Memory-efficient batch size management
- Multi-model support with model-specific batch sizes

Usage:
    from quant_trading_system.models.inference_batcher import (
        InferenceBatcher, get_inference_batcher
    )

    # Get singleton batcher
    batcher = get_inference_batcher()

    # Register a model
    batcher.register_model("lstm_v1", model, batch_size=64)

    # Submit prediction request (will be batched)
    result = await batcher.predict("lstm_v1", features)

    # Or use synchronous API
    result = batcher.predict_sync("lstm_v1", features)
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar
from uuid import UUID, uuid4

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BatchPriority(Enum):
    """Priority levels for inference requests."""

    HIGH = 0      # Real-time trading signals
    NORMAL = 1    # Standard predictions
    LOW = 2       # Background/batch processing


@dataclass
class InferenceRequest:
    """Single inference request with metadata.

    Attributes:
        request_id: Unique request identifier.
        model_name: Target model for inference.
        features: Input feature array.
        priority: Request priority level.
        future: Future for returning result.
        created_at: Request creation timestamp.
    """

    request_id: UUID = field(default_factory=uuid4)
    model_name: str = ""
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    priority: BatchPriority = BatchPriority.NORMAL
    future: Future | None = None
    created_at: float = field(default_factory=time.time)

    def __lt__(self, other: InferenceRequest) -> bool:
        """Compare by priority, then by creation time."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


@dataclass
class BatchResult:
    """Result of a batched inference.

    Attributes:
        request_id: Original request ID.
        predictions: Model output predictions.
        latency_ms: Time from request to result.
        batch_size: Size of batch this request was part of.
        model_name: Model that produced the result.
    """

    request_id: UUID
    predictions: np.ndarray
    latency_ms: float
    batch_size: int
    model_name: str


@dataclass
class ModelRegistration:
    """Registered model with its configuration.

    Attributes:
        model: The model object with predict() method.
        batch_size: Maximum batch size for this model.
        use_amp: Whether to use automatic mixed precision.
        device: Device for inference (cuda, cpu, mps).
        predict_method: Method to call for predictions.
    """

    model: Any
    batch_size: int = 64
    use_amp: bool = True
    device: str = "cuda"
    predict_method: str = "predict"


@dataclass
class BatcherMetrics:
    """Metrics for the inference batcher.

    Attributes:
        total_requests: Total number of requests processed.
        total_batches: Total number of batches processed.
        avg_batch_size: Running average batch size.
        avg_latency_ms: Running average latency.
        queue_depth: Current queue depth by model.
        throughput_per_second: Requests processed per second.
    """

    total_requests: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_latency_ms: float = 0.0
    queue_depth: dict[str, int] = field(default_factory=dict)
    throughput_per_second: float = 0.0
    _start_time: float = field(default_factory=time.time)

    def record_batch(self, batch_size: int, latency_ms: float) -> None:
        """Record a batch completion."""
        self.total_requests += batch_size
        self.total_batches += 1

        # Exponential moving average
        alpha = 0.1
        self.avg_batch_size = alpha * batch_size + (1 - alpha) * self.avg_batch_size
        self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms

        elapsed = time.time() - self._start_time
        if elapsed > 0:
            self.throughput_per_second = self.total_requests / elapsed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "avg_batch_size": round(self.avg_batch_size, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "queue_depth": dict(self.queue_depth),
            "throughput_per_second": round(self.throughput_per_second, 2),
        }


class InferenceBatcher:
    """
    GPU inference batcher for high-throughput predictions.

    Collects multiple prediction requests and batches them together
    for more efficient GPU processing. Supports multiple models with
    model-specific batch sizes and priorities.

    Features:
    - Dynamic batching with configurable timeouts
    - Priority queue for latency-sensitive requests
    - Automatic batch size optimization
    - Memory-aware batching
    - Multi-model support

    Example:
        batcher = InferenceBatcher(max_wait_ms=5.0)
        batcher.register_model("lstm", model, batch_size=64)

        # Async usage
        result = await batcher.predict("lstm", features)

        # Sync usage
        result = batcher.predict_sync("lstm", features)
    """

    _instance: InferenceBatcher | None = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> InferenceBatcher:
        """Singleton pattern for global batcher access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(
        self,
        max_wait_ms: float = 5.0,
        default_batch_size: int = 64,
        num_workers: int = 2,
        enable_metrics: bool = True,
    ) -> None:
        """
        Initialize the inference batcher.

        Args:
            max_wait_ms: Maximum time to wait for batch to fill.
            default_batch_size: Default batch size for models.
            num_workers: Number of worker threads for batch processing.
            enable_metrics: Whether to collect metrics.
        """
        if self._initialized:
            return

        self._initialized = True
        self._max_wait_ms = max_wait_ms
        self._default_batch_size = default_batch_size
        self._enable_metrics = enable_metrics

        # Model registry
        self._models: dict[str, ModelRegistration] = {}

        # Request queues (per model, priority-sorted)
        self._queues: dict[str, queue.PriorityQueue] = defaultdict(queue.PriorityQueue)
        self._queue_lock = threading.RLock()

        # Worker thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="inference_batcher",
        )

        # Background batch processor
        self._running = True
        self._process_thread = threading.Thread(
            target=self._batch_processor_loop,
            daemon=True,
            name="inference_batch_processor",
        )
        self._process_thread.start()

        # Metrics
        self._metrics = BatcherMetrics() if enable_metrics else None

        logger.info(
            f"InferenceBatcher initialized: max_wait={max_wait_ms}ms, "
            f"default_batch={default_batch_size}, workers={num_workers}"
        )

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
            cls._instance = None

    def register_model(
        self,
        name: str,
        model: Any,
        batch_size: int | None = None,
        use_amp: bool = True,
        device: str | None = None,
        predict_method: str = "predict",
    ) -> None:
        """
        Register a model for batched inference.

        Args:
            name: Unique model identifier.
            model: Model object with predict() method.
            batch_size: Maximum batch size (default: self._default_batch_size).
            use_amp: Whether to use automatic mixed precision.
            device: Device for inference (auto-detect if None).
            predict_method: Name of the prediction method.
        """
        if device is None:
            if hasattr(model, "device"):
                device = str(model.device)
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        registration = ModelRegistration(
            model=model,
            batch_size=batch_size or self._default_batch_size,
            use_amp=use_amp,
            device=device,
            predict_method=predict_method,
        )

        self._models[name] = registration
        logger.info(
            f"Registered model '{name}': batch_size={registration.batch_size}, "
            f"device={device}, amp={use_amp}"
        )

    def unregister_model(self, name: str) -> bool:
        """
        Unregister a model.

        Args:
            name: Model identifier.

        Returns:
            True if model was unregistered.
        """
        if name in self._models:
            del self._models[name]
            logger.info(f"Unregistered model '{name}'")
            return True
        return False

    async def predict(
        self,
        model_name: str,
        features: np.ndarray,
        priority: BatchPriority = BatchPriority.NORMAL,
        timeout: float | None = None,
    ) -> np.ndarray:
        """
        Submit prediction request for batching (async).

        Args:
            model_name: Target model name.
            features: Input features (1D or 2D array).
            priority: Request priority.
            timeout: Maximum time to wait for result.

        Returns:
            Model predictions.

        Raises:
            ValueError: If model not registered.
            TimeoutError: If timeout exceeded.
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not registered")

        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Create request
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        request = InferenceRequest(
            model_name=model_name,
            features=features,
            priority=priority,
            future=future,
        )

        # Add to queue
        with self._queue_lock:
            self._queues[model_name].put(request)
            if self._metrics:
                self._metrics.queue_depth[model_name] = self._queues[model_name].qsize()

        # Wait for result
        try:
            if timeout:
                result = await asyncio.wait_for(future, timeout=timeout)
            else:
                result = await future
            return result.predictions
        except asyncio.TimeoutError:
            logger.warning(f"Prediction timeout for model '{model_name}'")
            raise

    def predict_sync(
        self,
        model_name: str,
        features: np.ndarray,
        priority: BatchPriority = BatchPriority.NORMAL,
        timeout: float = 30.0,
    ) -> np.ndarray:
        """
        Submit prediction request for batching (sync).

        Args:
            model_name: Target model name.
            features: Input features.
            priority: Request priority.
            timeout: Maximum time to wait.

        Returns:
            Model predictions.
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not registered")

        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Create request with threading Future
        future: Future = Future()

        request = InferenceRequest(
            model_name=model_name,
            features=features,
            priority=priority,
            future=future,  # type: ignore
        )

        # Add to queue
        with self._queue_lock:
            self._queues[model_name].put(request)
            if self._metrics:
                self._metrics.queue_depth[model_name] = self._queues[model_name].qsize()

        # Wait for result
        try:
            result = future.result(timeout=timeout)
            return result.predictions
        except TimeoutError:
            logger.warning(f"Prediction timeout for model '{model_name}'")
            raise

    def predict_immediate(
        self,
        model_name: str,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Bypass batching for immediate prediction.

        Use this for single urgent requests where latency is critical.
        Does not benefit from batching but avoids queue wait time.

        Args:
            model_name: Target model name.
            features: Input features.

        Returns:
            Model predictions.
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not registered")

        registration = self._models[model_name]
        model = registration.model

        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Direct prediction
        predict_fn = getattr(model, registration.predict_method)
        return predict_fn(features)

    def _batch_processor_loop(self) -> None:
        """Background loop that processes batched requests."""
        while self._running:
            try:
                self._process_all_queues()
                time.sleep(self._max_wait_ms / 1000.0)
            except Exception as e:
                logger.error(f"Batch processor error: {e}")

    def _process_all_queues(self) -> None:
        """Process pending requests for all models."""
        for model_name in list(self._models.keys()):
            self._process_model_queue(model_name)

    def _process_model_queue(self, model_name: str) -> None:
        """Process pending requests for a specific model."""
        if model_name not in self._models:
            return

        registration = self._models[model_name]
        q = self._queues[model_name]

        # Collect batch
        batch: list[InferenceRequest] = []
        batch_start = time.time()

        with self._queue_lock:
            while not q.empty() and len(batch) < registration.batch_size:
                try:
                    request = q.get_nowait()
                    batch.append(request)
                except queue.Empty:
                    break

            if self._metrics:
                self._metrics.queue_depth[model_name] = q.qsize()

        if not batch:
            return

        # Process batch
        try:
            self._execute_batch(model_name, registration, batch)

            # Record metrics
            batch_latency = (time.time() - batch_start) * 1000
            if self._metrics:
                self._metrics.record_batch(len(batch), batch_latency)

        except Exception as e:
            logger.error(f"Batch execution error for '{model_name}': {e}")
            # Fail all requests in batch
            for request in batch:
                if request.future is not None:
                    if isinstance(request.future, Future):
                        request.future.set_exception(e)
                    elif asyncio.isfuture(request.future):
                        request.future.set_exception(e)

    def _execute_batch(
        self,
        model_name: str,
        registration: ModelRegistration,
        batch: list[InferenceRequest],
    ) -> None:
        """Execute batched inference."""
        start_time = time.time()

        # Stack features into batch
        batch_features = np.vstack([req.features for req in batch])

        # Get predict function
        model = registration.model
        predict_fn = getattr(model, registration.predict_method)

        # Execute with optional AMP
        if registration.use_amp and registration.device == "cuda":
            with torch.cuda.amp.autocast():
                batch_predictions = predict_fn(batch_features)
        else:
            batch_predictions = predict_fn(batch_features)

        # Ensure predictions are numpy
        if isinstance(batch_predictions, torch.Tensor):
            batch_predictions = batch_predictions.cpu().numpy()

        # Split results and fulfill futures
        latency_ms = (time.time() - start_time) * 1000
        idx = 0

        for request in batch:
            num_samples = request.features.shape[0]
            predictions = batch_predictions[idx:idx + num_samples]
            idx += num_samples

            result = BatchResult(
                request_id=request.request_id,
                predictions=predictions,
                latency_ms=latency_ms,
                batch_size=len(batch),
                model_name=model_name,
            )

            if request.future is not None:
                if isinstance(request.future, Future):
                    request.future.set_result(result)
                elif asyncio.isfuture(request.future):
                    # Schedule on event loop
                    loop = request.future.get_loop()
                    loop.call_soon_threadsafe(
                        request.future.set_result, result
                    )

        logger.debug(
            f"Batch completed: model={model_name}, size={len(batch)}, "
            f"latency={latency_ms:.2f}ms"
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get batcher metrics."""
        if self._metrics:
            return self._metrics.to_dict()
        return {}

    def get_registered_models(self) -> list[str]:
        """Get list of registered model names."""
        return list(self._models.keys())

    def get_queue_depth(self, model_name: str | None = None) -> dict[str, int]:
        """Get current queue depths."""
        with self._queue_lock:
            if model_name:
                return {model_name: self._queues[model_name].qsize()}
            return {name: q.qsize() for name, q in self._queues.items()}

    def flush(self, model_name: str | None = None) -> int:
        """
        Flush pending requests (process immediately).

        Args:
            model_name: Model to flush (all if None).

        Returns:
            Number of requests flushed.
        """
        count = 0
        models = [model_name] if model_name else list(self._models.keys())

        for name in models:
            with self._queue_lock:
                count += self._queues[name].qsize()
            self._process_model_queue(name)

        return count

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the batcher.

        Args:
            wait: Whether to wait for pending requests.
        """
        self._running = False

        if wait:
            # Process remaining requests
            for model_name in self._models:
                self._process_model_queue(model_name)

        self._executor.shutdown(wait=wait)
        logger.info("InferenceBatcher shutdown complete")


def get_inference_batcher(
    max_wait_ms: float = 5.0,
    default_batch_size: int = 64,
) -> InferenceBatcher:
    """
    Get the global inference batcher instance.

    Args:
        max_wait_ms: Maximum wait time for batching.
        default_batch_size: Default batch size.

    Returns:
        InferenceBatcher singleton.
    """
    return InferenceBatcher(
        max_wait_ms=max_wait_ms,
        default_batch_size=default_batch_size,
    )


# =============================================================================
# BATCH INFERENCE DECORATOR
# =============================================================================


def batched_inference(
    model_name: str,
    priority: BatchPriority = BatchPriority.NORMAL,
) -> Callable:
    """
    Decorator to enable batched inference for a function.

    The decorated function should accept features as the first argument
    and return predictions. The decorator routes calls through the
    global inference batcher for automatic batching.

    Args:
        model_name: Model name for batching.
        priority: Default priority for requests.

    Example:
        @batched_inference("lstm_v1")
        def predict_returns(features: np.ndarray) -> np.ndarray:
            return model.predict(features)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(features: np.ndarray, **kwargs) -> np.ndarray:
            batcher = get_inference_batcher()

            # Check if model registered
            if model_name not in batcher.get_registered_models():
                # Fall back to direct call
                return func(features, **kwargs)

            # Use batched inference
            return batcher.predict_sync(
                model_name=model_name,
                features=features,
                priority=priority,
            )

        async def async_wrapper(features: np.ndarray, **kwargs) -> np.ndarray:
            batcher = get_inference_batcher()

            if model_name not in batcher.get_registered_models():
                return func(features, **kwargs)

            return await batcher.predict(
                model_name=model_name,
                features=features,
                priority=priority,
            )

        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
