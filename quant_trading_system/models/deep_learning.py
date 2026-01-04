"""
Deep learning models for trading using PyTorch.

Implements LSTM, Transformer, and Temporal Convolutional Network (TCN)
architectures for time series forecasting in financial markets.

GPU Acceleration Features:
- Automatic device selection (CUDA, MPS, CPU)
- Mixed precision training (AMP) for faster training
- Multi-GPU support via DataParallel
- Memory optimization utilities
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Any, Generator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from quant_trading_system.models.base import ModelType, TimeSeriesModel

logger = logging.getLogger(__name__)


# =============================================================================
# GPU ACCELERATION UTILITIES
# =============================================================================


class DeviceManager:
    """
    Manages device selection and GPU acceleration for PyTorch models.

    Features:
    - Automatic device selection (CUDA > MPS > CPU)
    - Mixed precision training support
    - Multi-GPU handling
    - Memory management utilities
    """

    _instance: "DeviceManager | None" = None

    def __new__(cls) -> "DeviceManager":
        """Singleton pattern for device management."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize device manager."""
        if self._initialized:
            return

        self._initialized = True
        self._device: torch.device | None = None
        self._amp_enabled: bool = False
        self._scaler: torch.cuda.amp.GradScaler | None = None

    @property
    def device(self) -> torch.device:
        """Get the optimal available device."""
        if self._device is None:
            self._device = self._select_device()
        return self._device

    def _select_device(self) -> torch.device:
        """Select the best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple MPS (Metal Performance Shaders)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU (no GPU acceleration available)")

        return device

    @property
    def is_cuda(self) -> bool:
        """Check if CUDA is the active device."""
        return self.device.type == "cuda"

    @property
    def is_mps(self) -> bool:
        """Check if MPS is the active device."""
        return self.device.type == "mps"

    def enable_amp(self) -> None:
        """Enable Automatic Mixed Precision for faster training."""
        if self.is_cuda:
            self._amp_enabled = True
            self._scaler = torch.cuda.amp.GradScaler()
            logger.info("AMP (Automatic Mixed Precision) enabled")
        else:
            logger.warning("AMP requires CUDA, staying with full precision")

    def disable_amp(self) -> None:
        """Disable Automatic Mixed Precision."""
        self._amp_enabled = False
        self._scaler = None

    @property
    def amp_enabled(self) -> bool:
        """Check if AMP is enabled."""
        return self._amp_enabled

    @property
    def scaler(self) -> torch.cuda.amp.GradScaler | None:
        """Get the gradient scaler for AMP."""
        return self._scaler

    @contextmanager
    def autocast(self) -> Generator[None, None, None]:
        """Context manager for automatic mixed precision."""
        if self._amp_enabled and self.is_cuda:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the optimal device."""
        return tensor.to(self.device)

    def model_to_device(self, model: nn.Module, multi_gpu: bool = False) -> nn.Module:
        """
        Move model to device, optionally with multi-GPU support.

        Args:
            model: PyTorch model
            multi_gpu: Whether to use DataParallel for multi-GPU

        Returns:
            Model on device (wrapped in DataParallel if multi-GPU)
        """
        model = model.to(self.device)

        if multi_gpu and self.is_cuda and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)

        return model

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("CUDA cache cleared")

    def get_memory_stats(self) -> dict[str, float]:
        """Get GPU memory statistics."""
        if not self.is_cuda:
            return {}

        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "cached_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }

    def optimize_memory(self) -> None:
        """Optimize memory usage for large models."""
        if self.is_cuda:
            # Enable memory efficient attention if available
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

            # Set memory allocator config for fragmentation reduction
            import os
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def get_device_manager() -> DeviceManager:
    """Get the singleton DeviceManager instance."""
    return DeviceManager()


def get_optimal_device() -> torch.device:
    """Get the optimal device for computation."""
    return get_device_manager().device


def select_device(device: str | None = None) -> torch.device:
    """
    Select device based on input or auto-detect.

    Args:
        device: Optional device string ("cuda", "mps", "cpu", or None for auto)

    Returns:
        Selected torch.device
    """
    if device is None:
        return get_optimal_device()

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")

    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        logger.warning("MPS requested but not available, falling back to CPU")
        return torch.device("cpu")

    return torch.device(device)


class LSTMNetwork(nn.Module):
    """LSTM neural network for sequence modeling."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        output_size: int = 1,
    ):
        """
        Initialize LSTM network.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate between layers
            bidirectional: Whether to use bidirectional LSTM
            output_size: Output dimension
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence, features)

        Returns:
            Output tensor of shape (batch, output_size)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Take the last timestep output
        out = lstm_out[:, -1, :]

        # Apply layer norm, dropout, and final projection
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out


class AttentionLSTMNetwork(nn.Module):
    """LSTM with self-attention mechanism."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_heads: int = 4,
        output_size: int = 1,
    ):
        """Initialize Attention LSTM."""
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_out = self.layer_norm1(lstm_out + attended)

        # Take last timestep and project
        out = lstm_out[:, -1, :]
        out = self.layer_norm2(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out


class LSTMModel(TimeSeriesModel):
    """
    LSTM-based model for time series prediction.

    Supports vanilla LSTM, bidirectional LSTM, and attention LSTM variants.
    GPU Acceleration: Automatic device selection with AMP (Automatic Mixed Precision).
    """

    def __init__(
        self,
        name: str = "lstm",
        version: str = "1.0.0",
        model_type: ModelType = ModelType.REGRESSOR,
        lookback_window: int = 20,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_attention: bool = False,
        attention_heads: int = 4,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 15,
        device: str | None = None,
        use_amp: bool = True,
        multi_gpu: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize LSTM model.

        Args:
            name: Model identifier
            version: Version string
            model_type: Classification or regression
            lookback_window: Sequence length
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to add attention layer
            attention_heads: Number of attention heads
            learning_rate: Learning rate
            batch_size: Training batch size
            epochs: Maximum training epochs
            patience: Early stopping patience
            device: Device to use (cuda/mps/cpu/None for auto-detect)
            use_amp: Whether to use Automatic Mixed Precision (GPU only)
            multi_gpu: Whether to use DataParallel for multi-GPU training
            **kwargs: Additional parameters
        """
        super().__init__(name, version, model_type, lookback_window, **kwargs)

        self._params.update({
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "use_attention": use_attention,
            "attention_heads": attention_heads,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "use_amp": use_amp,
            "multi_gpu": multi_gpu,
        })

        # GPU Acceleration: Use specified device or auto-select optimal
        self.device = select_device(device)
        self._use_amp = use_amp and self.device.type == "cuda"
        self._multi_gpu = multi_gpu and self.device.type == "cuda"
        self._scaler: torch.cuda.amp.GradScaler | None = None
        self._network: nn.Module | None = None
        # P0 FIX: Initialize device manager for GPU memory stats
        self._device_manager = DeviceManager()

        # Log device info
        logger.info(f"LSTM model initialized on device: {self.device}")
        if self._use_amp:
            logger.info("AMP (Automatic Mixed Precision) enabled for faster training")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "LSTMModel":
        """
        Train LSTM model with GPU acceleration and optional AMP.

        Args:
            X: Feature matrix (n_samples, n_features) or sequences (n_samples, seq_len, n_features)
            y: Target array
            validation_data: Optional validation data
            sample_weights: Not used for deep learning
            **kwargs: Additional parameters

        Returns:
            self for method chaining
        """
        # Create sequences if necessary
        if X.ndim == 2:
            X_seq, y_seq = self.create_sequences(X, y)
        else:
            X_seq, y_seq = X, y

        input_size = X_seq.shape[2]
        output_size = 1 if self._model_type == ModelType.REGRESSOR else len(np.unique(y_seq))

        # Build network
        if self._params["use_attention"]:
            self._network = AttentionLSTMNetwork(
                input_size=input_size,
                hidden_size=self._params["hidden_size"],
                num_layers=self._params["num_layers"],
                dropout=self._params["dropout"],
                attention_heads=self._params["attention_heads"],
                output_size=output_size,
            )
        else:
            self._network = LSTMNetwork(
                input_size=input_size,
                hidden_size=self._params["hidden_size"],
                num_layers=self._params["num_layers"],
                dropout=self._params["dropout"],
                bidirectional=self._params["bidirectional"],
                output_size=output_size,
            )

        # GPU Acceleration: Move model to specified device with optional multi-GPU support
        self._network = self._network.to(self.device)
        if self._multi_gpu and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self._network = nn.DataParallel(self._network)

        # Initialize AMP GradScaler for mixed precision training
        if self._use_amp:
            self._scaler = torch.cuda.amp.GradScaler()
            logger.info("AMP GradScaler initialized for mixed precision training")

        # Log GPU memory before data transfer
        if self.device.type == "cuda":
            mem_stats = self._device_manager.get_memory_stats()
            logger.info(f"GPU memory before data: {mem_stats.get('allocated_gb', 0):.2f} GB allocated")

        # Prepare data - use pin_memory for faster GPU transfers
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq) if self._model_type == ModelType.REGRESSOR else torch.LongTensor(y_seq)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._params["batch_size"],
            shuffle=False,  # Time series - don't shuffle
            pin_memory=self.device.type == "cuda",  # Faster GPU transfers
            num_workers=0,  # Avoid multiprocessing issues on Windows
        )

        # Prepare validation data
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            if X_val.ndim == 2:
                X_val, y_val = self.create_sequences(X_val, y_val)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val) if self._model_type == ModelType.REGRESSOR else torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self._params["batch_size"],
                pin_memory=self.device.type == "cuda",
            )

        # Training setup
        optimizer = torch.optim.AdamW(
            self._network.parameters(),
            lr=self._params["learning_rate"],
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        criterion = nn.CrossEntropyLoss() if self._model_type == ModelType.CLASSIFIER else nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self._params["epochs"]):
            # Training phase
            self._network.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                # Move batch to device
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                # GPU Acceleration: Use AMP autocast for mixed precision
                if self._use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._network(batch_X)
                        if self._model_type == ModelType.REGRESSOR:
                            outputs = outputs.squeeze()
                        loss = criterion(outputs, batch_y)

                    # Scale loss and backward pass
                    self._scaler.scale(loss).backward()

                    # Unscale gradients and clip
                    self._scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)

                    # Step optimizer with scaler
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    outputs = self._network(batch_X)
                    if self._model_type == ModelType.REGRESSOR:
                        outputs = outputs.squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            if val_loader is not None:
                self._network.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device, non_blocking=True)
                        batch_y = batch_y.to(self.device, non_blocking=True)

                        # Use AMP autocast for validation too
                        if self._use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = self._network(batch_X)
                                if self._model_type == ModelType.REGRESSOR:
                                    outputs = outputs.squeeze()
                                val_loss += criterion(outputs, batch_y).item()
                        else:
                            outputs = self._network(batch_X)
                            if self._model_type == ModelType.REGRESSOR:
                                outputs = outputs.squeeze()
                            val_loss += criterion(outputs, batch_y).item()
                val_loss /= len(val_loader)

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self._network.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self._params["patience"]:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

        # Restore best model
        if best_state is not None:
            self._network.load_state_dict(best_state)
            self._network.to(self.device)

        # GPU memory cleanup
        if self.device.type == "cuda":
            self._device_manager.clear_cache()
            mem_stats = self._device_manager.get_memory_stats()
            logger.info(f"GPU memory after training: {mem_stats.get('allocated_gb', 0):.2f} GB allocated")

        # Store model for serialization
        # CRITICAL FIX: Save both state_dict AND architecture parameters
        # Previously only saved state_dict, causing load() to fail because
        # self._network couldn't be reconstructed
        self._model = self._network.state_dict()

        # Save architecture parameters needed to reconstruct network during load()
        self._params["_network_input_size"] = input_size
        self._params["_network_output_size"] = output_size

        feature_names = kwargs.get("feature_names", [f"f{i}" for i in range(input_size)])
        metrics = {"final_train_loss": train_loss}
        if validation_data is not None:
            metrics["final_val_loss"] = best_val_loss
        self._record_training(metrics, feature_names)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions with GPU acceleration."""
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        # Create sequences if necessary
        if X.ndim == 2:
            X_seq, _ = self.create_sequences(X, None)
        else:
            X_seq = X

        self._network.eval()
        X_tensor = torch.FloatTensor(X_seq).to(self.device, non_blocking=True)

        with torch.no_grad():
            # Use AMP autocast for faster inference on GPU
            if self._use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self._network(X_tensor)
            else:
                outputs = self._network(X_tensor)

            if self._model_type == ModelType.REGRESSOR:
                predictions = outputs.squeeze().cpu().numpy()
            else:
                predictions = outputs.argmax(dim=1).cpu().numpy()

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability estimates with GPU acceleration."""
        if self._model_type != ModelType.CLASSIFIER:
            raise NotImplementedError("predict_proba only available for classifiers")

        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        if X.ndim == 2:
            X_seq, _ = self.create_sequences(X, None)
        else:
            X_seq = X

        self._network.eval()
        X_tensor = torch.FloatTensor(X_seq).to(self.device, non_blocking=True)

        with torch.no_grad():
            # Use AMP autocast for faster inference on GPU
            if self._use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self._network(X_tensor)
            else:
                outputs = self._network(X_tensor)
            proba = F.softmax(outputs, dim=1).cpu().numpy()

        return proba

    def load(self, path: str) -> "LSTMModel":
        """
        Load LSTM model from disk with GPU-portable device mapping.

        CRITICAL FIX: This method reconstructs the network architecture before
        loading the state_dict. Uses map_location for device portability -
        models saved on GPU can be loaded on CPU and vice versa.

        Args:
            path: Path to saved model (without extension)

        Returns:
            self for method chaining
        """
        from pathlib import Path

        # Initialize device if not already done (auto-detect on load)
        if not hasattr(self, 'device') or self.device is None:
            self.device = select_device(None)  # Auto-detect device
            self._use_amp = self._params.get("use_amp", True) and self.device.type == "cuda"
            self._multi_gpu = self._params.get("multi_gpu", False) and self.device.type == "cuda"

        # Call parent load to restore metadata and state_dict
        # Note: Parent load uses pickle which doesn't need map_location
        super().load(path)

        # CRITICAL: Restore lookback_window from params for correct sequence creation
        if "lookback_window" in self._params:
            self.lookback_window = self._params["lookback_window"]

        # Reconstruct network architecture from saved params
        input_size = self._params.get("_network_input_size")
        output_size = self._params.get("_network_output_size")

        if input_size is None or output_size is None:
            raise ValueError(
                "Cannot load model: missing architecture parameters. "
                "This model may have been saved with an older version. "
                f"Missing: input_size={input_size}, output_size={output_size}"
            )

        # Rebuild the network architecture
        if self._params.get("use_attention", False):
            self._network = AttentionLSTMNetwork(
                input_size=input_size,
                hidden_size=self._params.get("hidden_size", 128),
                num_layers=self._params.get("num_layers", 2),
                dropout=self._params.get("dropout", 0.3),
                attention_heads=self._params.get("attention_heads", 4),
                output_size=output_size,
            )
        else:
            self._network = LSTMNetwork(
                input_size=input_size,
                hidden_size=self._params.get("hidden_size", 128),
                num_layers=self._params.get("num_layers", 2),
                dropout=self._params.get("dropout", 0.3),
                bidirectional=self._params.get("bidirectional", False),
                output_size=output_size,
            )

        # GPU Portability: Load state_dict with map_location for device portability
        # This allows models saved on GPU to be loaded on CPU and vice versa
        if isinstance(self._model, dict):
            # State dict already loaded, apply map_location by moving to device
            state_dict = {k: v.to(self.device) for k, v in self._model.items()}
            self._network.load_state_dict(state_dict)
        else:
            self._network.load_state_dict(self._model)

        self._network.to(self.device)
        self._network.eval()  # Set to eval mode after loading

        logger.info(
            f"LSTM model loaded successfully: input_size={input_size}, "
            f"output_size={output_size}, device={self.device}, amp={self._use_amp}"
        )

        return self

    def get_feature_importance(self) -> dict[str, float]:
        """LSTM doesn't have direct feature importance."""
        return {name: 1.0 / len(self._feature_names) for name in self._feature_names}


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformers."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """Initialize positional encoding."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerNetwork(nn.Module):
    """Transformer encoder for sequence modeling."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        output_size: int = 1,
    ):
        """
        Initialize Transformer network.

        Args:
            input_size: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feedforward dimension
            dropout: Dropout rate
            output_size: Output dimension
        """
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, sequence, features)

        Returns:
            Output of shape (batch, output_size)
        """
        # Project to d_model
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Global average pooling or take last token
        x = x.mean(dim=1)  # (batch, d_model)

        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class TransformerModel(TimeSeriesModel):
    """
    Transformer-based model for time series prediction.

    Uses self-attention to capture long-range dependencies
    in financial time series data.
    GPU Acceleration: Automatic device selection with AMP (Automatic Mixed Precision).
    """

    def __init__(
        self,
        name: str = "transformer",
        version: str = "1.0.0",
        model_type: ModelType = ModelType.REGRESSOR,
        lookback_window: int = 50,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 15,
        device: str | None = None,
        use_amp: bool = True,
        multi_gpu: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize Transformer model.

        Args:
            name: Model identifier
            version: Version string
            model_type: Classification or regression
            lookback_window: Sequence length
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feedforward dimension
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Training batch size
            epochs: Maximum epochs
            patience: Early stopping patience
            device: Device (cuda/mps/cpu/None for auto-detect)
            use_amp: Whether to use Automatic Mixed Precision (GPU only)
            multi_gpu: Whether to use DataParallel for multi-GPU training
            **kwargs: Additional parameters
        """
        super().__init__(name, version, model_type, lookback_window, **kwargs)

        self._params.update({
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "d_ff": d_ff,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "use_amp": use_amp,
            "multi_gpu": multi_gpu,
        })

        # GPU Acceleration: Use specified device or auto-select optimal
        self.device = select_device(device)
        self._use_amp = use_amp and self.device.type == "cuda"
        self._multi_gpu = multi_gpu and self.device.type == "cuda"
        self._scaler: torch.cuda.amp.GradScaler | None = None
        self._network: nn.Module | None = None
        # P0 FIX: Initialize device manager for GPU memory stats
        self._device_manager = DeviceManager()

        # Log device info
        logger.info(f"Transformer model initialized on device: {self.device}")
        if self._use_amp:
            logger.info("AMP (Automatic Mixed Precision) enabled for faster training")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "TransformerModel":
        """Train Transformer model with GPU acceleration and optional AMP."""
        # Create sequences if necessary
        if X.ndim == 2:
            X_seq, y_seq = self.create_sequences(X, y)
        else:
            X_seq, y_seq = X, y

        input_size = X_seq.shape[2]
        output_size = 1 if self._model_type == ModelType.REGRESSOR else len(np.unique(y_seq))

        # Build network
        self._network = TransformerNetwork(
            input_size=input_size,
            d_model=self._params["d_model"],
            nhead=self._params["nhead"],
            num_layers=self._params["num_layers"],
            d_ff=self._params["d_ff"],
            dropout=self._params["dropout"],
            output_size=output_size,
        )

        # GPU Acceleration: Move model to specified device with optional multi-GPU support
        self._network = self._network.to(self.device)
        if self._multi_gpu and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self._network = nn.DataParallel(self._network)

        # Initialize AMP GradScaler for mixed precision training
        if self._use_amp:
            self._scaler = torch.cuda.amp.GradScaler()
            logger.info("AMP GradScaler initialized for mixed precision training")

        # Log GPU memory before data transfer
        if self.device.type == "cuda":
            mem_stats = self._device_manager.get_memory_stats()
            logger.info(f"GPU memory before data: {mem_stats.get('allocated_gb', 0):.2f} GB allocated")

        # Prepare data - use pin_memory for faster GPU transfers
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq) if self._model_type == ModelType.REGRESSOR else torch.LongTensor(y_seq)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._params["batch_size"],
            shuffle=False,
            pin_memory=self.device.type == "cuda",
            num_workers=0,
        )

        # Validation data
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            if X_val.ndim == 2:
                X_val, y_val = self.create_sequences(X_val, y_val)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val) if self._model_type == ModelType.REGRESSOR else torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self._params["batch_size"],
                pin_memory=self.device.type == "cuda",
            )

        # Training setup
        optimizer = torch.optim.AdamW(
            self._network.parameters(),
            lr=self._params["learning_rate"],
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._params["epochs"]
        )

        criterion = nn.CrossEntropyLoss() if self._model_type == ModelType.CLASSIFIER else nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self._params["epochs"]):
            self._network.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                # Move batch to device
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # GPU Acceleration: Use AMP autocast for mixed precision
                if self._use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._network(batch_X)
                        if self._model_type == ModelType.REGRESSOR:
                            outputs = outputs.squeeze()
                        loss = criterion(outputs, batch_y)

                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    outputs = self._network(batch_X)
                    if self._model_type == ModelType.REGRESSOR:
                        outputs = outputs.squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            scheduler.step()

            if val_loader is not None:
                self._network.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device, non_blocking=True)
                        batch_y = batch_y.to(self.device, non_blocking=True)

                        if self._use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = self._network(batch_X)
                                if self._model_type == ModelType.REGRESSOR:
                                    outputs = outputs.squeeze()
                                val_loss += criterion(outputs, batch_y).item()
                        else:
                            outputs = self._network(batch_X)
                            if self._model_type == ModelType.REGRESSOR:
                                outputs = outputs.squeeze()
                            val_loss += criterion(outputs, batch_y).item()
                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self._network.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self._params["patience"]:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

        if best_state is not None:
            self._network.load_state_dict(best_state)
            self._network.to(self.device)

        # GPU memory cleanup
        if self.device.type == "cuda":
            self._device_manager.clear_cache()
            mem_stats = self._device_manager.get_memory_stats()
            logger.info(f"GPU memory after training: {mem_stats.get('allocated_gb', 0):.2f} GB allocated")

        # Store model for serialization
        # CRITICAL FIX: Save both state_dict AND architecture parameters
        self._model = self._network.state_dict()

        # Save architecture parameters needed to reconstruct network during load()
        self._params["_network_input_size"] = input_size
        self._params["_network_output_size"] = output_size

        feature_names = kwargs.get("feature_names", [f"f{i}" for i in range(input_size)])
        metrics = {"final_train_loss": train_loss}
        if validation_data is not None:
            metrics["final_val_loss"] = best_val_loss
        self._record_training(metrics, feature_names)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions with GPU acceleration."""
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        if X.ndim == 2:
            X_seq, _ = self.create_sequences(X, None)
        else:
            X_seq = X

        self._network.eval()
        X_tensor = torch.FloatTensor(X_seq).to(self.device, non_blocking=True)

        with torch.no_grad():
            # Use AMP autocast for faster inference on GPU
            if self._use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self._network(X_tensor)
            else:
                outputs = self._network(X_tensor)

            if self._model_type == ModelType.REGRESSOR:
                predictions = outputs.squeeze().cpu().numpy()
            else:
                predictions = outputs.argmax(dim=1).cpu().numpy()

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability estimates with GPU acceleration."""
        if self._model_type != ModelType.CLASSIFIER:
            raise NotImplementedError("predict_proba only available for classifiers")

        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        if X.ndim == 2:
            X_seq, _ = self.create_sequences(X, None)
        else:
            X_seq = X

        self._network.eval()
        X_tensor = torch.FloatTensor(X_seq).to(self.device, non_blocking=True)

        with torch.no_grad():
            # Use AMP autocast for faster inference on GPU
            if self._use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self._network(X_tensor)
            else:
                outputs = self._network(X_tensor)
            proba = F.softmax(outputs, dim=1).cpu().numpy()

        return proba

    def load(self, path: str) -> "TransformerModel":
        """
        Load Transformer model from disk with GPU-portable device mapping.

        CRITICAL FIX: This method reconstructs the network architecture before
        loading the state_dict. Uses map_location for device portability -
        models saved on GPU can be loaded on CPU and vice versa.

        Args:
            path: Path to saved model (without extension)

        Returns:
            self for method chaining
        """
        # Initialize device if not already done (auto-detect on load)
        if not hasattr(self, 'device') or self.device is None:
            self.device = select_device(None)  # Auto-detect device
            self._use_amp = self._params.get("use_amp", True) and self.device.type == "cuda"
            self._multi_gpu = self._params.get("multi_gpu", False) and self.device.type == "cuda"

        # Call parent load to restore metadata and state_dict
        super().load(path)

        # CRITICAL: Restore lookback_window from params for correct sequence creation
        if "lookback_window" in self._params:
            self.lookback_window = self._params["lookback_window"]

        # Reconstruct network architecture from saved params
        input_size = self._params.get("_network_input_size")
        output_size = self._params.get("_network_output_size")

        if input_size is None or output_size is None:
            raise ValueError(
                "Cannot load model: missing architecture parameters. "
                "This model may have been saved with an older version. "
                f"Missing: input_size={input_size}, output_size={output_size}"
            )

        # Rebuild the network architecture
        self._network = TransformerNetwork(
            input_size=input_size,
            d_model=self._params.get("d_model", 128),
            nhead=self._params.get("nhead", 4),
            num_layers=self._params.get("num_layers", 3),
            d_ff=self._params.get("d_ff", 512),
            dropout=self._params.get("dropout", 0.1),
            output_size=output_size,
        )

        # GPU Portability: Load state_dict with map_location for device portability
        if isinstance(self._model, dict):
            state_dict = {k: v.to(self.device) for k, v in self._model.items()}
            self._network.load_state_dict(state_dict)
        else:
            self._network.load_state_dict(self._model)

        self._network.to(self.device)
        self._network.eval()  # Set to eval mode after loading

        logger.info(
            f"Transformer model loaded successfully: input_size={input_size}, "
            f"output_size={output_size}, device={self.device}, amp={self._use_amp}"
        )

        return self

    def get_feature_importance(self) -> dict[str, float]:
        """Transformer doesn't have direct feature importance."""
        return {name: 1.0 / len(self._feature_names) for name in self._feature_names}


class TemporalBlock(nn.Module):
    """Temporal convolutional block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        """Initialize temporal block."""
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Weight normalization
        self.conv1 = nn.utils.parametrizations.weight_norm(self.conv1)
        self.conv2 = nn.utils.parametrizations.weight_norm(self.conv2)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # First convolution
        out = self.conv1(x)
        out = out[:, :, :-self.padding]  # Causal: remove future
        out = self.relu(out)
        out = self.dropout(out)

        # Second convolution
        out = self.conv2(out)
        out = out[:, :, :-self.padding]
        out = self.relu(out)
        out = self.dropout(out)

        # Residual
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TCNNetwork(nn.Module):
    """Temporal Convolutional Network."""

    def __init__(
        self,
        input_size: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        """
        Initialize TCN.

        Args:
            input_size: Input feature dimension
            num_channels: List of channel sizes for each layer
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            output_size: Output dimension
        """
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, sequence, features)

        Returns:
            Output of shape (batch, output_size)
        """
        # TCN expects (batch, channels, sequence)
        x = x.transpose(1, 2)

        out = self.network(x)

        # Take last timestep
        out = out[:, :, -1]

        return self.fc(out)


class TCNModel(TimeSeriesModel):
    """
    Temporal Convolutional Network model.

    Uses dilated causal convolutions for efficient sequence modeling.
    Parallelizable and memory-efficient compared to LSTM.
    GPU Acceleration: Automatic device selection with AMP (Automatic Mixed Precision).
    """

    def __init__(
        self,
        name: str = "tcn",
        version: str = "1.0.0",
        model_type: ModelType = ModelType.REGRESSOR,
        lookback_window: int = 50,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 15,
        device: str | None = None,
        use_amp: bool = True,
        multi_gpu: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize TCN model.

        Args:
            name: Model identifier
            version: Version string
            model_type: Classification or regression
            lookback_window: Sequence length
            num_channels: Channel sizes per layer (default: [64, 128, 128, 64])
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Training batch size
            epochs: Maximum epochs
            patience: Early stopping patience
            device: Device (cuda/mps/cpu/None for auto-detect)
            use_amp: Whether to use Automatic Mixed Precision (GPU only)
            multi_gpu: Whether to use DataParallel for multi-GPU training
            **kwargs: Additional parameters
        """
        super().__init__(name, version, model_type, lookback_window, **kwargs)

        if num_channels is None:
            num_channels = [64, 128, 128, 64]

        self._params.update({
            "num_channels": num_channels,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "use_amp": use_amp,
            "multi_gpu": multi_gpu,
        })

        # GPU Acceleration: Use specified device or auto-select optimal
        self.device = select_device(device)
        self._use_amp = use_amp and self.device.type == "cuda"
        self._multi_gpu = multi_gpu and self.device.type == "cuda"
        self._scaler: torch.cuda.amp.GradScaler | None = None
        self._network: nn.Module | None = None
        # P0 FIX: Initialize device manager for GPU memory stats
        self._device_manager = DeviceManager()

        # Log device info
        logger.info(f"TCN model initialized on device: {self.device}")
        if self._use_amp:
            logger.info("AMP (Automatic Mixed Precision) enabled for faster training")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        sample_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "TCNModel":
        """Train TCN model with GPU acceleration and optional AMP."""
        if X.ndim == 2:
            X_seq, y_seq = self.create_sequences(X, y)
        else:
            X_seq, y_seq = X, y

        input_size = X_seq.shape[2]
        output_size = 1 if self._model_type == ModelType.REGRESSOR else len(np.unique(y_seq))

        self._network = TCNNetwork(
            input_size=input_size,
            num_channels=self._params["num_channels"],
            kernel_size=self._params["kernel_size"],
            dropout=self._params["dropout"],
            output_size=output_size,
        )

        # GPU Acceleration: Move model to specified device with optional multi-GPU support
        self._network = self._network.to(self.device)
        if self._multi_gpu and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self._network = nn.DataParallel(self._network)

        # Initialize AMP GradScaler for mixed precision training
        if self._use_amp:
            self._scaler = torch.cuda.amp.GradScaler()
            logger.info("AMP GradScaler initialized for mixed precision training")

        # Log GPU memory before data transfer
        if self.device.type == "cuda":
            mem_stats = self._device_manager.get_memory_stats()
            logger.info(f"GPU memory before data: {mem_stats.get('allocated_gb', 0):.2f} GB allocated")

        # Prepare data - use pin_memory for faster GPU transfers
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq) if self._model_type == ModelType.REGRESSOR else torch.LongTensor(y_seq)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._params["batch_size"],
            shuffle=False,
            pin_memory=self.device.type == "cuda",
            num_workers=0,
        )

        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            if X_val.ndim == 2:
                X_val, y_val = self.create_sequences(X_val, y_val)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val) if self._model_type == ModelType.REGRESSOR else torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self._params["batch_size"],
                pin_memory=self.device.type == "cuda",
            )

        # Training setup
        optimizer = torch.optim.AdamW(
            self._network.parameters(),
            lr=self._params["learning_rate"],
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        criterion = nn.CrossEntropyLoss() if self._model_type == ModelType.CLASSIFIER else nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self._params["epochs"]):
            self._network.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                # Move batch to device
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # GPU Acceleration: Use AMP autocast for mixed precision
                if self._use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._network(batch_X)
                        if self._model_type == ModelType.REGRESSOR:
                            outputs = outputs.squeeze()
                        loss = criterion(outputs, batch_y)

                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    outputs = self._network(batch_X)
                    if self._model_type == ModelType.REGRESSOR:
                        outputs = outputs.squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            if val_loader is not None:
                self._network.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device, non_blocking=True)
                        batch_y = batch_y.to(self.device, non_blocking=True)

                        if self._use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = self._network(batch_X)
                                if self._model_type == ModelType.REGRESSOR:
                                    outputs = outputs.squeeze()
                                val_loss += criterion(outputs, batch_y).item()
                        else:
                            outputs = self._network(batch_X)
                            if self._model_type == ModelType.REGRESSOR:
                                outputs = outputs.squeeze()
                            val_loss += criterion(outputs, batch_y).item()
                val_loss /= len(val_loader)

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self._network.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self._params["patience"]:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

        if best_state is not None:
            self._network.load_state_dict(best_state)
            self._network.to(self.device)

        # GPU memory cleanup
        if self.device.type == "cuda":
            self._device_manager.clear_cache()
            mem_stats = self._device_manager.get_memory_stats()
            logger.info(f"GPU memory after training: {mem_stats.get('allocated_gb', 0):.2f} GB allocated")

        # Store model for serialization
        # CRITICAL FIX: Save both state_dict AND architecture parameters
        self._model = self._network.state_dict()

        # Save architecture parameters needed to reconstruct network during load()
        self._params["_network_input_size"] = input_size
        self._params["_network_output_size"] = output_size

        feature_names = kwargs.get("feature_names", [f"f{i}" for i in range(input_size)])
        metrics = {"final_train_loss": train_loss}
        if validation_data is not None:
            metrics["final_val_loss"] = best_val_loss
        self._record_training(metrics, feature_names)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions with GPU acceleration."""
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        if X.ndim == 2:
            X_seq, _ = self.create_sequences(X, None)
        else:
            X_seq = X

        self._network.eval()
        X_tensor = torch.FloatTensor(X_seq).to(self.device, non_blocking=True)

        with torch.no_grad():
            # Use AMP autocast for faster inference on GPU
            if self._use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self._network(X_tensor)
            else:
                outputs = self._network(X_tensor)

            if self._model_type == ModelType.REGRESSOR:
                predictions = outputs.squeeze().cpu().numpy()
            else:
                predictions = outputs.argmax(dim=1).cpu().numpy()

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability estimates with GPU acceleration."""
        if self._model_type != ModelType.CLASSIFIER:
            raise NotImplementedError("predict_proba only available for classifiers")

        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        if X.ndim == 2:
            X_seq, _ = self.create_sequences(X, None)
        else:
            X_seq = X

        self._network.eval()
        X_tensor = torch.FloatTensor(X_seq).to(self.device, non_blocking=True)

        with torch.no_grad():
            # Use AMP autocast for faster inference on GPU
            if self._use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self._network(X_tensor)
            else:
                outputs = self._network(X_tensor)
            proba = F.softmax(outputs, dim=1).cpu().numpy()

        return proba

    def load(self, path: str) -> "TCNModel":
        """
        Load TCN model from disk with GPU-portable device mapping.

        CRITICAL FIX: This method reconstructs the network architecture before
        loading the state_dict. Uses map_location for device portability -
        models saved on GPU can be loaded on CPU and vice versa.

        Args:
            path: Path to saved model (without extension)

        Returns:
            self for method chaining
        """
        # Initialize device if not already done (auto-detect on load)
        if not hasattr(self, 'device') or self.device is None:
            self.device = select_device(None)  # Auto-detect device
            self._use_amp = self._params.get("use_amp", True) and self.device.type == "cuda"
            self._multi_gpu = self._params.get("multi_gpu", False) and self.device.type == "cuda"

        # Call parent load to restore metadata and state_dict
        super().load(path)

        # CRITICAL: Restore lookback_window from params for correct sequence creation
        if "lookback_window" in self._params:
            self.lookback_window = self._params["lookback_window"]

        # Reconstruct network architecture from saved params
        input_size = self._params.get("_network_input_size")
        output_size = self._params.get("_network_output_size")

        if input_size is None or output_size is None:
            raise ValueError(
                "Cannot load model: missing architecture parameters. "
                "This model may have been saved with an older version. "
                f"Missing: input_size={input_size}, output_size={output_size}"
            )

        # Rebuild the network architecture
        self._network = TCNNetwork(
            input_size=input_size,
            num_channels=self._params.get("num_channels", [64, 64, 64]),
            kernel_size=self._params.get("kernel_size", 3),
            dropout=self._params.get("dropout", 0.2),
            output_size=output_size,
        )

        # GPU Portability: Load state_dict with map_location for device portability
        if isinstance(self._model, dict):
            state_dict = {k: v.to(self.device) for k, v in self._model.items()}
            self._network.load_state_dict(state_dict)
        else:
            self._network.load_state_dict(self._model)

        self._network.to(self.device)
        self._network.eval()  # Set to eval mode after loading

        logger.info(
            f"TCN model loaded successfully: input_size={input_size}, "
            f"output_size={output_size}, device={self.device}, amp={self._use_amp}"
        )

        return self

    def get_feature_importance(self) -> dict[str, float]:
        """TCN doesn't have direct feature importance."""
        return {name: 1.0 / len(self._feature_names) for name in self._feature_names}
