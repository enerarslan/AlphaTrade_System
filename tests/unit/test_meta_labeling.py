"""Unit tests for meta-labeling safeguards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_trading_system.models.meta_labeling import MetaLabelConfig, MetaLabeler


def test_meta_labeler_rejects_small_samples_without_temporal_holdout(monkeypatch) -> None:
    """Meta-labeler must not score itself on training data when samples are scarce."""
    labeler = MetaLabeler(config=MetaLabelConfig())
    X = pd.DataFrame(
        {
            "feat_1": np.linspace(0.0, 1.0, 20),
            "feat_2": np.linspace(1.0, 2.0, 20),
        }
    )
    signals = pd.Series(np.where(np.arange(20) % 2 == 0, 1.0, -1.0))
    prices = pd.Series(np.linspace(100.0, 110.0, 20))

    monkeypatch.setattr(
        labeler._labeler,
        "generate_labels",
        lambda prices, signals: pd.DataFrame({"label": np.tile([0, 1], 10)}),
    )
    monkeypatch.setattr(labeler, "_build_meta_features", lambda X, signals, prices: X)

    with pytest.raises(ValueError, match="unbiased temporal validation"):
        labeler.fit(X, signals, prices)
