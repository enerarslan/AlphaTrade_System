from __future__ import annotations

import numpy as np
import pandas as pd

from quant_trading_system.models.model_manager import FutureLeakValidator
from quant_trading_system.models.purged_cv import PurgedKFold, WalkForwardCV


def test_purged_kfold_uses_event_end_groups_to_purge_overlaps() -> None:
    X = np.zeros((6, 1), dtype=float)
    times = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
    event_ends = pd.to_datetime(
        [
            "2025-01-01T00:00:00Z",
            "2025-01-04T00:00:00Z",
            "2025-01-03T00:00:00Z",
            "2025-01-04T00:00:00Z",
            "2025-01-05T00:00:00Z",
            "2025-01-06T00:00:00Z",
        ],
        utc=True,
    )

    cv = PurgedKFold(n_splits=3, purge_gap=0, embargo_pct=0.01, prediction_horizon=1)
    folds = list(cv.split(X, times=times, groups=event_ends))

    train_idx, test_idx = folds[1]
    assert test_idx.tolist() == [2, 3]
    assert 1 not in train_idx.tolist()
    assert 4 not in train_idx.tolist()


def test_walk_forward_uses_supplied_time_order() -> None:
    X = np.zeros((5, 1), dtype=float)
    times = pd.to_datetime(
        [
            "2025-01-05T00:00:00Z",
            "2025-01-01T00:00:00Z",
            "2025-01-04T00:00:00Z",
            "2025-01-02T00:00:00Z",
            "2025-01-03T00:00:00Z",
        ],
        utc=True,
    )

    cv = WalkForwardCV(
        n_splits=2,
        train_pct=0.4,
        purge_gap=0,
        embargo_pct=0.01,
        prediction_horizon=1,
        min_train_size=1,
    )
    train_idx, test_idx = next(cv.split(X, times=times))

    assert train_idx.tolist() == [1]
    assert test_idx.tolist() == [4]


def test_validate_time_series_split_enforces_gap_with_reference_timeline() -> None:
    validator = FutureLeakValidator(strict_mode=True)
    reference = pd.to_datetime(
        [
            "2025-01-02T00:00:00Z",
            "2025-01-03T00:00:00Z",
            "2025-01-06T00:00:00Z",
            "2025-01-07T00:00:00Z",
        ],
        utc=True,
    ).to_numpy()

    valid, issues = validator.validate_time_series_split(
        train_timestamps=reference[:2],
        test_timestamps=reference[2:3],
        gap_bars=1,
        reference_timestamps=reference,
    )

    assert valid is False
    assert any("GAP VIOLATION" in issue for issue in issues)

    valid, issues = validator.validate_time_series_split(
        train_timestamps=reference[:2],
        test_timestamps=reference[3:4],
        gap_bars=1,
        reference_timestamps=reference,
    )

    assert valid is True
    assert issues == []
