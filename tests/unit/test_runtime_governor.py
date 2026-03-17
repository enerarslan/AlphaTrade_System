from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal

from quant_trading_system.core.data_types import Direction, ModelPrediction
from quant_trading_system.models.runtime_governor import ModelRuntimeGovernor, RuntimeGovernorConfig


def _write_runtime_registry(
    tmp_path,
    *,
    canary_rollout: list[dict[str, float | int]] | None = None,
    max_slippage_bps: float = 25.0,
) -> tuple:
    models_root = tmp_path / "models"
    registry_root = models_root / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)

    registered_at = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
    canary_rollout = canary_rollout or [{"phase": 1, "capital_fraction": 1.0, "min_trades": 1}]
    registry_payload = {
        "xgboost": [
            {
                "model_name": "xgboost",
                "version_id": "v1",
                "registered_at": registered_at,
                "metrics": {"sharpe": 1.2},
                "deployment_plan": {
                    "ready_for_production": True,
                    "canary_rollout": [{"phase": 1, "capital_fraction": 1.0, "min_trades": 1}],
                },
            }
        ],
        "lightgbm": [
            {
                "model_name": "lightgbm",
                "version_id": "v2",
                "registered_at": registered_at,
                "metrics": {"sharpe": 1.5},
                "deployment_plan": {
                    "ready_for_production": True,
                    "canary_rollout": canary_rollout,
                    "kill_switch_guardrails": {"max_intraday_drawdown": 0.05},
                    "tca_guardrails": {"max_slippage_bps": max_slippage_bps},
                },
            }
        ],
    }
    with (registry_root / "registry.json").open("w", encoding="utf-8") as handle:
        json.dump(registry_payload, handle, indent=2)

    with (models_root / "active_model.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": "xgboost",
                "version_id": "v1",
                "updated_at": registered_at,
                "updated_by": "unit_test",
                "reason": "seed_pointer",
            },
            handle,
            indent=2,
        )

    return models_root, registry_root


def _sample_predictions() -> dict[str, ModelPrediction]:
    return {
        "champion": ModelPrediction(
            model_name="xgboost",
            symbol="AAPL",
            prediction=0.2,
            direction=Direction.LONG,
            confidence=0.7,
            horizon=5,
        ),
        "challenger": ModelPrediction(
            model_name="lightgbm",
            symbol="AAPL",
            prediction=0.4,
            direction=Direction.LONG,
            confidence=0.8,
            horizon=5,
        ),
    }


def test_route_predictions_selects_challenger_for_canary_symbol(tmp_path) -> None:
    models_root, _ = _write_runtime_registry(
        tmp_path,
        canary_rollout=[{"phase": 1, "capital_fraction": 1.0, "min_trades": 10}],
    )
    governor = ModelRuntimeGovernor(
        RuntimeGovernorConfig(models_root=models_root, refresh_interval_seconds=0)
    )

    routed = governor.route_predictions(_sample_predictions())

    assert {pred.model_name for pred in routed.values()} == {"lightgbm"}
    decision = governor.get_routing_decision("AAPL")
    assert decision is not None
    assert decision.role == "challenger"
    assert decision.selected_model == "lightgbm:v2"


def test_slippage_guardrail_quarantines_challenger_and_falls_back(tmp_path) -> None:
    models_root, _ = _write_runtime_registry(tmp_path, max_slippage_bps=5.0)
    governor = ModelRuntimeGovernor(
        RuntimeGovernorConfig(
            models_root=models_root,
            refresh_interval_seconds=0,
            min_orders_for_guardrail=1,
        )
    )
    governor.refresh_deployment_state(force=True)

    metadata = {
        "runtime_selected_model": "lightgbm:v2",
        "runtime_model_role": "challenger",
    }
    governor.record_order_submission(metadata)
    governor.record_order_fill(
        metadata,
        fill_price=Decimal("101"),
        expected_price=Decimal("100"),
    )

    stats = governor.get_statistics()
    assert stats["models"]["lightgbm:v2"]["quarantined"] is True

    routed = governor.route_predictions(_sample_predictions())
    assert {pred.model_name for pred in routed.values()} == {"xgboost"}


def test_final_canary_phase_promotes_challenger_when_enabled(tmp_path) -> None:
    models_root, registry_root = _write_runtime_registry(
        tmp_path,
        canary_rollout=[{"phase": 1, "capital_fraction": 1.0, "min_trades": 1}],
        max_slippage_bps=50.0,
    )
    governor = ModelRuntimeGovernor(
        RuntimeGovernorConfig(
            models_root=models_root,
            runtime_mode="live",
            allow_persistent_promotion=True,
            refresh_interval_seconds=0,
            min_orders_for_guardrail=1,
        )
    )
    governor.refresh_deployment_state(force=True)

    metadata = {
        "runtime_selected_model": "lightgbm:v2",
        "runtime_model_role": "challenger",
    }
    governor.record_order_submission(metadata)
    governor.record_order_fill(
        metadata,
        fill_price=Decimal("100"),
        expected_price=Decimal("100"),
    )

    with (models_root / "active_model.json").open("r", encoding="utf-8") as handle:
        pointer = json.load(handle)
    assert pointer["model_name"] == "lightgbm"
    assert pointer["version_id"] == "v2"

    with (registry_root / "registry.json").open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    assert registry["lightgbm"][0]["is_active"] is True
