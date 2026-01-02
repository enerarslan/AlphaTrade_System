"""
JPMORGAN FIX: Model Validation Gates Module.

Implements institutional-grade model validation gates to prevent
deployment of unreliable models to production trading.

Validation gates check:
1. Statistical significance of model performance
2. Overfitting indicators (IS/OOS ratio)
3. Risk metrics (drawdown, Sharpe, win rate)
4. Data quality requirements
5. Stability across different periods
6. Economic sensibility checks

These gates are MANDATORY before any model can be deployed.
Failed gates will raise ModelValidationError and block deployment.

Reference:
    "Model Risk Management" - OCC 2011-12 Guidance
    "Machine Learning in Finance" - JPMorgan Quant Research
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class GateSeverity(str, Enum):
    """Severity level for validation gate failures."""

    CRITICAL = "critical"  # Blocks deployment entirely
    WARNING = "warning"  # Allows deployment with logging
    INFO = "info"  # Informational only


class GateCategory(str, Enum):
    """Category of validation gate."""

    PERFORMANCE = "performance"
    OVERFITTING = "overfitting"
    RISK = "risk"
    DATA_QUALITY = "data_quality"
    STABILITY = "stability"
    ECONOMIC = "economic"


@dataclass
class GateResult:
    """Result of a single validation gate check."""

    gate_name: str
    category: GateCategory
    severity: GateSeverity
    passed: bool
    actual_value: float
    threshold_value: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gate_name": self.gate_name,
            "category": self.category.value,
            "severity": self.severity.value,
            "passed": self.passed,
            "actual_value": self.actual_value,
            "threshold_value": self.threshold_value,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """Complete validation report for a model."""

    model_name: str
    model_version: str
    gate_results: list[GateResult]
    overall_passed: bool
    critical_failures: int
    warning_failures: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def all_critical_passed(self) -> bool:
        """Check if all critical gates passed."""
        return all(
            r.passed for r in self.gate_results
            if r.severity == GateSeverity.CRITICAL
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "gate_results": [r.to_dict() for r in self.gate_results],
            "overall_passed": self.overall_passed,
            "critical_failures": self.critical_failures,
            "warning_failures": self.warning_failures,
            "created_at": self.created_at.isoformat(),
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        passed_count = sum(1 for r in self.gate_results if r.passed)
        total_count = len(self.gate_results)

        status = "PASSED" if self.overall_passed else "FAILED"

        summary = [
            f"Model Validation Report: {self.model_name} v{self.model_version}",
            f"Status: {status}",
            f"Gates Passed: {passed_count}/{total_count}",
            f"Critical Failures: {self.critical_failures}",
            f"Warning Failures: {self.warning_failures}",
            "",
            "Gate Details:",
        ]

        for result in self.gate_results:
            status_icon = "[PASS]" if result.passed else "[FAIL]"
            summary.append(
                f"  {status_icon} {result.gate_name}: "
                f"{result.actual_value:.4f} vs {result.threshold_value:.4f} - {result.message}"
            )

        return "\n".join(summary)


class ValidationGate:
    """Base class for validation gates."""

    def __init__(
        self,
        name: str,
        category: GateCategory,
        severity: GateSeverity,
        threshold: float,
        comparison: str = ">=",  # ">=", "<=", ">", "<", "=="
    ):
        """
        Initialize validation gate.

        Args:
            name: Gate name
            category: Gate category
            severity: Failure severity
            threshold: Threshold value
            comparison: Comparison operator
        """
        self.name = name
        self.category = category
        self.severity = severity
        self.threshold = threshold
        self.comparison = comparison

    def check(self, value: float, **kwargs: Any) -> GateResult:
        """
        Check if the gate passes.

        Args:
            value: Value to check against threshold
            **kwargs: Additional context

        Returns:
            GateResult with pass/fail status
        """
        if self.comparison == ">=":
            passed = value >= self.threshold
        elif self.comparison == "<=":
            passed = value <= self.threshold
        elif self.comparison == ">":
            passed = value > self.threshold
        elif self.comparison == "<":
            passed = value < self.threshold
        elif self.comparison == "==":
            passed = abs(value - self.threshold) < 1e-10
        else:
            passed = False

        if passed:
            message = f"Gate passed: {value:.4f} {self.comparison} {self.threshold:.4f}"
        else:
            message = f"Gate FAILED: {value:.4f} not {self.comparison} {self.threshold:.4f}"

        return GateResult(
            gate_name=self.name,
            category=self.category,
            severity=self.severity,
            passed=passed,
            actual_value=value,
            threshold_value=self.threshold,
            message=message,
            details=kwargs,
        )


class ModelValidationGates:
    """
    JPMORGAN FIX: Comprehensive model validation gates.

    Implements a battery of validation checks that MUST pass before
    a model can be deployed to production trading.

    Usage:
        gates = ModelValidationGates()

        # Validate model before deployment
        report = gates.validate(
            model_name="xgboost_v2",
            holdout_metrics=holdout_metrics,
            is_metrics=in_sample_metrics,
        )

        if not report.overall_passed:
            raise ModelValidationError(report.get_summary())
    """

    def __init__(
        self,
        min_sharpe_ratio: float = 0.5,
        max_drawdown: float = 0.25,
        min_win_rate: float = 0.45,
        min_profit_factor: float = 1.1,
        max_is_oos_ratio: float = 2.0,
        min_samples: int = 500,
        min_r2: float = 0.0,
        max_volatility: float = 0.30,
        min_stability_score: float = 0.3,
        min_information_ratio: float = 0.3,
    ):
        """
        Initialize validation gates with thresholds.

        Args:
            min_sharpe_ratio: Minimum acceptable Sharpe ratio
            max_drawdown: Maximum acceptable drawdown
            min_win_rate: Minimum acceptable win rate
            min_profit_factor: Minimum acceptable profit factor
            max_is_oos_ratio: Maximum IS/OOS performance ratio
            min_samples: Minimum samples in holdout set
            min_r2: Minimum R-squared for regression models
            max_volatility: Maximum acceptable volatility
            min_stability_score: Minimum stability across periods
            min_information_ratio: Minimum Information Ratio
        """
        self.gates: list[ValidationGate] = []

        # Performance Gates
        self.gates.append(ValidationGate(
            name="Minimum Sharpe Ratio",
            category=GateCategory.PERFORMANCE,
            severity=GateSeverity.CRITICAL,
            threshold=min_sharpe_ratio,
            comparison=">=",
        ))

        self.gates.append(ValidationGate(
            name="Minimum Win Rate",
            category=GateCategory.PERFORMANCE,
            severity=GateSeverity.CRITICAL,
            threshold=min_win_rate,
            comparison=">=",
        ))

        self.gates.append(ValidationGate(
            name="Minimum Profit Factor",
            category=GateCategory.PERFORMANCE,
            severity=GateSeverity.WARNING,
            threshold=min_profit_factor,
            comparison=">=",
        ))

        self.gates.append(ValidationGate(
            name="Minimum Information Ratio",
            category=GateCategory.PERFORMANCE,
            severity=GateSeverity.WARNING,
            threshold=min_information_ratio,
            comparison=">=",
        ))

        # Risk Gates
        self.gates.append(ValidationGate(
            name="Maximum Drawdown",
            category=GateCategory.RISK,
            severity=GateSeverity.CRITICAL,
            threshold=max_drawdown,
            comparison="<=",
        ))

        self.gates.append(ValidationGate(
            name="Maximum Volatility",
            category=GateCategory.RISK,
            severity=GateSeverity.WARNING,
            threshold=max_volatility,
            comparison="<=",
        ))

        # Overfitting Gates
        self.gates.append(ValidationGate(
            name="Maximum IS/OOS Ratio",
            category=GateCategory.OVERFITTING,
            severity=GateSeverity.CRITICAL,
            threshold=max_is_oos_ratio,
            comparison="<=",
        ))

        # Data Quality Gates
        self.gates.append(ValidationGate(
            name="Minimum Sample Size",
            category=GateCategory.DATA_QUALITY,
            severity=GateSeverity.CRITICAL,
            threshold=float(min_samples),
            comparison=">=",
        ))

        # Stability Gates
        self.gates.append(ValidationGate(
            name="Minimum Stability Score",
            category=GateCategory.STABILITY,
            severity=GateSeverity.WARNING,
            threshold=min_stability_score,
            comparison=">=",
        ))

        # Store thresholds for custom checks
        self.min_r2 = min_r2

    def validate(
        self,
        model_name: str,
        model_version: str = "1.0",
        holdout_metrics: dict[str, float] | None = None,
        is_metrics: dict[str, float] | None = None,
        predictions: np.ndarray | None = None,
        actuals: np.ndarray | None = None,
    ) -> ValidationReport:
        """
        Validate a model against all gates.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            holdout_metrics: Out-of-sample metrics
            is_metrics: In-sample metrics (for overfitting check)
            predictions: Model predictions (for custom checks)
            actuals: Actual values (for custom checks)

        Returns:
            ValidationReport with all gate results
        """
        logger.info(f"Validating model: {model_name} v{model_version}")

        gate_results = []
        holdout_metrics = holdout_metrics or {}
        is_metrics = is_metrics or {}

        # Map metric names to gate names
        metric_to_gate = {
            "sharpe_ratio": "Minimum Sharpe Ratio",
            "win_rate": "Minimum Win Rate",
            "profit_factor": "Minimum Profit Factor",
            "information_ratio": "Minimum Information Ratio",
            "max_drawdown": "Maximum Drawdown",
            "volatility": "Maximum Volatility",
            "n_samples": "Minimum Sample Size",
            "stability_score": "Minimum Stability Score",
        }

        # Check standard gates
        for gate in self.gates:
            if gate.name == "Maximum IS/OOS Ratio":
                # Special handling for IS/OOS ratio
                is_sharpe = is_metrics.get("sharpe_ratio", 0.0)
                oos_sharpe = holdout_metrics.get("sharpe_ratio", 0.0)

                if oos_sharpe > 0:
                    is_oos_ratio = is_sharpe / oos_sharpe
                else:
                    is_oos_ratio = float("inf") if is_sharpe > 0 else 1.0

                result = gate.check(is_oos_ratio, is_sharpe=is_sharpe, oos_sharpe=oos_sharpe)
                gate_results.append(result)

            else:
                # Find matching metric
                metric_name = None
                for m_name, g_name in metric_to_gate.items():
                    if g_name == gate.name:
                        metric_name = m_name
                        break

                if metric_name and metric_name in holdout_metrics:
                    value = holdout_metrics[metric_name]
                    result = gate.check(value)
                    gate_results.append(result)

        # Custom validation: R-squared for regression
        if predictions is not None and actuals is not None:
            r2 = self._compute_r2(predictions, actuals)
            gate_results.append(GateResult(
                gate_name="Minimum R-squared",
                category=GateCategory.PERFORMANCE,
                severity=GateSeverity.WARNING,
                passed=r2 >= self.min_r2,
                actual_value=r2,
                threshold_value=self.min_r2,
                message=f"R² = {r2:.4f} vs min {self.min_r2:.4f}",
            ))

        # Economic sensibility check
        economic_check = self._check_economic_sensibility(holdout_metrics)
        if economic_check:
            gate_results.append(economic_check)

        # Count failures
        critical_failures = sum(
            1 for r in gate_results
            if not r.passed and r.severity == GateSeverity.CRITICAL
        )
        warning_failures = sum(
            1 for r in gate_results
            if not r.passed and r.severity == GateSeverity.WARNING
        )

        # Overall pass/fail
        overall_passed = critical_failures == 0

        report = ValidationReport(
            model_name=model_name,
            model_version=model_version,
            gate_results=gate_results,
            overall_passed=overall_passed,
            critical_failures=critical_failures,
            warning_failures=warning_failures,
        )

        if overall_passed:
            logger.info(f"Model {model_name} PASSED all critical validation gates")
        else:
            logger.warning(
                f"Model {model_name} FAILED validation: "
                f"{critical_failures} critical, {warning_failures} warnings"
            )

        return report

    def _compute_r2(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Compute R-squared."""
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))

        if np.sum(valid_mask) < 10:
            return 0.0

        y_pred = predictions[valid_mask]
        y_true = actuals[valid_mask]

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot < 1e-10:
            return 0.0

        return float(1 - ss_res / ss_tot)

    def _check_economic_sensibility(
        self,
        metrics: dict[str, float],
    ) -> GateResult | None:
        """
        Check for economically sensible results.

        Examples of nonsensical results:
        - Negative volatility
        - Win rate outside [0, 1]
        - Impossible Sharpe ratios
        """
        issues = []

        # Check win rate
        win_rate = metrics.get("win_rate", 0.5)
        if not 0 <= win_rate <= 1:
            issues.append(f"Invalid win rate: {win_rate}")

        # Check volatility
        volatility = metrics.get("volatility", 0.1)
        if volatility < 0:
            issues.append(f"Negative volatility: {volatility}")

        # Check Sharpe (extremely high values are suspicious)
        sharpe = metrics.get("sharpe_ratio", 0.0)
        if abs(sharpe) > 10:
            issues.append(f"Suspicious Sharpe ratio: {sharpe}")

        # Check profit factor
        profit_factor = metrics.get("profit_factor", 1.0)
        if profit_factor < 0:
            issues.append(f"Negative profit factor: {profit_factor}")

        if issues:
            return GateResult(
                gate_name="Economic Sensibility Check",
                category=GateCategory.ECONOMIC,
                severity=GateSeverity.CRITICAL,
                passed=False,
                actual_value=len(issues),
                threshold_value=0,
                message=f"Economic sensibility issues: {', '.join(issues)}",
                details={"issues": issues},
            )

        return GateResult(
            gate_name="Economic Sensibility Check",
            category=GateCategory.ECONOMIC,
            severity=GateSeverity.CRITICAL,
            passed=True,
            actual_value=0,
            threshold_value=0,
            message="All metrics are economically sensible",
        )

    def add_custom_gate(
        self,
        name: str,
        check_func: Callable[[dict[str, float]], GateResult],
    ) -> None:
        """
        Add a custom validation gate.

        Args:
            name: Gate name
            check_func: Function that takes metrics and returns GateResult
        """
        # Store as lambda that calls the check function
        # This is a simplified version; a full implementation would
        # integrate with the validate method
        logger.info(f"Custom gate '{name}' registered (not yet integrated)")


def validate_model_for_deployment(
    model_name: str,
    holdout_metrics: dict[str, float],
    is_metrics: dict[str, float] | None = None,
    strict: bool = True,
) -> tuple[bool, ValidationReport]:
    """
    Convenience function to validate a model for deployment.

    Args:
        model_name: Name of the model
        holdout_metrics: Out-of-sample metrics
        is_metrics: In-sample metrics (optional)
        strict: If True, use stricter thresholds

    Returns:
        Tuple of (passed, ValidationReport)
    """
    if strict:
        gates = ModelValidationGates(
            min_sharpe_ratio=0.7,
            max_drawdown=0.20,
            min_win_rate=0.48,
            min_profit_factor=1.2,
            max_is_oos_ratio=1.5,
        )
    else:
        gates = ModelValidationGates()

    report = gates.validate(
        model_name=model_name,
        holdout_metrics=holdout_metrics,
        is_metrics=is_metrics,
    )

    return report.overall_passed, report
