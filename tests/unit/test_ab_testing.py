"""Unit tests for A/B testing utilities."""

import numpy as np

from quant_trading_system.models.ab_testing import (
    FrequentistTester,
    SequentialTester,
    TestResult,
    TestType,
    VariantMetrics,
)


def _make_variant_metrics(name: str, mean: float) -> VariantMetrics:
    sample_size = 50
    std = 1.0
    return VariantMetrics(
        variant_name=name,
        sample_size=sample_size,
        mean=mean,
        std=std,
        min_value=mean - 2.0,
        max_value=mean + 2.0,
        sum_value=mean * sample_size,
        sum_squared=sample_size * (std ** 2 + mean ** 2),
        successes=25,
        failures=25,
    )


def test_sequential_tester_advances_analysis_and_spends_alpha(monkeypatch):
    """Sequential tester should progress interim looks and spend non-zero alpha."""
    captured_alphas: list[float] = []

    def _capture_alpha(self, control, treatment, alpha=0.05):
        captured_alphas.append(alpha)
        return TestResult(
            is_significant=False,
            p_value=0.5,
            confidence_interval=(-0.1, 0.1),
            effect_size=0.0,
            relative_effect=0.0,
            power_achieved=0.0,
            test_type=TestType.FREQUENTIST,
            sample_size_control=control.sample_size,
            sample_size_treatment=treatment.sample_size,
        )

    monkeypatch.setattr(FrequentistTester, "test", _capture_alpha)

    control = _make_variant_metrics("control", 0.1)
    treatment = _make_variant_metrics("treatment", 0.12)
    tester = SequentialTester(n_analyses=4)

    tester.test(control, treatment, alpha=0.05)
    tester.test(control, treatment, alpha=0.05)

    assert tester._current_analysis == 2
    assert len(captured_alphas) == 2
    assert all(np.isfinite(alpha) for alpha in captured_alphas)
    assert captured_alphas[0] > 0.0
    assert captured_alphas[1] > captured_alphas[0]
    assert captured_alphas[1] <= 0.05
