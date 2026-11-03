"""Preserve historical M4 settings while rejecting a changed stopping contract."""

import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def finalizer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/turbompc"))
    return importlib.import_module("finalize_acceptance")


def test_historical_missing_false_is_compatible_without_modifying_evidence(finalizer):
    original = {"max_iter": 15, "tolerance": 1e-6, "linesearch": "merit"}
    current = {**original, "require_qp_convergence": False}
    before = deepcopy((original, current))
    finalizer.require_matching_sqp_configs(original, current)
    finalizer.require_matching_sqp_configs(current, current)
    finalizer.require_matching_sqp_configs(original, original)
    assert (original, current) == before


@pytest.mark.parametrize("side", ["original", "current", "both"])
def test_strict_qp_stopping_cannot_be_accepted_as_original_m4_settings(finalizer, side):
    original = {"max_iter": 15, "require_qp_convergence": False}
    current = original.copy()
    if side in ("original", "both"):
        original["require_qp_convergence"] = True
    if side in ("current", "both"):
        current["require_qp_convergence"] = True
    with pytest.raises(ValueError, match="must preserve require_qp_convergence=False"):
        finalizer.require_matching_sqp_configs(original, current)


@pytest.mark.parametrize("change", [{"max_iter": 16}, {"tolerance": 2e-6}, {"unexpected": False}])
def test_other_setting_changes_remain_exact_mismatches(finalizer, change):
    original = {"max_iter": 15, "tolerance": 1e-6}
    current = {**original, "require_qp_convergence": False, **change}
    with pytest.raises(ValueError, match="effective SQP settings differ"):
        finalizer.require_matching_sqp_configs(original, current)
