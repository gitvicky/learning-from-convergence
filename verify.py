"""Check numerical claims and the budget ledger, independently of training."""
import csv
import json
from pathlib import Path
import numpy as np

root = Path(__file__).parent / "results"
summary = json.loads((root / "summary.json").read_text())
rows = list(csv.DictReader((root / "metrics.csv").open()))
assert len(rows) == 24
assert len({(r["M"], r["N"], r["seed"]) for r in rows}) == 24
for row in rows:
    assert int(row["M"]) * int(row["N"]) * int(row["K"]) == 4096 == int(row["budget"])
curves = np.load(root / "predictions.npz")
assert curves["predictions"].shape == (24, 1001)
assert np.isfinite(curves["predictions"]).all()
np.testing.assert_allclose(curves["exact"], curves["grid"]**2 + 1)
for row, prediction in zip(rows, curves["predictions"]):
    np.testing.assert_allclose(float(row["test_mse"]), np.mean((prediction-curves["exact"])**2))
for item in summary["allocation_summary"]:
    errors = [float(r["test_mse"]) for r in rows if int(r["M"]) == item["M"]]
    np.testing.assert_allclose(item["mean_test_mse"], np.mean(errors))
    np.testing.assert_allclose(item["sample_sd_test_mse"], np.std(errors, ddof=1))
j = summary["jensen"]
assert abs(j["empirical_mean_log_estimator"] - j["exact_expected_log_estimator"]) < 5 * j["standard_error_mean_log"]
assert abs(j["empirical_log_mean_estimator"]) < 0.02

# Independently check the conditional MC mean and variance at three inputs.
# These diagnostic draws are separate from the budget used to train each model.
rng = np.random.default_rng(9876)
for a in (-3, 0, 3):
    samples = rng.normal(a, 1, 200_000)**2
    exact_variance = 2 + 4*a*a
    standard_error = (exact_variance / len(samples))**0.5
    assert abs(samples.mean() - (a*a + 1)) < 5 * standard_error
    assert abs(samples.var(ddof=1)/exact_variance - 1) < 0.03
for name in ("noisy_expectation", "budget_allocation"):
    for ext in ("svg", "png"):
        assert (root / f"{name}.{ext}").stat().st_size > 1000
print("Verified: 24 budgets, exact-reference errors, seed summaries, MC moments, Jensen gap, and figures.")
