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

# Check the complete-trajectory example and exported model weights independently.
from random_walk import LENGTH, walk_labels

walk_summary = json.loads((root/"walk_summary.json").read_text())
walk_rows = list(csv.DictReader((root/"walk_metrics.csv").open()))
walk_curves = np.load(root/"walk_predictions.npz")
assert len(walk_rows) == 8 and {int(row["seed"]) for row in walk_rows} == set(range(8))
grid, exact = walk_curves["grid"], walk_curves["exact"]
assert walk_curves["predictions"].shape == (8, 31, 2)
np.testing.assert_allclose(exact[:,0], grid/LENGTH)
np.testing.assert_allclose(exact[:,1], grid*(LENGTH-grid))
# Independently solve the finite-state harmonic and Poisson equations.
operator = np.eye(LENGTH-1)
operator += np.diag(np.full(LENGTH-2, -.5), 1) + np.diag(np.full(LENGTH-2, -.5), -1)
boundary = np.zeros(LENGTH-1); boundary[-1] = .5
np.testing.assert_allclose(np.linalg.solve(operator, boundary), exact[:,0])
np.testing.assert_allclose(np.linalg.solve(operator, np.ones(LENGTH-1)), exact[:,1])
errors = []
for row, curve in zip(walk_rows, walk_curves["predictions"]):
    assert int(row["trajectories"]) == 4096 and int(row["distinct_starting_sites"]) == 31
    relative = np.linalg.norm(curve-exact, axis=0)/np.linalg.norm(exact, axis=0)
    np.testing.assert_allclose(relative, [float(row["probability_relative_l2"]), float(row["duration_relative_l2"])])
    errors.append(relative)
np.testing.assert_allclose(walk_summary["relative_l2_mean"], np.mean(errors, axis=0))
np.testing.assert_allclose(walk_summary["relative_l2_sample_sd"], np.std(errors, axis=0, ddof=1))
training = np.load(root/"walk_training.npz")
assert int(training["labels"][:,1].sum()) == int(walk_rows[0]["transitions"]) == 716131
assert np.isin(training["labels"][:,0], [0,1]).all()
assert (training["labels"][:,1] >= np.minimum(training["starts"], LENGTH-training["starts"])).all()
assert (training["labels"][:,1] % 2 == training["starts"] % 2).all()
for x in (1, 16, 31):
    draws = walk_labels(np.full(20_000, x), 7000+x)
    reference = np.array([x/LENGTH, x*(LENGTH-x)])
    standard_errors = draws.std(axis=0, ddof=1)/len(draws)**.5
    assert (np.abs(draws.mean(axis=0)-reference) < 6*standard_errors).all()
payload = json.loads((root/"walk_models.json").read_text())
for model, reference in zip(payload["models"], walk_curves["predictions"]):
    values = grid[:,None]/payload["input_scale"]
    for layer in model["layers"]:
        values = values @ np.asarray(layer["weights"]).T + layer["biases"]
        if layer["activation"] == "tanh":
            values = np.tanh(values)
    np.testing.assert_allclose(values*payload["output_scales"], reference, atol=4e-4)
assert walk_summary["settings"]["cutoff"] is None
assert "__WALK_MODELS__" not in (root/"_walk_demo.qmd").read_text()
for name in ("random_walk.svg", "random_walk.png", "walk_demo.html", "walk_demo.js", "walk_demo.css"):
    assert (root/name).stat().st_size > 1000
print("Verified: all eight walk fits, transition ledger, exact equations, MC means, and exported model predictions.")
