"""Reproduce the two figures and the Jensen-gap check on CPU.

No clean target is used for training, stopping, or hyperparameter selection.
The analytic expectation is used only for evaluation and plotting.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

BUDGET = 4096
ALLOCATIONS = ((4096, 1), (512, 8), (64, 64))
SEEDS = tuple(range(8))
STEPS = 800
LEARNING_RATE = 0.01
WIDTH = 32


def labels(m: int, n: int, seed: int):
    """Exactly m*n normal draws; configurations are uniform on [-3, 3]."""
    configurations = np.random.default_rng(1000 + seed).uniform(-3, 3, m)
    noise = np.random.default_rng(2000 + seed).standard_normal((m, n))
    target = np.square(configurations[:, None] + noise).mean(axis=1)
    return configurations, target


def fit(configurations, target, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(1, WIDTH), nn.Tanh(),
                          nn.Linear(WIDTH, WIDTH), nn.Tanh(), nn.Linear(WIDTH, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Fixed, linear rescaling preserves the squared-loss optimum.
    inputs = torch.tensor(configurations[:, None] / 3, dtype=torch.float32)
    targets = torch.tensor(target[:, None] / 10, dtype=torch.float32)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - targets).square().mean()
        loss.backward()
        optimizer.step()
    return model.eval()


def predict(model, configurations):
    with torch.no_grad():
        inputs = torch.tensor(configurations[:, None] / 3, dtype=torch.float32)
        return (10 * model(inputs)).numpy().ravel()


def save_figure(fig, output, name):
    for extension in ("svg", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=240, bbox_inches="tight",
                    metadata={"Date": None} if extension == "svg" else None)
    plt.close(fig)


def jensen_check():
    # a=0, N=1: F_hat=Z^2, E[log(Z^2)]=-EulerGamma-log(2).
    estimates = np.random.default_rng(3000).standard_normal(200_000) ** 2
    log_estimates = np.log(estimates)  # No clipping or epsilon added.
    exact = -0.5772156649015329 - math.log(2)
    return {
        "a": 0, "N": 1, "replicates": len(estimates),
        "exact_log_expectation": 0.0,
        "exact_expected_log_estimator": exact,
        "empirical_log_mean_estimator": float(np.log(estimates.mean())),
        "empirical_mean_log_estimator": float(log_estimates.mean()),
        "standard_error_mean_log": float(log_estimates.std(ddof=1) / len(estimates)**0.5),
        "exact_variance_log_estimator": math.pi**2 / 2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": "#222222",
        "text.color": "#222222", "svg.fonttype": "none", "svg.hashsalt": "mc-expectations",
    })
    start = time.perf_counter()
    grid = np.linspace(-3, 3, 1001)
    exact = grid**2 + 1
    rows = []
    curves = []
    figure_one_data = None
    for m, n in ALLOCATIONS:
        assert m * n == BUDGET
        for seed in SEEDS:
            configurations, targets = labels(m, n, seed)
            model = fit(configurations, targets, seed)
            prediction = predict(model, grid)
            mse = float(np.mean((prediction - exact)**2))
            rows.append({"seed": seed, "M": m, "N": n, "K": 1,
                         "budget": m*n, "test_mse": mse,
                         "relative_l2": float(np.linalg.norm(prediction-exact)/np.linalg.norm(exact))})
            curves.append(prediction)
            if m == BUDGET and seed == 0:
                figure_one_data = (configurations, targets, prediction)
    np.savez(output / "predictions.npz", grid=grid, exact=exact, predictions=np.array(curves))
    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    configurations, targets, prediction = figure_one_data
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.scatter(configurations, targets, s=7, alpha=0.11, color="#777777", rasterized=True,
               label=r"Noisy labels: $N=1$")
    ax.plot(grid, exact, color="#171717", lw=2.2, label=r"Exact $F(a)=a^2+1$")
    ax.plot(grid, prediction, color="#356aa0", lw=2, ls="--", label="Learned expectation")
    # Show every label: no selection or cropping of high values.
    ax.set(xlabel=r"Configuration $a$", ylabel=r"$X^2$ or expectation", xlim=(-3.1, 3.1))
    ax.legend(frameon=False, loc="upper center")
    ax.grid(axis="y", alpha=0.15)
    save_figure(fig, output, "noisy_expectation")

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    summary = []
    for index, (m, n) in enumerate(ALLOCATIONS):
        errors = np.array([r["test_mse"] for r in rows if r["M"] == m])
        ax.scatter(np.full(len(errors), index) + np.linspace(-0.13, 0.13, len(errors)),
                   errors, s=32, color="#888888", alpha=0.7,
                   label="Individual seeds" if index == 0 else None)
        ax.scatter(index, errors.mean(), marker="D", color="#356aa0", s=64,
                   label="Mean across seeds" if index == 0 else None)
        summary.append({"M": m, "N": n, "mean_test_mse": float(errors.mean()),
                        "sample_sd_test_mse": float(errors.std(ddof=1))})
    ax.set_xticks(range(3), [f"M={m:,}\nN={n}" for m,n in ALLOCATIONS])
    ax.set(ylabel="Test mean-squared error", xlabel="Allocation of 4,096 simulation draws (K=1)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    save_figure(fig, output, "budget_allocation")
    table = ["| Configurations $M$ | Samples $N$ | Test MSE (mean ± SD) |",
             "|---:|---:|---:|"]
    for item in summary:
        table.append(f"| {item['M']:,} | {item['N']} | "
                     f"{item['mean_test_mse']:.4f} ± {item['sample_sd_test_mse']:.4f} |")
    (output / "allocation_table.md").write_text("\n".join(table) + "\n")
    report = {
        "settings": {"budget_per_fit": BUDGET, "allocations": ALLOCATIONS,
                     "seeds": SEEDS, "steps": STEPS, "learning_rate": LEARNING_RATE,
                     "architecture": [1, WIDTH, WIDTH, 1], "activation": "tanh",
                     "optimizer": "Adam", "batch": "full", "device": "cpu",
                     "grid_points": len(grid), "interval": [-3, 3],
                     "input_scale": 3, "output_scale": 10},
        "allocation_summary": summary, "jensen": jensen_check(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__, "matplotlib": matplotlib.__version__},
        "runtime_seconds": time.perf_counter() - start,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
