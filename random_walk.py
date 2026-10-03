"""Learn two conditional expectations from complete, untruncated random walks.

The exact gambler's-ruin solution is used only for evaluation and plotting.
Training uses one exit indicator and one duration per sampled starting site.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

from experiments import LEARNING_RATE, SEEDS, STEPS, WIDTH, save_figure
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

LENGTH = 32
TRAJECTORIES = 4096
OUTPUT_SCALES = np.array([1., LENGTH**2])


def walk_labels(starts, seed, length=LENGTH):
    """One independent, complete symmetric walk per start; no time cutoff."""
    starts = np.asarray(starts, dtype=np.int64)
    if not np.all((starts > 0) & (starts < length)):
        raise ValueError("starting sites must be strictly between the boundaries")
    rng = np.random.default_rng(seed)
    positions = starts.copy()
    durations = np.zeros(len(starts), dtype=np.int64)
    active = np.arange(len(starts))
    while len(active):
        positions[active] += 2*rng.integers(0, 2, len(active)) - 1
        durations[active] += 1
        active = active[(positions[active] > 0) & (positions[active] < length)]
    return np.column_stack((positions == length, durations))


def fit(starts, targets, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(1, WIDTH), nn.Tanh(),
                          nn.Linear(WIDTH, WIDTH), nn.Tanh(), nn.Linear(WIDTH, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    inputs = torch.tensor(starts[:, None]/LENGTH, dtype=torch.float32)
    labels = torch.tensor(targets/OUTPUT_SCALES, dtype=torch.float32)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs)-labels).square().mean()
        loss.backward()
        optimizer.step()
    return model.eval()


def predict(model, starts):
    with torch.no_grad():
        inputs = torch.tensor(starts[:, None]/LENGTH, dtype=torch.float32)
        return model(inputs).numpy()*OUTPUT_SCALES


def model_weights(model, seed):
    linear = [layer for layer in model if isinstance(layer, nn.Linear)]
    return {"seed": seed, "layers": [
        {"weights": layer.weight.detach().tolist(), "biases": layer.bias.detach().tolist(),
         "activation": "linear" if index == len(linear)-1 else "tanh"}
        for index, layer in enumerate(linear)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"results")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "mc-random-walks"})
    beginning = time.perf_counter()
    grid = np.arange(1, LENGTH)
    exact = np.column_stack((grid/LENGTH, grid*(LENGTH-grid)))
    rows, curves, models = [], [], []
    for seed in SEEDS:
        starts = np.random.default_rng(4000+seed).integers(1, LENGTH, TRAJECTORIES)
        labels = walk_labels(starts, 5000+seed)
        model = fit(starts, labels, seed)
        prediction = predict(model, grid)
        relative = np.linalg.norm(prediction-exact, axis=0)/np.linalg.norm(exact, axis=0)
        rows.append({"seed": seed, "trajectories": len(starts),
                     "distinct_starting_sites": len(np.unique(starts)),
                     "transitions": int(labels[:, 1].sum()),
                     "probability_relative_l2": float(relative[0]),
                     "duration_relative_l2": float(relative[1])})
        curves.append(prediction)
        models.append(model_weights(model, seed))
        if seed == 0:
            np.savez(output/"walk_training.npz", starts=starts, labels=labels)
    np.savez(output/"walk_predictions.npz", grid=grid, exact=exact, predictions=np.array(curves))
    with (output/"walk_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    weights = {"length": LENGTH, "input_scale": LENGTH,
               "output_scales": OUTPUT_SCALES.tolist(), "models": models,
               "training": {"trajectories": TRAJECTORIES, "steps": STEPS,
                            "architecture": [1, WIDTH, WIDTH, 2], "learning_rate": LEARNING_RATE}}
    (output/"walk_models.json").write_text(json.dumps(weights, separators=(",", ":"))+"\n")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)
    training = np.load(output/"walk_training.npz")
    site_means = np.array([training["labels"][training["starts"] == x].mean(axis=0) for x in grid])
    for index, ax in enumerate(axes):
        ax.scatter(grid, site_means[:, index], color="#999999", s=19,
                   label="Mean labels at each site")
        ax.plot(grid, exact[:, index], color="#222222", lw=2, label="Exact expectation")
        ax.plot(grid, curves[0][:, index], color="#356aa0", lw=2, ls="--", label="Learned (seed 0)")
        ax.set(xlabel="Starting site x", xlim=(0, LENGTH),
               ylabel="Exit-right probability" if index == 0 else "Mean exit time (steps)")
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(frameon=False, fontsize=9)
    save_figure(fig, output, "random_walk")
    summary = {
        "settings": {"length": LENGTH, "trajectories_per_fit": TRAJECTORIES,
                     "starting_distribution": "uniform on integers 1,...,31",
                     "seeds": SEEDS, "steps": STEPS, "learning_rate": LEARNING_RATE,
                     "architecture": [1, WIDTH, WIDTH, 2], "input_scale": LENGTH,
                     "output_scales": OUTPUT_SCALES.tolist(), "cutoff": None},
        "seed_zero": rows[0],
        "relative_l2_mean": np.array([[row["probability_relative_l2"], row["duration_relative_l2"]]
                                      for row in rows]).mean(axis=0).tolist(),
        "relative_l2_sample_sd": np.array([[row["probability_relative_l2"], row["duration_relative_l2"]]
                                           for row in rows]).std(axis=0, ddof=1).tolist(),
        "transition_range": [min(row["transitions"] for row in rows), max(row["transitions"] for row in rows)],
        "runtime_seconds": time.perf_counter()-beginning,
    }
    (output/"walk_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
