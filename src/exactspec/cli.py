import torch
from pathlib import Path
import json
import numpy as np
from torch.nn import functional as F

from .io import configure, seed, device, csv_write, json_write, plot
from .data import prepare, tensors, network
from .runtime import Runtime, stochastic_path, task_seed

VARIANTS = [
    "exactspec",
    "sync",
    "no_prediction",
    "no_cooldown",
    "no_budget",
    "backward_first",
]


def reference_gradient(model, x, y, step, config):
    values = [torch.zeros_like(parameter) for parameter in model.parameters()]
    model.train()
    for offset in range(0, len(x), config["microbatch_size"]):
        data = x[offset : offset + config["microbatch_size"]]
        labels = y[offset : offset + config["microbatch_size"]]
        model.zero_grad(set_to_none=True)
        with stochastic_path(
            task_seed(config["seed"], step, offset, len(data)), torch.device("cpu")
        ):
            loss = F.cross_entropy(model(data), labels, reduction="sum")
            loss.backward()
        for result, parameter in zip(values, model.parameters()):
            result.add_(parameter.grad)
    return [value / len(x) for value in values]


def train(config):
    if config["variant"] not in VARIANTS:
        raise ValueError(f"Unknown variant: {config['variant']}")
    if (
        config["workers"] < 1
        or config["microbatch_size"] < 1
        or not 0 <= config["rho"] <= 1
    ):
        raise ValueError(
            "workers and microbatch_size must be positive and rho must be in [0, 1]"
        )
    if config["gradient_diagnostic"] and any(
        torch.device(value).type != "cpu" for value in config["devices"]
    ):
        raise ValueError(
            "gradient_diagnostic requires CPU workers so the reference uses the same stochastic kernels"
        )
    seed(config["seed"])
    x, y = tensors(config)
    model = network(config, stateless=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    runtime = Runtime(config)
    history, tasks = [], []
    rng = np.random.default_rng(config["seed"])
    try:
        for step in range(1, config["steps"] + 1):
            indices = rng.choice(
                len(x), config["batch_size"], replace=len(x) < config["batch_size"]
            )
            batch, labels = x[indices], y[indices]
            gradients, row, records = runtime.iteration(
                model.state_dict(), batch, labels, step
            )
            if config["gradient_diagnostic"]:
                reference = reference_gradient(model, batch, labels, step, config)
                error = max(
                    (a - b).abs().max().item() for a, b in zip(gradients, reference)
                )
                row["gradient_max_error"] = error
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(model.parameters(), gradients):
                parameter.grad = gradient
            optimizer.step()
            row.update(
                {
                    "variant": config["variant"],
                    "step": step,
                    "samples_per_s": len(batch) / row["iteration_s"],
                }
            )
            history.append(row)
            tasks.extend([dict(record, step=step) for record in records])
            print(json.dumps(row), flush=True)
    finally:
        runtime.close()
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        output / "model.pt",
    )
    csv_write(output / "training.csv", history)
    csv_write(output / "tasks.csv", tasks)
    json_write(output / "config.json", config)
    summary = {
        "variant": config["variant"],
        "measurement": "process_runtime_with_injected_delay",
        "mean_iteration_s": float(np.mean([row["iteration_s"] for row in history])),
        "p99_iteration_s": float(
            np.quantile([row["iteration_s"] for row in history], 0.99)
        ),
        "mean_redundant_compute": float(
            np.mean([row["redundant_compute"] for row in history])
        ),
        "final_loss": history[-1]["loss"],
    }
    json_write(output / "metrics.json", summary)
    return summary


@torch.no_grad()
def evaluate(config):
    state = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    model = network(state["config"], stateless=True).to(device(config)).eval()
    model.load_state_dict(state["model"])
    x, y = tensors(config, "test")
    predictions = []
    for offset in range(0, len(x), config["batch_size"]):
        predictions.append(
            model(x[offset : offset + config["batch_size"]].to(device(config)))
            .argmax(1)
            .cpu()
        )
    json_write(
        Path(config["output"]) / "evaluation.json",
        {"accuracy": 100 * (torch.cat(predictions) == y).float().mean().item()},
    )


def compare(config):
    rows = []
    for variant in config["variants"]:
        current = dict(
            config, variant=variant, output=str(Path(config["output"]) / variant)
        )
        rows.append(train(current))
    csv_write(Path(config["output"]) / "comparison.csv", rows)


def main():
    command, config = configure(["prepare", "train", "evaluate", "compare", "plot"])
    if command == "prepare":
        prepare(config)
    elif command == "train":
        train(config)
    elif command == "evaluate":
        evaluate(config)
    elif command == "compare":
        compare(config)
    else:
        root = Path(config["output"])
        if (root / "training.csv").exists():
            plot(
                root / "training.csv",
                root / "iteration_tails.pdf",
                "step",
                "iteration_s",
            )
        else:
            plot(
                root / "comparison.csv",
                root / "tail_overhead.pdf",
                "mean_redundant_compute",
                "p99_iteration_s",
            )
