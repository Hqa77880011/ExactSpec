# ExactSpec

Implementation of **ExactSpec: Semantics-Preserving Speculative Microbatch Reassignment for Straggler-Resilient Synchronous Training**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

`configs/reproduce.json` generates a fixed-rule classification workload with two CPU spawn workers and declared fail-slow delays. `configs/cifar10.json` downloads [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html); adjust its `devices` list to the available accelerators before using it.

```bash
python -m exactspec prepare --config configs/reproduce.json
```

## Training and execution

`configs/reproduce.json` fixes seed `17`. Use `--seed` for a new run and keep its data/config paired. Paths are configurable with the CLI or JSON.

```bash
python -m exactspec train --config configs/reproduce.json --output outputs/exactspec
```

## Evaluation

```bash
python -m exactspec evaluate --config configs/reproduce.json --checkpoint outputs/exactspec/model.pt --output outputs/exactspec
```

## Comparisons and ablations

`compare` runs the configured variants; it includes model fitting where required. Keep comparison outputs separate from the main run.

```bash
python -m exactspec compare --config configs/reproduce.json --output outputs/comparison
python -m exactspec plot --config configs/reproduce.json --output outputs/comparison
```

## Outputs

Run directories contain `model.pt`, `training.csv`, `tasks.csv`, `metrics.json`, `evaluation.json` as produced by the corresponding steps. `config.json` records effective settings. Comparison results are in `outputs/comparison/comparison.csv`; plotting reads those computed files.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
