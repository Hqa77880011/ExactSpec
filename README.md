# ExactSpec

Implementation of **ExactSpec: Semantics-Preserving Speculative Microbatch Reassignment for Straggler-Resilient Synchronous Training**. Each optimizer batch becomes immutable microbatch tasks. Idle processes can execute a slow worker's unfinished task, while a versioned commit token admits one gradient contribution.

## Installation

Use Python 3.10 or newer. Install matching PyTorch and torchvision builds for your device, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. For NVIDIA GPUs, choose a CUDA build from the [PyTorch installer](https://pytorch.org/get-started/locally/) in place of the CPU command. The supported dependency range is PyTorch 2.4–2.7 with its matching torchvision release.

## Data and workflow

`configs/small.json` uses a seeded generated classification workload for inspecting the execution protocol with little compute. Labels come from a fixed linear decision rule; this dataset is a development workload, separate from the paper's vision and language benchmarks. `configs/cifar10.json` downloads the official [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) splits through torchvision and uses a CIFAR-style ResNet-18. `prepare` writes training and test arrays under the configured data directory. CIFAR test labels are used only by `evaluate`.

```bash
python -m exactspec prepare --config configs/small.json
python -m exactspec train --config configs/small.json --output outputs/exactspec
python -m exactspec evaluate --config configs/small.json --checkpoint outputs/exactspec/model.pt --output outputs/exactspec
python -m exactspec plot --config configs/small.json --output outputs/exactspec
python -m exactspec compare --config configs/small.json --output outputs/comparison
python -m exactspec plot --config configs/small.json --output outputs/comparison
```

The runtime starts persistent independent Python workers with the spawn start method, including on Windows. Worker startup is excluded from iteration timing. Each iteration distributes the current parameter snapshot, executes tasks and applies one central optimizer update after every task has committed. CPU execution is the default for the small configuration. `devices` can list CUDA devices, one entry per worker or a shorter repeated list. The CIFAR configuration expects two available CUDA devices; adapt `devices` for your machine.

## Protocol and ablations

| Variant | Policy |
| --- | --- |
| `sync` | Fixed primary queues without speculation |
| `exactspec` | Predicted late-worker frontier with budget and cooldown |
| `no_prediction` | Allow speculation whenever a suitable idle worker exists |
| `no_cooldown` | Remove the spacing between speculative leases |
| `no_budget` | Remove the duplicate-task reservation cap |
| `backward_first` | Compute backward before requesting the commit token |

Task identity is optimizer step, global offset and sample count. Its seed depends on that identity, not the executor. Every copy uses the same isolated Torch, NumPy and Python random states. The default token is requested after forward and before backward. The coordinator serializes grant decisions and checks owner/version at commit. Leases expire only before grant; a token owner must remain alive until commit. Process failure after grant is outside the fail-slow model.

## Parameters and results

Commands read the JSON configuration. `--data`, `--output`, `--device`, `--variant`, `--steps`, `--batch-size`, `--seed`, `--workers`, `--limit` and `--checkpoint` override run settings. Worker devices are selected by the `devices` array; `--device` selects the evaluation device. Other fields use `--set KEY=JSON`, for example `--set rho=0.1`. String values need JSON quotes. Use separate output directories for policies.

`microbatch_size` determines task granularity. `rho` reserves a maximum fraction of microbatch copies; `ema`, `delta_s` and `min_remaining` control the remaining-time trigger. `cooldown_s` spaces speculative launches. The local copy reservation is a task-count bound, not a hard bound on variable-duration device compute; the measured redundant-compute ratio is reported separately. The shared model's gradients are summed in canonical task order and divided by the global sample count, regardless of tasks accepted by each worker.

`straggler_worker`, `straggler_every` and `straggler_delay_s` inject a delay before selected tasks. Set `straggler_delay_s=0` for a no-delay run. These injected traces are declared in `config.json`; they are not production traces. `lease_s`, `timeout_s` and `startup_timeout_s` bound protocol waits. `gradient_diagnostic=true` compares the accepted aggregate with a serial same-seed microbatch calculation on CPU, at additional cost.

`training.csv` contains measured iteration time, loss, throughput, speculative-copy count and redundant compute. `tasks.csv` records the accepted/discarded executions with sample offsets, seed, version and worker. `metrics.json` summarizes mean/P99 iteration time and compute overhead. `model.pt` stores the central optimizer and model; `evaluation.json` contains test accuracy. `iteration_tails.pdf` and `tail_overhead.pdf` visualize measurements from their corresponding CSV files.

## Implementation conventions

The supplied worker model has no mutable forward buffers: CIFAR ResNet uses GroupNorm, and the small MLP uses task-seeded dropout. This makes losing forward executions side-effect free. General BatchNorm shadow-buffer publication is not implemented. The central gradient transfer uses Python process pipes and canonical aggregation rather than a multi-node DDP SUM collective. It preserves the accepted-gradient semantics for this execution model, while IPC overhead and local device sharing affect its performance. The supplied suite covers CIFAR and the development workload, and does not preload the paper's language-model results.

Data, weights and generated outputs are ignored by Git. Run directories store the effective `config.json`, CSV measurements and JSON metrics. Plots read the generated measurement files.

## Registered compact reproduction setting

Two CPU spawn workers, a 64-wide MLP with task-seeded dropout, batch 32 split into microbatches of four, and declared fail-slow delay. No BatchNorm state or post-grant process crash is part of this setting.

Run from the repository root after installation. The following commands use the registered setting; they are provided for execution and have not been run as part of the source review.

```bash
python -m exactspec prepare --config configs/reproduce.json
python -m exactspec compare --config configs/reproduce.json --output outputs/reproduce_comparison
```

This setting needs no external dataset or pretrained weights. Generated inputs and declared fault/shift laws define the finite task; reported quantities retain that provenance.
