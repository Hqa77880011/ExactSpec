import torch
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import multiprocessing as mp
from multiprocessing.connection import wait
import random
import time
import numpy as np
from torch.nn import functional as F

from .data import network


def task_seed(base, step, offset, count):
    value = (
        base + step * 0x9E3779B1 + offset * 0x85EBCA77 + count * 0xC2B2AE3D
    ) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 27)) & 0x7FFFFFFFFFFFFFFF


@contextmanager
def stochastic_path(value, target):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = (
        [target.index if target.index is not None else torch.cuda.current_device()]
        if target.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(value)
        random.seed(value)
        np.random.seed(value % (2**32))
        if devices:
            torch.cuda.manual_seed(value)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


@dataclass
class Task:
    offset: int
    count: int
    primary: int
    seed: int
    version: int = 0
    state: str = "LEASED"
    owner: int | None = None
    leases: set = field(default_factory=set)
    speculated: bool = False
    expires: float = 0.0

    def grant(self, worker, version):
        if (
            version != self.version
            or worker not in self.leases
            or self.state != "LEASED"
        ):
            return False
        self.owner = worker
        self.state = "COMMIT_GRANTED"
        return True

    def commit(self, worker, version):
        if (
            worker != self.owner
            or version != self.version
            or self.state != "COMMIT_GRANTED"
        ):
            raise RuntimeError("Invalid task commit")
        self.state = "COMMITTED"


def worker_loop(index, pipe, config):
    torch.set_num_threads(config["threads"])
    target = torch.device(config["devices"][index % len(config["devices"])])
    model = network(config, stateless=True).to(target)
    model.train()
    pipe.send(("started", index))
    while True:
        message = pipe.recv()
        if message[0] == "stop":
            break
        if message[0] == "state":
            model.load_state_dict(message[1])
            pipe.send(("ready", index))
            continue
        _, identity, version, x, y, value, delay = message
        if delay:
            time.sleep(delay)
        model.zero_grad(set_to_none=True)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        started = time.perf_counter()
        gradients = None
        with stochastic_path(value, target):
            loss = F.cross_entropy(model(x.to(target)), y.to(target), reduction="sum")
            if config["variant"] == "backward_first":
                loss.backward()
                gradients = [p.grad.detach().cpu().clone() for p in model.parameters()]
            if target.type == "cuda":
                torch.cuda.synchronize(target)
            forward_s = time.perf_counter() - started
            pipe.send(("token", index, identity, version, forward_s))
            granted = pipe.recv()
            if granted:
                if gradients is None:
                    loss.backward()
                    gradients = [
                        p.grad.detach().cpu().clone() for p in model.parameters()
                    ]
                if target.type == "cuda":
                    torch.cuda.synchronize(target)
                elapsed = time.perf_counter() - started
                pipe.send(
                    (
                        "result",
                        index,
                        identity,
                        version,
                        elapsed,
                        loss.item(),
                        gradients,
                    )
                )
            else:
                pipe.send(
                    ("discard", index, identity, version, time.perf_counter() - started)
                )


class Runtime:
    def __init__(self, config):
        self.config = config
        context = mp.get_context("spawn")
        self.pipes, self.processes = [], []
        for index in range(config["workers"]):
            parent, child = context.Pipe()
            process = context.Process(target=worker_loop, args=(index, child, config))
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)
        for pipe in self.pipes:
            if not pipe.poll(config["startup_timeout_s"]):
                self.close()
                raise RuntimeError("Worker startup timed out")
            pipe.recv()
        self.windows = [deque(maxlen=20) for _ in self.pipes]
        self.estimates = [config["initial_task_s"]] * len(self.pipes)

    def iteration(self, state, x, y, step):
        config = self.config
        for pipe in self.pipes:
            pipe.send(("state", state))
        for pipe in self.pipes:
            if not pipe.poll(config["timeout_s"]):
                raise RuntimeError("Worker did not accept model state")
            pipe.recv()
        tasks = []
        count = int(np.ceil(len(x) / config["microbatch_size"]))
        for identity, offset in enumerate(range(0, len(x), config["microbatch_size"])):
            samples = min(config["microbatch_size"], len(x) - offset)
            primary = min(len(self.pipes) - 1, identity * len(self.pipes) // count)
            task = Task(
                offset,
                samples,
                primary,
                task_seed(config["seed"], step, offset, samples),
            )
            task.leases.add(primary)
            tasks.append(task)
        queues = [
            deque(i for i, task in enumerate(tasks) if task.primary == worker)
            for worker in range(len(self.pipes))
        ]
        busy = [None] * len(self.pipes)
        gradients, losses, records = {}, {}, []
        started = time.perf_counter()
        duplicate_s = 0.0
        copies = 0
        last_copy = -float("inf")
        cap = (
            len(tasks)
            if config["variant"] == "no_budget"
            else int(config["rho"] * len(tasks))
        )

        def dispatch(worker, identity):
            task = tasks[identity]
            task.leases.add(worker)
            task.expires = time.perf_counter() + config["lease_s"]
            busy[worker] = identity
            delay = (
                config["straggler_delay_s"]
                if worker == config["straggler_worker"]
                and step % config["straggler_every"] == 0
                else 0.0
            )
            self.pipes[worker].send(
                (
                    "task",
                    identity,
                    task.version,
                    x[task.offset : task.offset + task.count].clone(),
                    y[task.offset : task.offset + task.count].clone(),
                    task.seed,
                    delay,
                )
            )

        while any(task.state != "COMMITTED" for task in tasks) or any(
            value is not None for value in busy
        ):
            now = time.perf_counter()
            if now - started > config["timeout_s"]:
                raise RuntimeError(
                    "Iteration exceeded timeout; token owners must remain alive until commit"
                )
            for worker in range(len(self.pipes)):
                if busy[worker] is not None:
                    continue
                while queues[worker] and tasks[queues[worker][0]].state != "LEASED":
                    queues[worker].popleft()
                if queues[worker]:
                    dispatch(worker, queues[worker].popleft())
                    continue
                if config["variant"] == "sync" or copies >= cap:
                    continue
                if (
                    config["variant"] != "no_cooldown"
                    and now - last_copy < config["cooldown_s"]
                ):
                    continue
                remaining = [
                    sum(task.state == "LEASED" and task.primary == i for task in tasks)
                    for i in range(len(self.pipes))
                ]
                frontier = [
                    remaining[i] * self.estimates[i] for i in range(len(self.pipes))
                ]
                median = float(np.median(frontier))
                candidates = [
                    i
                    for i, task in enumerate(tasks)
                    if task.state == "LEASED"
                    and not task.speculated
                    and task.primary != worker
                    and remaining[task.primary] >= config["min_remaining"]
                    and (
                        config["variant"] == "no_prediction"
                        or frontier[task.primary] > median + config["delta_s"]
                    )
                ]
                if candidates:
                    identity = max(
                        candidates, key=lambda i: (frontier[tasks[i].primary], -i)
                    )
                    tasks[identity].speculated = True
                    copies += 1
                    last_copy = now
                    dispatch(worker, identity)
            for pipe in wait(
                [self.pipes[i] for i in range(len(self.pipes)) if busy[i] is not None],
                timeout=0.01,
            ):
                message = pipe.recv()
                kind, worker, identity, version = message[:4]
                task = tasks[identity]
                if kind == "token":
                    pipe.send(task.grant(worker, version))
                elif kind == "result":
                    elapsed, loss, values = message[4:]
                    task.commit(worker, version)
                    gradients[identity] = values
                    losses[identity] = loss
                    busy[worker] = None
                    window = self.windows[worker]
                    clipped = (
                        float(
                            np.clip(
                                elapsed,
                                np.quantile(window, 0.1),
                                np.quantile(window, 0.9),
                            )
                        )
                        if len(window) >= 5
                        else elapsed
                    )
                    self.estimates[worker] = (
                        config["ema"] * clipped
                        + (1 - config["ema"]) * self.estimates[worker]
                    )
                    window.append(elapsed)
                    records.append(
                        {
                            "task": identity,
                            "offset": task.offset,
                            "samples": task.count,
                            "seed": task.seed,
                            "version": version,
                            "worker": worker,
                            "accepted": True,
                            "elapsed_s": elapsed,
                        }
                    )
                elif kind == "discard":
                    elapsed = message[4]
                    duplicate_s += elapsed
                    busy[worker] = None
                    records.append(
                        {
                            "task": identity,
                            "offset": task.offset,
                            "samples": task.count,
                            "seed": task.seed,
                            "version": version,
                            "worker": worker,
                            "accepted": False,
                            "elapsed_s": elapsed,
                        }
                    )
                else:
                    raise RuntimeError(f"Unexpected worker report: {kind}")
            for identity, task in enumerate(tasks):
                if task.state == "LEASED" and task.expires and now > task.expires:
                    task.version += 1
                    task.leases = {task.primary}
                    task.speculated = False
                    task.expires = 0
                    queues[task.primary].appendleft(identity)
        if set(gradients) != set(range(len(tasks))) or sum(
            task.count for task in tasks
        ) != len(x):
            raise RuntimeError(
                "Accepted gradient set does not cover the global batch exactly"
            )
        summed = [
            sum(gradients[i][parameter] for i in range(len(tasks))) / len(x)
            for parameter in range(len(gradients[0]))
        ]
        useful_s = sum(row["elapsed_s"] for row in records if row["accepted"])
        return (
            summed,
            {
                "iteration_s": time.perf_counter() - started,
                "loss": sum(losses.values()) / len(x),
                "copies": copies,
                "redundant_compute": duplicate_s / max(useful_s, 1e-8),
                "accepted_samples": len(x),
            },
            records,
        )

    def close(self):
        for pipe, process in zip(self.pipes, self.processes):
            if process.is_alive():
                pipe.send(("stop",))
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        for pipe in self.pipes:
            pipe.close()
