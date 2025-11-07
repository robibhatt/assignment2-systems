from dataclasses import dataclass
import yaml
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
import torch
from torch import Tensor, dtype
from jaxtyping import Int
import time
import statistics
from datetime import datetime
import os
import shutil
import torch.cuda.nvtx as nvtx


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float

    def __str__(self):
        return (
            "ModelConfig:\n"
            f"  vocab_size={self.vocab_size}\n"
            f"  context_length={self.context_length}\n"
            f"  d_model={self.d_model}\n"
            f"  num_layers={self.num_layers}\n"
            f"  num_heads={self.num_heads}\n"
            f"  d_ff={self.d_ff}\n"
            f"  rope_theta={self.rope_theta}"
        )


@dataclass
class BenchmarkConfig:
    model_config: ModelConfig
    batch_size: int
    seq_length: int
    warmup_steps: int
    profile_steps: int
    include_backward: bool

    def __str__(self):
        return (
            "BenchmarkConfig:\n"
            f"  batch_size={self.batch_size}\n"
            f"  seq_length={self.seq_length}\n"
            f"  warmup_steps={self.warmup_steps}\n"
            f"  profile_steps={self.profile_steps}\n"
            f"  include_backward={self.include_backward}\n"
            f"  {str(self.model_config).replace(chr(10), chr(10)+'  ')}"
        )


def get_benchmark_config_and_output():
    print('started the process')

    # get the config
    with open("scripts/end_to_end/end_to_end_benchmark.yaml") as f:
        data = yaml.safe_load(f)
    cfg = BenchmarkConfig(
        model_config=ModelConfig(**data["model_config"]),
        **{k: v for k, v in data.items() if k != "model_config"}
    )
    print('got the config')

    # get the new experiment directory name
    dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.mkdir("scripts/end_to_end/results/" + dir_name)
    print('created results directory')

    # copy the yaml file into it
    shutil.copy("scripts/end_to_end/end_to_end_benchmark.yaml", "scripts/end_to_end/results/"+dir_name+'/end_to_end_benchmark.yaml')
    print("copied the yaml file")

    return cfg, "scripts/end_to_end/results/"+dir_name+'/results.out'


def random_batch(config: BenchmarkConfig, device:torch.device)->tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    batch_size = config.batch_size
    vocab_size = config.model_config.vocab_size
    context_length = config.model_config.context_length
    xy = torch.randint(low=0,
                       high=vocab_size,
                       size=(batch_size,context_length+1),
                       device=device)
    x = xy[:, :-1]
    y = xy[:, 1:]
    return (x,y)


def run_benchmark(config: BenchmarkConfig, out_file: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lm = BasicsTransformerLM(
        vocab_size=config.model_config.vocab_size,
        context_length=config.model_config.context_length,
        d_model=config.model_config.d_model,
        num_layers=config.model_config.num_layers,
        num_heads=config.model_config.num_heads,
        d_ff=config.model_config.d_ff,
        rope_theta=config.model_config.rope_theta
    )
    print('created model')

    lm.to(device)
    print('moved model to gpu')

    lm.train()
    print('put model in training mode')

    # --- Warm-up (wrapped for GUI filtering) ---
    with nvtx.range("WARMUP"):
        for _ in range(config.warmup_steps):
            print('started warm up step')
            for p in lm.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            print('zeroed out gradients')
            x, y = random_batch(config, device)
            print('grabbed a batch')

            grad_ctx = torch.enable_grad() if config.include_backward else torch.no_grad()
            with grad_ctx:
                with nvtx.range("FORWARD (warmup)"):
                    logits = lm(x)
            print('forward pass')

            if config.include_backward:
                loss = cross_entropy(inputs=logits, targets=y)
                with nvtx.range("BACKWARD (warmup)"):
                    loss.backward()
                print('backward pass')
            print('did a warm-up step')

    # --- Profile each step individually (separate fwd/bwd) ---
    fwd_times = []
    bwd_times = []  # only filled if include_backward
    print('started real profiling')

    # Coarse range to “zoom to” in Nsight Systems
    with nvtx.range("PROFILED_STEPS"):
        for _ in range(config.profile_steps):
            # prep
            for p in lm.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            x, y = random_batch(config, device)

            # ----- forward timing -----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            grad_ctx = torch.enable_grad() if config.include_backward else torch.no_grad()
            with grad_ctx:
                with nvtx.range("FORWARD (profile)"):
                    logits = lm(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            fwd_times.append(t1 - t0)

            # ----- backward timing (optional) -----
            if config.include_backward:
                loss = cross_entropy(inputs=logits, targets=y)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                tb0 = time.perf_counter()
                with nvtx.range("BACKWARD (profile)"):
                    loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                tb1 = time.perf_counter()
                bwd_times.append(tb1 - tb0)

            print('did a profile step')
    
    # --- Stats helpers ---
    def mean_std(xs):
        if not xs:
            return None, None
        m = statistics.fmean(xs)
        s = statistics.pstdev(xs) if len(xs) == 1 else statistics.stdev(xs)
        return m, s

    fwd_mean, fwd_std = mean_std(fwd_times)
    bwd_mean, bwd_std = mean_std(bwd_times)

    # --- Write results ---
    out_lines = [
        str(config),
        f"Forward time per step: {fwd_mean:.6f}s ± {fwd_std:.6f}s (n_steps={len(fwd_times)})",
    ]
    if config.include_backward:
        out_lines.append(
            f"Backward time per step: {bwd_mean:.6f}s ± {bwd_std:.6f}s (n_steps={len(bwd_times)})"
        )
        total_times = [f + b for f, b in zip(fwd_times, bwd_times)]
        tot_mean, tot_std = mean_std(total_times)
        out_lines.append(
            f"Total (fwd+bwd) per step: {tot_mean:.6f}s ± {tot_std:.6f}s (n_steps={len(total_times)})"
        )

    out = "\n".join(out_lines) + "\n"
    with open(out_file, "w") as f:
        f.write(out)


if __name__ == "__main__":
    print('we going hard')
    cfg, out_file = get_benchmark_config_and_output()
    print('obtained config')
    run_benchmark(cfg, out_file)
