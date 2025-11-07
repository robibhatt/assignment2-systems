from dataclasses import dataclass
import yaml
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
import torch
from torch import Tensor
from jaxtyping import Int
import timeit
import statistics

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


def get_benchmark_config():
    with open("cs336_systems/end_to_end_benchmark/end_to_end_benchmark.yaml") as f:
        data = yaml.safe_load(f)
    cfg = BenchmarkConfig(
        model_config=ModelConfig(**data["model_config"]),
        **{k: v for k, v in data.items() if k != "model_config"}
    )
    return cfg


def random_batch(config: BenchmarkConfig)->tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    batch_size = config.batch_size
    vocab_size = config.model_config.vocab_size
    context_length = config.model_config.context_length
    xy = torch.randint(low=0,
                       high=vocab_size,
                       size=(batch_size,context_length+1))
    x = xy[:, :-1]
    y = xy[:, 1:]
    return (x,y)




def run_benchmark(config: BenchmarkConfig):
    lm = BasicsTransformerLM(
        vocab_size=config.model_config.vocab_size,
        context_length=config.model_config.context_length,
        d_model=config.model_config.d_model,
        num_layers=config.model_config.num_layers,
        num_heads=config.model_config.num_heads,
        d_ff=config.model_config.d_ff,
        rope_theta=config.model_config.rope_theta,
    )

    lm.train()

    # --- Warm-up ---
    for _ in range(config.warmup_steps):
        for p in lm.parameters():
            if p.grad is not None:
                p.grad.zero_()
        x, y = random_batch(config)
        logits = lm(x)
        if config.include_backward:
            loss = cross_entropy(inputs=logits, targets=y)
            loss.backward()

    # --- Define a single timed step ---
    def step():
        for p in lm.parameters():
            if p.grad is not None:
                p.grad.zero_()
        x, y = random_batch(config)
        logits = lm(x)
        if config.include_backward:
            loss = cross_entropy(inputs=logits, targets=y)
            loss.backward()

    # --- CUDA sync helper ---
    def timed_step():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # --- Run repeats ---
    repeats = 5  # or config.repeats if you have that in your config
    results = timeit.repeat(timed_step, number=config.profile_steps, repeat=repeats)

    # Convert to per-step
    per_step_times = [t / config.profile_steps for t in results]
    mean_time = statistics.mean(per_step_times)
    std_time = statistics.stdev(per_step_times)

    # --- Write results ---
    with open("cs336_systems/end_to_end_benchmark/benchmark_results.out", "w") as f:
        f.write(str(config) + "\n")
        f.write(f"Mean step time: {mean_time:.6f}s ± {std_time:.6f}s (n={repeats})\n")

    print(f"Mean step time: {mean_time:.6f}s ± {std_time:.6f}s (n={repeats})")


if __name__ == "__main__":
    cfg = get_benchmark_config()
    run_benchmark(cfg)