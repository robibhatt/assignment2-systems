from dataclasses import dataclass
import yaml
from cs336_basics.model import BasicsTransformerLM


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


def run_benchmark(config: BenchmarkConfig):

    lm = BasicsTransformerLM(vocab_size=config.model_config.vocab_size,
                            context_length=config.model_config.context_length,
                            d_model=config.model_config.context_length,
                            num_layers=config.model_config.num_layers,
                            num_heads=config.model_config.num_heads,
                            d_ff=config.model_config.d_ff,
                            rope_theta=config.model_config.rope_theta)


    with open("cs336_systems/end_to_end_benchmark/benchmark_results.out", "w") as f:
        f.write(str(config) + '\n')
        f.write('yowza')


if __name__ == "__main__":
    cfg = get_benchmark_config()
    run_benchmark(cfg)