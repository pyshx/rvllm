"""Engine configuration -- mirrors rvllm-config crate."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model: str
    dtype: str = "auto"
    max_model_len: int | None = None
    enforce_eager: bool = False


@dataclass
class CacheConfig:
    block_size: int = 16
    gpu_memory_utilization: float = 0.90
    swap_space_gb: float = 4.0


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int | None = None
    max_prefill_chunk: int | None = None
    preemption_mode: str = "recompute"  # "recompute" or "swap"


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1


@dataclass
class EngineConfig:
    model: ModelConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    @classmethod
    def from_args(cls, args) -> "EngineConfig":
        return cls(
            model=ModelConfig(
                model=args.model,
                dtype=args.dtype,
                max_model_len=args.max_model_len,
                enforce_eager=args.enforce_eager,
            ),
            cache=CacheConfig(gpu_memory_utilization=args.gpu_memory_utilization),
            scheduler=SchedulerConfig(
                max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=args.max_num_batched_tokens,
            ),
            parallel=ParallelConfig(tensor_parallel_size=args.tensor_parallel_size),
        )
