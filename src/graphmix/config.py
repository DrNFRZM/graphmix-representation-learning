"""Experiment configuration: plain dataclasses loaded from YAML, with strict key checking."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Representation methods. "raw" / "pca" / "prop_*" need no torch.
NEURAL_KINDS = ("mlp_ae", "sage_selfloop", "sage_knn", "sage_rand", "sage_bip", "sage_bip_nopool")
PROP_KINDS = ("prop_knn", "prop_rand", "prop_bip")
ALL_KINDS = ("raw", "pca", *PROP_KINDS, *NEURAL_KINDS)

# Which graph (if any) a method uses.
GRAPH_OF_KIND = {
    "prop_knn": "knn",
    "sage_knn": "knn",
    "prop_rand": "random",
    "sage_rand": "random",
    "prop_bip": "bipartite",
    "sage_bip": "bipartite",
    "sage_bip_nopool": "bipartite",
    "sage_selfloop": "selfloop",
}


@dataclass
class DataConfig:
    raw_dir: str = "data/raw/adult"
    files: list[str] = field(default_factory=lambda: ["adult.data", "adult.test"])
    n_rows: int | None = None  # stratified subsample of the cleaned table; None keeps everything
    min_freq: float = 0.005  # categorical levels rarer than this (in train) are merged
    val_frac: float = 0.2
    test_frac: float = 0.2


@dataclass
class GraphConfig:
    k: int = 10
    cat_cost: float = 1.0  # squared-distance cost of one categorical mismatch (1 SD numeric gap = 1)
    chunk: int = 1024  # query rows per block in the brute-force neighbour search


@dataclass
class ModelConfig:
    hidden: int = 64
    dim: int = 16


@dataclass
class TrainConfig:
    steps: int = 2000
    lr: float = 3e-3
    weight_decay: float = 0.0
    mask_ratio: float = 0.3  # share of cells hidden by the corruption (denoising objective)
    n_views: int = 8  # fixed corruption patterns that training cycles through (see corruption.py)
    eval_every: int = 50
    device: str = "cpu"  # "cpu", "cuda" or "auto"


@dataclass
class EvalConfig:
    c_grid: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])
    label_sizes: list[int] = field(default_factory=lambda: [50, 200, 1000])
    label_repeats: int = 5
    knn_k: int = 25
    tsne_points: int = 3000
    hgb_reference: bool = True


@dataclass
class MethodConfig:
    name: str
    kind: str
    k: int | None = None  # neighbours for knn / random graphs (default: graph.k)
    mask_ratio: float | None = None  # per-method override of train.mask_ratio
    lr: float | None = None  # per-method override of train.lr (only the optimisation control uses it)
    # sage_knn only. "visible": every corruption gets its own kNN graph, built from the cells that
    # stay visible in it. "clean": one graph from the uncorrupted rows, which lets the encoder read
    # the hidden cells off its neighbours (a shortcut; kept as a control, see docs/design_decisions.md)
    neighbors: str = "visible"


@dataclass
class Config:
    name: str
    seeds: list[int]
    methods: list[MethodConfig]
    runs_dir: str = "runs"
    results_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    @property
    def run_path(self) -> Path:
        return Path(self.runs_dir) / self.name

    @property
    def result_path(self) -> Path:
        return Path(self.results_dir) / self.name

    def k_of(self, method: MethodConfig) -> int:
        return method.k if method.k is not None else self.graph.k

    def mask_of(self, method: MethodConfig) -> float:
        return method.mask_ratio if method.mask_ratio is not None else self.train.mask_ratio


def _build(cls: type, raw: dict[str, Any] | None, where: str) -> Any:
    raw = dict(raw or {})
    unknown = set(raw) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown key(s) in '{where}': {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> Config:
    """Read a YAML file into a validated Config. Unknown keys are errors, not silently ignored."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    raw = dict(raw)
    methods = [_build(MethodConfig, m, "methods") for m in raw.pop("methods", [])]
    sections = {
        "data": DataConfig,
        "graph": GraphConfig,
        "model": ModelConfig,
        "train": TrainConfig,
        "evaluation": EvalConfig,
    }
    for key, cls in sections.items():
        raw[key] = _build(cls, raw.get(key), key)
    top_unknown = set(raw) - {f.name for f in fields(Config)}
    if top_unknown:
        raise ValueError(f"unknown top-level key(s): {sorted(top_unknown)}")
    cfg = Config(methods=methods, **raw)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if not cfg.seeds:
        raise ValueError("seeds must not be empty")
    names = [m.name for m in cfg.methods]
    if len(names) != len(set(names)):
        raise ValueError("method names must be unique")
    for m in cfg.methods:
        if m.kind not in ALL_KINDS:
            raise ValueError(f"method '{m.name}': unknown kind '{m.kind}' (choose from {ALL_KINDS})")
        if m.k is not None and m.k < 1:
            raise ValueError(f"method '{m.name}': k must be >= 1")
        if m.mask_ratio is not None and not 0.0 <= m.mask_ratio < 1.0:
            raise ValueError(f"method '{m.name}': mask_ratio must be in [0, 1)")
        if m.lr is not None and m.lr <= 0:
            raise ValueError(f"method '{m.name}': lr must be > 0")
        if m.neighbors not in ("visible", "clean"):
            raise ValueError(f"method '{m.name}': neighbors must be 'visible' or 'clean'")
        if m.neighbors != "visible" and m.kind != "sage_knn":
            raise ValueError(f"method '{m.name}': 'neighbors' only applies to kind sage_knn")
    d = cfg.data
    if not (0 < d.val_frac < 1 and 0 < d.test_frac < 1 and d.val_frac + d.test_frac < 1):
        raise ValueError("val_frac and test_frac must be in (0, 1) and sum to less than 1")
    if not 0.0 <= d.min_freq < 0.5:
        raise ValueError("data.min_freq must be in [0, 0.5)")
    if cfg.graph.k < 1 or cfg.graph.cat_cost <= 0:
        raise ValueError("graph.k must be >= 1 and graph.cat_cost > 0")
    if cfg.model.dim < 1 or cfg.model.hidden < 1:
        raise ValueError("model.dim and model.hidden must be >= 1")
    t = cfg.train
    if t.steps < 1 or t.eval_every < 1 or t.n_views < 1 or t.lr <= 0 or not 0.0 <= t.mask_ratio < 1.0:
        raise ValueError("invalid train section (steps, eval_every, n_views, lr, mask_ratio)")
    if t.device not in ("cpu", "cuda", "auto"):
        raise ValueError("train.device must be 'cpu', 'cuda' or 'auto'")
    e = cfg.evaluation
    if not e.c_grid or e.label_repeats < 1 or e.knn_k < 1:
        raise ValueError("invalid evaluation section")
