from pathlib import Path

import pytest
import yaml

from graphmix.config import ALL_KINDS, GRAPH_OF_KIND, NEURAL_KINDS, PROP_KINDS, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def write_cfg(tmp_path: Path, **overrides: object) -> Path:
    raw: dict[str, object] = {
        "name": "t",
        "seeds": [0, 1],
        "methods": [{"name": "raw", "kind": "raw"}, {"name": "sk", "kind": "sage_knn", "k": 5}],
    }
    raw.update(overrides)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_load(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.name == path.stem
    assert {m.kind for m in cfg.methods} <= set(ALL_KINDS)
    # every neural method must have a graph description (or be the flat MLP)
    for m in cfg.methods:
        if m.kind in NEURAL_KINDS and m.kind != "mlp_ae":
            assert m.kind in GRAPH_OF_KIND


def test_kind_tables_are_consistent() -> None:
    assert set(PROP_KINDS) | set(NEURAL_KINDS) | {"raw", "pca"} == set(ALL_KINDS)
    assert set(GRAPH_OF_KIND) <= set(ALL_KINDS)


def test_overrides_fall_back_to_defaults(tmp_path: Path) -> None:
    cfg = load_config(write_cfg(tmp_path, graph={"k": 7}, train={"mask_ratio": 0.4}))
    raw, sk = cfg.methods
    assert cfg.k_of(raw) == 7 and cfg.k_of(sk) == 5
    assert cfg.mask_of(raw) == 0.4
    assert cfg.run_path == Path("runs/t") and cfg.result_path == Path("results/t")


def test_mask_ratio_override_of_zero_is_respected(tmp_path: Path) -> None:
    methods = [{"name": "ae0", "kind": "mlp_ae", "mask_ratio": 0.0}]
    cfg = load_config(write_cfg(tmp_path, methods=methods))
    assert cfg.mask_of(cfg.methods[0]) == 0.0  # 0.0 must not be mistaken for "not set"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"datta": {}}, "unknown top-level"),
        ({"train": {"stepz": 3}}, "unknown key"),
        ({"methods": [{"name": "a", "kind": "nope"}]}, "unknown kind"),
        ({"methods": [{"name": "a", "kind": "raw"}, {"name": "a", "kind": "pca"}]}, "unique"),
        ({"methods": [{"name": "a", "kind": "sage_knn", "k": 0}]}, "k must be"),
        ({"methods": [{"name": "a", "kind": "mlp_ae", "mask_ratio": 1.0}]}, "mask_ratio"),
        ({"methods": [{"name": "a", "kind": "mlp_ae", "lr": 0.0}]}, "lr must be"),
        ({"seeds": []}, "seeds"),
        ({"data": {"val_frac": 0.6, "test_frac": 0.5}}, "sum to less than 1"),
        ({"train": {"device": "tpu"}}, "device"),
        ({"model": {"dim": 0}}, "dim"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(write_cfg(tmp_path, **overrides))


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
