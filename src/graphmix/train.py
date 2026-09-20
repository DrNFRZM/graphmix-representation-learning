"""Training of the denoising autoencoders (one loop for every encoder).

The training function receives features and a `Pretext` (corrupted views and the graphs that go
with them, see corruption.py); labels are never passed in, so the representations are label-free
by construction.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from graphmix.config import ModelConfig, TrainConfig
from graphmix.corruption import Pretext, View
from graphmix.graphs import Graph, row_normalized
from graphmix.models import Decoder, GraphTensors, build_encoder, reconstruction_loss


@dataclass
class TrainResult:
    z: np.ndarray  # (n_total, dim) embeddings from clean inputs, rows ordered [train | val | test]
    log: pd.DataFrame  # step, train_loss, val_loss
    n_params: int  # encoder parameters only
    best_step: int
    seconds: float
    recon_num_mse: float  # test rows, clean input, all cells
    recon_cat_acc: float  # test rows, clean input, mean over categorical columns
    masked_num_mse: float  # test rows, corrupted input, hidden numeric cells only
    masked_cat_acc: float  # test rows, corrupted input, hidden categorical cells (mean over columns)


@dataclass
class _Inputs:
    """One view, moved to the device."""

    m_num: Tensor
    m_cat: Tensor
    g: GraphTensors


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)  # numpy randomness in this package always goes through explicit Generators
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train.device is 'cuda' but no CUDA device is available")
    return torch.device(name)


def to_tensors(graph: Graph | None, n_total: int, n_train: int, device: torch.device) -> GraphTensors:
    """Homogeneous graphs become a row-normalised sparse CSR matrix; the bipartite encoder derives
    its (mask-dependent) adjacency from the categorical codes itself and only needs `is_train`."""
    adj = None
    if graph is not None:
        if graph.n_total != n_total:
            raise ValueError(f"graph has {graph.n_total} nodes but the inputs have {n_total} rows")
        if graph.kind != "bipartite":
            op = row_normalized(graph)
            adj = torch.sparse_csr_tensor(
                torch.from_numpy(op.indptr.astype(np.int64)),
                torch.from_numpy(op.indices.astype(np.int64)),
                torch.from_numpy(op.data.astype(np.float32)),
                size=op.shape,
                check_invariants=True,
            ).to(device)
    is_train = (torch.arange(n_total) < n_train).to(device)
    return GraphTensors(adj=adj, is_train=is_train)


def _to_device(view: View, n_total: int, n_train: int, device: torch.device) -> _Inputs:
    return _Inputs(
        m_num=torch.from_numpy(view.m_num).to(device),
        m_cat=torch.from_numpy(view.m_cat).to(device),
        g=to_tensors(view.graph, n_total, n_train, device),
    )


def _snapshot(*modules: nn.Module) -> list[dict[str, Tensor]]:
    return [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in modules]


def _scores(
    num_pred: Tensor,
    cat_logits: Sequence[Tensor],
    x_num: Tensor,
    x_cat: Tensor,
    m_num: Tensor | None = None,
    m_cat: Tensor | None = None,
) -> tuple[float, float]:
    """Numeric MSE and categorical accuracy (mean over columns). With masks, only the hidden
    cells count; the result is nan if nothing is hidden."""
    sq = (num_pred - x_num) ** 2
    if m_num is not None:
        sq = sq[m_num]
    mse = sq.mean().item() if sq.numel() else float("nan")
    accs = []
    for j, logits in enumerate(cat_logits):
        hit = logits.argmax(dim=1) == x_cat[:, j]
        if m_cat is not None:
            hit = hit[m_cat[:, j]]
        if hit.numel():
            accs.append(hit.float().mean())
    acc = torch.stack(accs).mean().item() if accs else float("nan")
    return mse, acc


def train_autoencoder(
    kind: str,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    cards: Sequence[int],
    pretext: Pretext,
    n_train: int,
    n_val: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    seed: int,
    verbose: bool = False,
) -> TrainResult:
    """Train encoder + decoder on the training rows and return embeddings for all rows.

    x_num / x_cat hold every row ordered [train | val | test]. Training cycles through the
    corrupted views of the training rows (each with its own graph); the encoder never sees
    anything but the training subgraph. Validation rows, corrupted with one fixed mask and
    attached to the clean training pool, pick the checkpoint with the lowest reconstruction loss.
    Test rows are only used at the very end, for the reported reconstruction scores.
    """
    started = time.time()
    device = resolve_device(train_cfg.device)
    seed_everything(seed)
    n_total, n_num = x_num.shape
    xn = torch.from_numpy(np.ascontiguousarray(x_num, dtype=np.float32)).to(device)
    xc = torch.from_numpy(np.ascontiguousarray(x_cat, dtype=np.int64)).to(device)
    tr, va, te = slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_total)

    train_views = [_to_device(v, n_train, n_train, device) for v in pretext.train]
    val_in = _to_device(pretext.val, n_total, n_train, device)
    test_in = _to_device(pretext.test, n_total, n_train, device)
    g_full = to_tensors(pretext.full, n_total, n_train, device)
    clean = [torch.zeros(n_total, n, dtype=torch.bool, device=device) for n in (n_num, x_cat.shape[1])]

    encoder = build_encoder(kind, n_num, cards, model_cfg.hidden, model_cfg.dim).to(device)
    decoder = Decoder(n_num, cards, model_cfg.hidden, model_cfg.dim).to(device)
    params = [*encoder.parameters(), *decoder.parameters()]
    opt = torch.optim.Adam(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    best_loss, best_step, best_state = float("inf"), 0, _snapshot(encoder, decoder)
    rows = []
    for step in range(1, train_cfg.steps + 1):
        encoder.train()
        decoder.train()
        view = train_views[(step - 1) % len(train_views)]
        z = encoder(xn[tr], xc[tr], view.m_num, view.m_cat, view.g)
        num_pred, cat_logits = decoder(z)
        loss, _, _ = reconstruction_loss(num_pred, cat_logits, xn[tr], xc[tr])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % train_cfg.eval_every == 0 or step == train_cfg.steps:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                z_all = encoder(xn, xc, val_in.m_num, val_in.m_cat, val_in.g)
                num_pred, cat_logits = decoder(z_all[va])
                val_loss, _, _ = reconstruction_loss(num_pred, cat_logits, xn[va], xc[va])
            rows.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss.item()})
            if rows[-1]["val_loss"] < best_loss:
                best_loss, best_step, best_state = rows[-1]["val_loss"], step, _snapshot(encoder, decoder)
            if verbose:
                print(
                    f"    step {step:4d}  train {rows[-1]['train_loss']:.4f}  val {rows[-1]['val_loss']:.4f}"
                )

    encoder.load_state_dict(best_state[0])
    decoder.load_state_dict(best_state[1])
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z_all = encoder(xn, xc, clean[0], clean[1], g_full)
        num_pred, cat_logits = decoder(z_all[te])
        recon_mse, recon_acc = _scores(num_pred, cat_logits, xn[te], xc[te])
        z_masked = encoder(xn, xc, test_in.m_num, test_in.m_cat, test_in.g)
        num_pred, cat_logits = decoder(z_masked[te])
        masked_mse, masked_acc = _scores(
            num_pred, cat_logits, xn[te], xc[te], test_in.m_num[te], test_in.m_cat[te]
        )

    return TrainResult(
        z=z_all.cpu().numpy(),
        log=pd.DataFrame(rows),
        n_params=sum(p.numel() for p in encoder.parameters()),
        best_step=best_step,
        seconds=time.time() - started,
        recon_num_mse=recon_mse,
        recon_cat_acc=recon_acc,
        masked_num_mse=masked_mse,
        masked_cat_acc=masked_acc,
    )
