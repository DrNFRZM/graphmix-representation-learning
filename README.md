# graphmix-representation-learning

Controlled study of whether graph-based structural priors improve representation learning for
heterogeneous tabular records. The experiments use UCI Adult and compare raw features, PCA, a
denoising MLP autoencoder, parameter-free propagation, and GraphSAGE over kNN, random, self-loop,
and record-value graphs.

**Main result:** under this dataset and protocol, no tested graph construction demonstrated a
meaningful advantage over its matched no-neighbour control. GraphSAGE on the kNN graph reached
`0.880 ± 0.007` linear-probe AUC, versus `0.881 ± 0.003` for the MLP autoencoder and
`0.884 ± 0.006` for self-loop GraphSAGE. The record-value encoder reached `0.887 ± 0.002`, but
removing record-to-value pooling produced `0.887 ± 0.003`; its small numerical edge therefore cannot
be attributed to cross-record messages. Raw features reached `0.895 ± 0.002`, and a supervised
gradient-boosting reference reached `0.927 ± 0.002`.

Values above are mean ± standard deviation across five random seeds. The seeded splits overlap, so
the spreads are sensitivity summaries, not confidence intervals or independent replications.

## Reproduce a smoke run

Python 3.11 or newer is required. A CPU-only PyTorch installation is sufficient.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"

make data      # download and verify UCI Adult
make test      # unit and end-to-end tests
make quick     # 6,000 rows, one seed, all code paths
```

`make quick` is a smoke test; its one-seed results are not intended for scientific interpretation.

## Research question

Does a graph constructed from heterogeneous records provide a useful inductive bias for
self-supervised representation learning when the graph contains no information beyond the table
used by a non-graph encoder?

The project deliberately treats a negative result as an outcome rather than a presentation problem.
It does not claim that graph learning is ineffective in general. It shows that these particular
constructions, encoder, objective, and controls did not establish a benefit on this benchmark.

## Dataset and scope

[UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) contains 48,842 records with numeric and
categorical attributes. The pipeline drops `fnlwgt` and the redundant string `education` column,
represents `?` as an explicit `Unknown` level, and removes 6,374 exact duplicates before splitting,
leaving 42,468 records. Each seed creates a stratified 60/20/20 train/validation/test split.

The preprocessor is fitted on training rows only:

- `log1p` is applied to `capital-gain` and `capital-loss`, followed by numeric standardisation;
- categorical levels below 0.5% frequency in training are merged into `__rare__`;
- validation and test rows are transformed without refitting.

The code is MIT-licensed. The Adult dataset is separately licensed by UCI under CC BY 4.0 and is
not committed to this repository. Its source, citation, file hashes, and cleaning details are in
[`data/README.md`](data/README.md).

Adult includes `sex`, `race`, and `native-country`. They are used here as benchmark input attributes
for a representation-learning experiment. No fairness or bias analysis is performed, so these
results must not be interpreted as evidence about fairness.

## Methodology

### Representations and controls

| Family | Methods | Purpose |
|---|---|---|
| Flat, no training | `raw`, `pca` | Input reference and 16-dimensional linear baseline |
| Flat, learned | `mlp_ae` | Denoising autoencoder without neighbours |
| Homogeneous GraphSAGE | `sage_knn`, `sage_rand`, `sage_selfloop` | kNN condition and matched structure/architecture controls |
| Record-value GraphSAGE | `sage_bip`, `sage_bip_nopool` | Categorical value nodes, with and without cross-record pooling |
| Parameter-free graph | `prop_knn`, `prop_rand`, `prop_bip` | Tests whether graph smoothing helps without learned message passing |
| Supervised reference | `hgb_ref` | Gradient boosting on flat features; not a learned representation |

The kNN graph uses a mixed distance: squared gaps between training-standardised numeric features plus
a fixed cost for each categorical mismatch. Search is brute-force; distances are quantised to `1e-6`
before deterministic index-based tie-breaking. Random graphs match the incoming degree, and self-loop
graphs contain no neighbour information. The record-value construction adds one node per
`(categorical column, level)` pair.

All neural encoders produce 16-dimensional representations and use the same decoder, reconstruction
loss, optimiser, training budget, and corruption views. GraphSAGE uses two mean-aggregation layers
with separate root weights. The denoising objective independently hides 30% of cells and reconstructs
the clean row; labels are not accepted by the training function.

### Leakage controls

The protocol is inductive. Encoders train on the training subgraph. Validation and test records may
receive messages from training records but cannot send messages into training nodes or value-node
aggregates.

A first development version selected kNN neighbours from clean rows while asking the model to
reconstruct hidden cells. That leaks information through neighbour selection. The reported method
instead builds each corrupted view's kNN graph from visible cells, replacing hidden cells with seeded
draws from the corresponding visible training distribution. Tests verify that changing hidden values
does not change the resulting neighbour table. The leaky version is retained only as the explicitly
named positive control `sage_knn_cleannbr`.

## Evaluation

Representations are frozen before labels are used. Evaluation includes:

- logistic-regression probes, with `C` chosen on validation AUC;
- low-label probes using 50, 200, or 1,000 labelled training rows;
- cosine-kNN retrieval from test rows into the training set;
- clean-row reconstruction and masked-cell reconstruction;
- graph diagnostics such as edge homophily, adjusted homophily, and hubness.

Low-label subsets are shared across methods within each seed. Reported tables use the sample standard
deviation across seeds and paired per-seed differences. No p-values or confidence intervals are
reported because the five runs reuse overlapping rows from one dataset.

## Results

Selected linear-probe AUCs from the committed full run:

| Method | AUC (mean ± SD, five seeds) | Interpretation |
|---|---:|---|
| `raw` | 0.895 ± 0.002 | Uncompressed flat features |
| `pca` | 0.886 ± 0.002 | 16-dimensional non-neural baseline |
| `mlp_ae` | 0.881 ± 0.003 | Flat denoising autoencoder |
| `sage_selfloop` | 0.884 ± 0.006 | GraphSAGE architecture without neighbours |
| `sage_rand` | 0.882 ± 0.005 | Degree-matched random neighbours |
| `sage_knn` | 0.880 ± 0.007 | Main kNN graph condition |
| `sage_bip` | 0.887 ± 0.002 | Record-value graph |
| `sage_bip_nopool` | 0.887 ± 0.003 | Same record-value model, no cross-record pooling |
| `hgb_ref` | 0.927 ± 0.002 | Supervised reference, not a representation |

The principal ablations support the same conclusion:

- kNN GraphSAGE at `k = 5, 10, 20` produced AUCs of 0.878, 0.880, and 0.881, with no useful trend;
- `sage_bip - sage_bip_nopool` was `+0.0005 ± 0.0030`, so record-to-value messages did not explain
  the record-value model's numerical edge over the MLP;
- parameter-free kNN and random propagation differed from PCA by less than 0.001 AUC;
- removing input corruption changed downstream AUC only within the seed-to-seed spread;
- the clean-neighbour leakage control sharply improved hidden-cell reconstruction but not downstream
  AUC, confirming that a better pretext score can be a shortcut rather than a better representation;
- the complete-row kNN graph had adjusted homophily around 0.43, yet its encoder did not improve,
  showing that feature-derived homophily alone is not evidence that aggregation adds value.

Full per-seed metrics, paired differences, reconstruction results, graph diagnostics, training logs,
and figures are in [`results/full/`](results/full/). The interpretation and post-result decisions are
documented in [`docs/research_note.md`](docs/research_note.md).

## Full experiment

```bash
make train       # five seeds x 19 configured methods; resumable
make evaluate    # probes, tables, diagnostics, and figures -> results/full/
```

The full CPU run is intentionally expensive; the committed experiment took roughly 3.7 hours for
training, followed by about 12 minutes for evaluation, on two CPU cores. Saved embeddings are
written to `runs/` and are not tracked. Existing embeddings are skipped unless
`scripts/train_encoders.py --overwrite` is used.

Key outputs:

- `results/full/metrics.csv`: every per-seed metric;
- `results/full/summary.md`: tables generated from the metrics;
- `results/full/graph_diagnostics.csv`: graph statistics by seed;
- `results/full/training_logs.csv`: training and validation losses;
- `results/full/config.yaml` and `run_info.json`: settings and software versions;
- `results/full/figures/`: generated plots.

## Limitations

- This is one low-dimensional, old, heavily studied dataset; it does not establish a general result
  for tabular graph learning.
- Hyperparameters were not tuned. The fixed step budget was informed by validation reconstruction
  loss for seed 0, which is also one of the reported seeds.
- Several controls were added after partial results were visible. Their timing is disclosed in the
  research note.
- There is no downstream positive control in which a graph is known to help, so the evaluation may
  be insensitive to small gains.
- The record-value encoder has about three times the MLP's encoder parameters and learned value
  embeddings. `sage_bip_nopool` rules out cross-record pooling as the explanation for its small edge,
  but does not identify which remaining modelling difference causes it.
- Training uses eight fixed corruption views and clean inputs at inference, creating a train/inference
  mismatch that was not isolated.
- Only a two-layer mean-aggregation GraphSAGE family was studied. The graph metric, mismatch cost,
  and construction rules were only lightly ablated.
- The brute-force kNN implementation is quadratic, full-batch training is CPU-oriented, and the CUDA path
  was not part of the reported run.

## Reproducibility and project notes

- Seeds control splitting, corruption, low-label subsets, random graphs, and PyTorch initialisation.
- Preprocessing, PCA, encoders, probe selection, and graph reference pools respect split boundaries.
- The committed result configuration matches `configs/full.yaml`; the result CSVs contain all five
  seeds without duplicate `(seed, method, metric)` rows.
- GitHub Actions runs the test suite and static checks on Python 3.11 with CPU PyTorch.
- Bitwise equality is expected only on the same platform and dependency versions; BLAS and sparse
  kernels can differ across systems.

For implementation rationale and honest interview preparation, see
[`docs/design_decisions.md`](docs/design_decisions.md) and
[`docs/interview_guide.md`](docs/interview_guide.md).

## References

1. Becker, B. and Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository.
   <https://doi.org/10.24432/C5XW20>
2. Hamilton, W. L., Ying, R. and Leskovec, J. (2017). Inductive representation learning on large
   graphs. *NeurIPS*. <https://arxiv.org/abs/1706.02216>
3. Vincent, P., Larochelle, H., Bengio, Y. and Manzagol, P.-A. (2008). Extracting and composing robust
   features with denoising autoencoders. *ICML*. <https://doi.org/10.1145/1390156.1390294>
4. Gower, J. C. (1971). A general coefficient of similarity and some of its properties.
   *Biometrics* 27(4), 857–871.
5. Fey, M. and Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric.
   <https://arxiv.org/abs/1903.02428>
6. Platonov, O. et al. (2023). A critical look at the evaluation of GNNs under heterophily.
   *ICLR*. <https://arxiv.org/abs/2302.11640>
7. Grinsztajn, L., Oyallon, E. and Varoquaux, G. (2022). Why do tree-based models still outperform
   deep learning on typical tabular data? *NeurIPS*. <https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html>
8. Gorishniy, Y. et al. (2021). Revisiting deep learning models for tabular data. *NeurIPS*.
   <https://arxiv.org/abs/2106.11959>
9. Fatemi, B., El Asri, L. and Kazemi, S. M. (2021). SLAPS: self-supervision improves structure
   learning for graph neural networks. *NeurIPS*.
   <https://proceedings.neurips.cc/paper/2021/hash/bf499a12e998d178afd964adf64a60cb-Abstract.html>
10. You, J. et al. (2020). Handling missing data with graph representation learning. *NeurIPS*.
    <https://proceedings.neurips.cc/paper/2020/hash/dc36f18a9a0a776671d4879cae69b551-Abstract.html>
11. Wu, F. et al. (2019). Simplifying graph convolutional networks. *ICML*.
    <https://arxiv.org/abs/1902.07153>
12. Frasca, F. et al. (2020). SIGN: scalable inception graph neural networks. *ICML Workshop*.
    <https://arxiv.org/abs/2004.11198>

Author: Farzam Nikbakhsh Jorshari.

“GraphMix” is a working project name and is unrelated to the semi-supervised GraphMix regulariser.
