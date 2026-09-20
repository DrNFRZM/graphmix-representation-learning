# Design decisions

Each entry says what was decided, why, and what the alternative would have cost. Some of these
came from getting something wrong first; those are marked. Nothing here was tuned against test
labels.

## Scope and data

**A table, not a graph dataset.** The question is what happens when a graph is *constructed* from
heterogeneous records, so the data has to start as a table. Cora-style datasets already contain an
edge set that carries information the node features do not, which makes "graph vs no graph" a
comparison of different information, not of different inductive biases.

**UCI Adult.** Public, small enough for a laptop CPU (48,842 rows), mixed numeric and categorical
columns, a binary label that the encoders never see, CC BY 4.0, a stable download URL that needs no
account. It is old and overused, and it is low-dimensional, which is a real weakness for this
question (see the research note); the reason to accept it is that the download, the terms and the
citation could all be checked, and that nothing else considered was both as small and as
heterogeneous. Details and citation: [`data/README.md`](../data/README.md).

**Cleaning.**

* `fnlwgt` (a survey weight) and `education` (a string copy of `education-num`) are dropped.
* `?` becomes an explicit `Unknown` level of its column. It is not imputed, because imputation
  would be a model, and the pretext task later hides cells anyway.
* Exact duplicates are removed *before* splitting (6,374 of 48,842 rows once `fnlwgt` is gone).
  Without this, identical rows land in training and test, and a nearest-neighbour graph links them
  at distance zero. This matters more here than in a normal supervised setup.
* Categorical levels below 0.5 % of the training rows are merged into `__rare__`, per split,
  using training frequencies only.
* The original train/test split of the UCI files is ignored. Every seed draws its own stratified
  60/20/20 split so that the seeds differ in what they train on and are evaluated on.

## Protocol

**Inductive: the encoder trains on the training subgraph only.** Validation and test rows are
attached afterwards to the training pool and receive messages from it but never send. A training
embedding then cannot depend on an evaluation row, which is checked directly: replacing all test
rows by noise leaves training and validation outputs unchanged, bit for bit
(`tests/test_train.py`). The transductive alternative (build one graph over all rows, train on
all of it, hide the labels of test nodes) is common in node classification, but with a
self-supervised objective it lets test features shape training.

**Labels enter in one place.** `train_autoencoder` has no label argument. Labels are used by the
probes, by the homophily diagnostics, and by nothing else. Diagnostics are computed after all
representations exist.

**Everything that has parameters is fitted on training rows.** Standardisation, rare-level merging,
PCA, encoders. Validation rows choose (i) the training checkpoint (lowest validation reconstruction
loss) and (ii) the logistic-regression strength of the probe. Test rows are only scored.

## Graph construction

**The graph is a function of the same table.** kNN on the columns, or one node per categorical
value. No external information. This is the point of the design (research note, section 1).

**Gower-style distance.** Standardised numeric columns plus a fixed price per categorical
mismatch, implemented as squared Euclidean distance on `[z-scores, one-hot * sqrt(cost/2)]`. The
price (`cat_cost = 1`, one standard deviation of a numeric column per mismatch) is a guess. A
learned metric or per-column weights would be better and would be a different experiment.

**Brute-force kNN in float64, with deterministic ties.** The table is small enough that an
exhaustive search is affordable, and it avoids depending on an approximate-neighbour library whose
results vary between versions. Distances are rounded to 1e-6 and ties are broken by row index through an
integer key, because on Adult many rows are exactly identical in the columns, and BLAS rounding
noise would otherwise reorder them between runs. The cost is one dense block of distances per 1024
query rows, quadratic in the number of training rows: fine for 25k, not for 250k.

**Controls.** A random graph with the same in-degree and a self-loop graph. The first tells apart
"neighbourhoods matter" from "any smoothing helps"; the second tells apart "graph structure" from
"a GNN has more parameters than an MLP". `sage_selfloop` also has the same parameter count as the
kNN encoder, so the comparison is fair on capacity. It turned out not to be a control for
optimisation as well; see "Controls added after results", below.

**Bipartite record-value graph.** Every categorical cell is an edge to a (column, level) node. This
is the natural graph of heterogeneous records and needs no distance. The price is that numeric
columns are not part of the structure (they are node features of the records), and that very
common values connect nearly everything. The message-passing depth is fixed by the design:
value → record, record → value, value → record. Only training records send to value nodes. Its
control is `sage_bip_nopool`: every weight and embedding is kept and the record → value messages are
dropped, so a record sees only itself and its own values (added after four seeds had been evaluated;
see below).

**Propagation baselines (`prop_*`).** Parameter-free: $[x,\hat Ax,\hat A^2x]$ reduced to 16
dimensions with PCA. They separate "the graph contains useful smoothing" from "a network learned
something from it".

## Pretext task

**Denoising reconstruction, label-free.** A GNN cannot be compared with an MLP through a
supervised loss without giving the labels to the training, and then the question would be about
supervised learning. Masking cells and reconstructing the row is one of the simplest objectives
that applies to numeric and categorical columns alike and needs no augmentation library.
Contrastive objectives would be the natural next comparison. The mask rate (0.3) was chosen before
any run and not tuned.

**Loss on all cells, not only hidden ones.** The decoder must also reproduce the visible cells, so
the code is a representation of the row and not only a set of imputations. The masked-cell
imputation scores in the tables are reported separately.

**Mistake found and fixed: the neighbour-selection shortcut.** The first version built the kNN graph
once from the clean rows and masked cells during training. A development run (full data, one
seed, 2000 steps) gave a validation loss of about 0.07 for the kNN encoder against about 0.50 for
the MLP, and no downstream improvement. The graph was giving away the hidden cells: neighbours are
chosen because they resemble the row in all of its cells, including the hidden ones. The fix
(`src/graphmix/corruption.py`):

* Each corruption ("view") gets its own kNN graph, built from a copy of the corrupted rows in which
  every hidden cell is replaced by a random draw from the visible values of the same column. The
  graph is then a function of the visible cells and a seed only, which `tests/test_corruption.py`
  checks by changing exactly the hidden cells and asserting that the neighbour table does not
  move (and that the clean table does).
* Random fill instead of zero- or mean-fill, because filling every hidden cell with one constant
  would make rows with the same missingness pattern look alike and create artificial hubs.
* Validation and test rows are corrupted in the same way and find their neighbours among the clean
  training pool using their filled query row.
* The clean-graph variant is kept as `sage_knn_cleannbr`, a positive control: if it did not look
  much better on the pretext task, the fix would not be what I think it is.

**Eight fixed views, not fresh masks at every step.** A fresh corruption at every step would need a
fresh kNN search per step, and one search takes about 20 seconds on 25k rows and two CPU cores (164
seconds for the eight views of a seed and the validation and test queries, against 36 seconds for the
search on the clean rows). Training therefore cycles through `n_views = 8`
corruptions that are the same for all encoders. The cost is that each row is seen with only eight
corruption patterns, which shows up as a gap between training and validation loss (the
validation rows always get a corruption pattern that the model has not seen).

**Full-batch training.** 25k rows fit in memory, the graphs are small, and full-batch gradient
steps remove one source of randomness (mini-batch sampling) from a comparison whose differences
are expected to be small. It stops being an option at a much larger scale (neighbour sampling would
change the method).

## Models

**GraphSAGE with mean aggregation and a root weight.** Two layers, hidden 64, code 16. GCN's
symmetric normalisation is not used: with a separate root weight the two are close on regular
graphs, and the kNN graph is directed and has no natural symmetric form (a training row picks its
neighbours; nobody picks a row back). The self-loop and random controls use the same layers.

**The mean is computed as a sum over a row-normalised sparse matrix.** PyG's `aggr="sum"` on a CSR
tensor uses a fused sparse product; `aggr="mean"` on an edge list gathers a (edges × features)
tensor first. For the two layers in isolation on this data the difference was about 270 ms against
65 ms per forward-backward pass on two CPU cores (measured during development at hidden width 128; not
repeated for the final width of 64). `tests/test_models.py` checks that the two
give the same output (including repeated edges and nodes without senders).

**Shared decoder, loss and budget.** Same decoder (dim → 64 → 64 → heads), same loss, same
optimiser (Adam, lr 3e-3), same number of steps and the same corruption views for all encoders. The
step count was picked from the validation curves of one seed and is the same for everybody, so a
method that converges more slowly is disadvantaged; the training-curve figure shows how much. In the
final runs the best validation checkpoint is at or near the last step for every method (mean best step
1,920 to 2,000 of 2,000), so no method had stopped improving, and the comparison is one at a fixed
budget and not at convergence.

**Embedding dimension 16.** Small enough to be a bottleneck in principle. In practice clean rows are
reconstructed almost perfectly (categorical accuracy 0.994 to 0.998 on clean test rows for the
encoders trained with corruption), so the code is not tight, and
downstream differences between encoders are expected to be small. A smaller code would test
representation quality more strongly and is not done here.

**A known asymmetry in the record-value encoder.** Only training records send messages to value
nodes, so an evaluation record is never part of the value groups it receives from, whereas a training
record is part of its own groups. For common values that is negligible (a group of thousands); for a
rare value a training record's own numeric features are a visible share of the group mean it gets back.
The kNN graph does not have this problem (a training record is never its own neighbour). I left it
because the alternative (leaving the record out of its own groups) needs a per-record correction and
would make the encoder a different model from the one usually called GraphSAGE on a bipartite graph.

## Controls added after results

These controls, and the second paired table (against `sage_selfloop`), were added after I had seen
results, which is not how the design was planned. Research note, section 9, lists each with the point
at which it was added. All five seeds were trained for every one of them, in later invocations of the
same command.

**A self-loop GraphSAGE is a reparametrised MLP (found after seed 0).** I introduced `sage_selfloop`
as "the GNN architecture without relational information" and expected it to behave like the MLP.
Written out, a GraphSAGE layer on a graph whose only edge is the self-loop computes
`W_root x + W_neigh x = (W_root + W_neigh) x`: the function class of the MLP, with the matrix stored
as a sum of two. The two summands receive identical gradients (`tests/test_models.py` checks this and
the equivalence), and Adam moves every weight by a step of about the same size whatever its gradient
scale, so the sum moves about twice as far per step as the MLP's single matrix. At a fixed budget of
2,000 steps `sage_selfloop` is therefore a faster optimiser and not a different model. `mlp_ae_lr2x`,
the MLP with twice the learning rate, is the control for that. On five seeds it is +0.001 ± 0.004 in
probe AUC against `mlp_ae`, and `sage_selfloop` is +0.003 ± 0.005; the experiment resolves neither.
I keep `sage_selfloop` as the control for the kNN encoder, as designed, show `mlp_ae_lr2x` next to it,
and do not choose the more flattering baseline.

**No-corruption controls for the graph encoders (after seed 0).** `sage_knn_nomask` had no control of
its own, so a difference between it and `mlp_ae_nomask` could not be attributed to the neighbours.
`sage_selfloop_nomask` and `sage_rand_nomask` supply the missing comparisons.

**`sage_bip_nopool` (after four seeds).** By the fourth evaluated seed `sage_bip` was ahead of the
MLP in probe AUC in every seed, and I had no control that removes the graph part of that encoder
while keeping its parametrisation (learned value embeddings, three layers, three times the parameters
of the MLP). `sage_bip_nopool` empties the record → value edge set, so a record can no longer learn
anything from other records. It is a follow-up to a result I had already seen and should be read that
way. Its outcome is the main finding about the record-value graph (README, section 10): it reproduces
the whole edge.

## Evaluation

**Frozen embeddings, linear probe.** The point is what the representation contains, so the encoder
is never fine-tuned. The probe standardises every dimension with training statistics and picks the
regularisation on validation AUC.

**AUC, average precision, and low-label probes.** Only 24.6 % of the rows are positive, so
accuracy at 0.5 is mostly the majority rate. The low-label probe (n = 50 / 200 / 1000 labelled
rows, several random subsets) tests whether the representation makes the label easier to learn from
few examples, which is the usual argument for pretrained representations. The same subsets are used
for every method within a seed, so comparisons are paired.

**Retrieval.** Cosine kNN from test rows into training rows: a probe-free view of local structure.
It is where a representation that clusters by income should show up most directly.

**Paired differences, win counts, no p-values.** Five seeds of one dataset with overlapping splits
are not independent samples. Reported: mean ± std over seeds, the paired mean difference to
`mlp_ae` and the number of seeds where a method is ahead. Reading rules were written down before the
run (research note, section 7).

**Supervised gradient boosting as a reference (`hgb_ref`).** Not a representation. It marks what a
strong supervised model on the raw columns reaches, so the reader can see how far every frozen
16-dimensional code is from it.

**Figures do not zoom into noise.** The AUC axes have a minimum span (0.02 or 0.03), and the two panels of
the main comparison share one width, so that differences of the size of the seed spread do not fill a
panel. Open markers mean no training, filled markers a trained encoder; the leaky control has its own
marker and its name says so.

## Engineering

**Reproducibility.** Seeds drive numpy `Generator`s keyed by (seed, purpose, mask rate, view) and
`torch.manual_seed`; `torch.use_deterministic_algorithms(True, warn_only=True)` is on. Two runs with
the same seed give identical embeddings on the same machine (tested for every encoder). Bitwise
identity across machines or library versions is not promised: BLAS and sparse kernels may sum in a
different order.

**Two stages.** `train_encoders.py` writes one `.npz` of embeddings per (seed, method); `evaluate.py`
reads them and writes tables and figures. Evaluation can be redone (and probes changed) without
retraining, and a run that stops half way resumes.

**Configs are strict.** Unknown keys in the YAML are errors, so a misspelt setting cannot silently
fall back to a default. The config is copied next to the results.

**Small on purpose.** One module per concern (data, graphs, corruption, models, training,
evaluation, figures, pipeline), no plug-in registry, no base classes beyond what PyTorch needs. The
list of methods is a YAML list, and adding one is a few lines in `pipeline.py`.

## Not done, on purpose

* Hyperparameter search (of anything).
* GCN, GAT, deeper GNNs, learned graphs.
* Contrastive or generative pretext tasks.
* More datasets. One dataset done properly was preferred to several done superficially, and the
  conclusion is limited accordingly.
* The generative stage. Nothing in this repository samples or diffuses.
