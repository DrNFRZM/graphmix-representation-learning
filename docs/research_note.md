# Research note

This note fixes the question, the protocol and the criteria for reading the results. Numbers are
not repeated here: the current results are in [`results/full/summary.md`](../results/full/summary.md)
and are summarised in the README. Section 9 says what the evidence supports.

## 1. Research question

Given heterogeneous records (numeric and categorical columns), does turning them into a graph and
learning representations with a GNN give more useful representations than treating the same records
as independent flat feature vectors? And when it does, what did the graph construction have to
assume for that to happen?

"More useful" is operationalised through frozen representations: a small linear model, a
low-label linear model and nearest-neighbour retrieval are trained or run on the learned vectors,
and the encoder is never fine-tuned with labels. Reconstruction quality is reported as well, but it
is not the target.

The comparison is set up so that the graph cannot add information. The graph is a deterministic
function of the same table that the flat encoder sees (kNN on the columns, or one node per
categorical value). Any difference between a graph encoder and its matched no-neighbour control,
trained with the same objective and budget, can therefore be attributed to aggregating over other
records rather than to extra data. Datasets that ship with a graph (citation networks,
molecules, relational databases) ask a different question, because there the edges carry
information the row does not have.

## 2. What is deliberately not asked

* Whether GNNs beat gradient boosting on tabular data. There is a gradient-boosting reference in
  the tables (`hgb_ref`), only to show what a supervised model on the raw columns reaches. The
  literature on this is more careful than one dataset can be (Grinsztajn et al., 2022).
* Whether a better graph can be learned. The graph is fixed by a rule; graph structure learning
  (for example SLAPS, Fatemi et al., 2021) would be a next step.
* The generative stage of the thesis pipeline. Nothing here samples or diffuses; the embeddings are
  only evaluated as representations.

## 3. Hypotheses

The outcome is not known in advance. My expectation before the final runs, from a few
development runs on one seed, was that any gain would be small on Adult: twelve informative
columns, a 16-dimensional code that reconstructs clean rows almost perfectly, and a label that is
largely determined by a few columns. The hypotheses below are written so that they can fail.

* **H1.** An encoder that aggregates over a construction graph (kNN or record-value) yields a
  higher frozen-probe AUC than an MLP autoencoder with the same hidden size, code size, decoder,
  objective and step budget.
* **H2.** If H1 holds, the specific neighbourhoods matter: a degree-matched random graph does not
  give the same gain, and neither does a graph that only contains self-loops (this second control
  checks that the GNN architecture alone, with its extra weight matrix, is not what helps).
* **H3.** The gain is larger when few labels are available (probe fitted on 50 or 200 labelled
  rows) than when all training labels are used.
* **H4.** The gain is related to how well neighbours agree on the label: a kNN graph whose
  adjusted homophily is close to zero should not help.

## 4. Formulation

### 4.1 Data

A record is $x_i=(u_i,c_i)$ with numeric part $u_i\in\mathbb R^{p}$ ($p=5$, standardised with
training-row statistics) and categorical part $c_i=(c_{i1},\dots,c_{iq})$, $c_{ij}\in\{1,\dots,C_j\}$
($q=7$). The label $y_i\in\{0,1\}$ (income above 50K) is used only by the evaluation. Rows are
split into training, validation and test sets $\mathcal T,\mathcal V,\mathcal E$ (60/20/20,
stratified, one split per seed). Everything that has parameters (standardisation, rare-level
merging, encoders, decoder) is fitted on $\mathcal T$ only.

### 4.2 Graph construction

A construction rule produces, for every node $i$, a set of senders $\mathcal N(i)$. Message passing
goes from senders to receivers.

* **kNN.** $\mathcal N_k(i)$ is the set of the $k$ training rows nearest to $x_i$ under a Gower-style
  mixed distance,
  $$d^2(x_i,x_j)=\lVert u_i-u_j\rVert^2+\lambda\sum_{l=1}^{q}\mathbf 1[c_{il}\ne c_{jl}],\qquad \lambda=1,$$
  where $\lambda$ (`cat_cost`) is the price of one categorical mismatch relative to a one-standard-deviation
  gap in a numeric column (Gower, 1971, is the origin of this kind of distance). A training row is
  never its own neighbour.
* **Random (control).** $k$ senders per node drawn uniformly from the training rows, so that the
  degree matches the kNN graph but the neighbourhoods carry no information.
* **Self-loop (control).** $\mathcal N(i)=\{i\}$. A GNN on this graph has the parameters of the GNN
  but sees nothing besides the row.
* **Record-value (bipartite).** Besides the record nodes there is one node per (column, level).
  Every categorical cell of a record is an edge to the node of its value. Numeric columns are
  node features of the records. Only training records send messages to value nodes.

Validation and test rows are attached to the training pool with the same rule (their $k$ nearest
training rows, or their value nodes) and are receivers only: a training node never receives from an
evaluation node, so a training embedding cannot depend on evaluation rows. This is tested
(`tests/test_models.py`, `tests/test_train.py`).

### 4.3 Encoders

All encoders map a corrupted row $\tilde x_i$ to $z_i\in\mathbb R^{16}$. Write $\phi(\tilde x_i)$ for
the numeric part with hidden cells set to zero, one flag per numeric cell marking whether it was
hidden, and the one-hot blocks with hidden columns set to all zeros.

* **MLP:** $z_i=W_2\,\sigma(W_1\phi(\tilde x_i)+b_1)+b_2$.
* **GraphSAGE (two layers, mean aggregator, separate root weight):**
  $$h_i^{(l)}=\sigma\Big(W_r^{(l)}h_i^{(l-1)}+W_n^{(l)}\,\tfrac1{|\mathcal N(i)|}\sum_{j\in\mathcal N(i)}h_j^{(l-1)}+b^{(l)}\Big),\quad h^{(0)}=\phi(\tilde x),\ z=h^{(2)}$$
  with no non-linearity on the last layer (Hamilton et al., 2017). It runs on the kNN, random and
  self-loop graphs.
* **Record-value GraphSAGE:** value nodes have learned embeddings; layer 1 updates a record from the mean
  of its value embeddings and a value from the mean of the numeric features of the training records
  that share it; layer 2 updates the record from the layer-1 states of its values. A hidden
  categorical cell is a missing edge. A control variant (`sage_bip_nopool`) has the same weights but
  no record-to-value messages, so a record sees only its own cells.

The decoder is shared by all methods (two hidden layers, one head for the numeric columns and one
logit block per categorical column), so that the methods differ only in the encoder.

### 4.4 Pretext task, and a shortcut it created

Every training step uses one of $K=8$ fixed corruptions (cycled through), in which each cell of each
training row is hidden independently with probability $\rho=0.3$, and asks the decoder to recover the
*clean* row from $z_i$:
$$\mathcal L=\mathbb E_M\Big[\tfrac1{p}\sum_{a=1}^{p}(\hat u_{ia}-u_{ia})^2+\tfrac1q\sum_{l=1}^{q}\mathrm{CE}(\hat\pi_{il},c_{il})\Big],$$
averaged over rows, with the loss taken over all cells, visible and hidden (Vincent et al., 2008,
for the denoising idea). No labels are involved.

With a graph there is a trap. If the kNN graph is built from the clean rows, neighbours are chosen
because they resemble the row in *all* its cells, including the ones that are hidden in this step.
The encoder can then copy hidden cells from its neighbours, which is imputation with the answer
key. In a development run on the full data (one seed, 2000 steps) the validation loss of such an
encoder was about 0.07 against about 0.50 for the MLP, with no downstream gain. To remove the
shortcut, every corruption ("view") gets its own kNN graph, built from a copy of the corrupted rows
in which each hidden cell is replaced by a random draw from the visible values of that column. The
graph of a view is then a function of the visible cells and a seed only (this is tested), and all
encoders see the same $K=8$ fixed views. The variant with a clean graph is kept as a control
(`sage_knn_cleannbr`): it should look far better on the pretext task and no better downstream.

### 4.5 Evaluation

After training, embeddings are computed for all rows from *clean* inputs and the clean graph, and
frozen. Then, with labels:

* a logistic regression on the embedding, regularisation strength chosen on the validation AUC
  (probe AUC, average precision, accuracy on the test rows);
* the same probe (fixed $C$) fitted on $n\in\{50,200,1000\}$ labelled training rows, repeated on
  random subsets (low-label AUC);
* cosine kNN retrieval from test rows into training rows: AUC of the fraction of positive
  neighbours ($k=25$) and precision of the positive class among 10 neighbours.

Reconstruction of clean test rows, and imputation of cells hidden in test rows (with a graph built
from visible cells only), are reported for the learned encoders as descriptive numbers. Graph
diagnostics use labels and are computed after all representations exist: edge homophily, adjusted
homophily (Platonov et al., 2023), label agreement inside value groups for the bipartite graph, and
how often training nodes are picked as neighbours (hubness).

## 5. Experimental design

### Variables

| | |
|---|---|
| Varied | representation family (flat / graph), graph rule (kNN, random, self-loop, record-value), $k\in\{5,10,20\}$, corruption on or off, neighbour selection (visible cells or clean rows), seed and split |
| Held fixed | dataset and cleaning, 60/20/20 stratified split, code size 16, hidden size 64, decoder, loss, optimiser (Adam, lr 3e-3), number of full-batch steps, mask rate 0.3, distance ($\lambda=1$), probe family |
| Measured | probe AUC / AP / accuracy, low-label AUC, kNN AUC and positive precision, reconstruction and imputation scores, graph diagnostics, parameters, training time |

### Methods and what each one is for

| Method | Role |
|---|---|
| `raw` | flat features, no learning, linear probe (the number every representation has to justify itself against) |
| `pca` | flat, no training, same 16 dimensions as the learned codes |
| `mlp_ae` | flat and learned: the baseline for H1 |
| `sage_selfloop` | GNN architecture without relational information: separates architecture from structure |
| `mlp_ae_lr2x` | the MLP with twice the learning rate: tests whether the difference between `sage_selfloop` and `mlp_ae` is optimisation speed (added after seed 0 had been evaluated; section 9) |
| `sage_rand` | GNN on a degree-matched random graph: separates structure from neighbourhood quality |
| `sage_knn` | GNN on the kNN graph (main condition), `_k5` / `_k20`: graph-size ablation |
| `sage_bip` | GNN on the record-value graph (second main condition) |
| `sage_bip_nopool` | the same encoder with the record-to-value messages removed (same weights, no information from other records): separates the graph part of `sage_bip` from its different parametrisation (added after four seeds had been evaluated; section 9) |
| `prop_knn`, `prop_rand`, `prop_bip` | parameter-free graph baselines: $[x,\hat Ax,\hat A^2x]$ (one hop for the bipartite graph) reduced to 16 dimensions with PCA, in the spirit of SGC and SIGN (Wu et al., 2019; Frasca et al., 2020) |
| `*_nomask` | the same encoders trained without corruption: is the denoising objective doing the work? (`sage_selfloop_nomask` and `sage_rand_nomask` were added after seed 0 had been evaluated, so that `sage_knn_nomask` has controls of its own; section 9) |
| `sage_knn_cleannbr` | positive control for the neighbour-selection shortcut of section 4.4 |
| `hgb_ref` | supervised gradient boosting on the raw columns, not a representation |

### Statistics

There are five seeds. Each seed changes the split, the subsets used by the low-label probes, the
corruptions and the initialisation. Tables give mean ± standard deviation over seeds, and paired
differences to `mlp_ae` (same split, same probe subsets) with the number of seeds in which the
method is ahead; a second table gives the differences to `sage_selfloop`, which has the same layers
and parameters as the kNN encoder but no neighbours. No significance tests are reported: five splits of one dataset overlap heavily
and are not independent samples, so a p-value would suggest more than the design supports. The
spread is a rough indication of sensitivity to the split and the initialisation.

## 6. Assumptions that graph construction introduces

None of these is checked by the experiment, and each is a way the conclusion could fail to
transfer.

1. **Similarity in feature space is relevance for the task.** kNN aggregation helps only if rows
   that look alike also tend to share the target beyond what the row already says. The homophily
   diagnostics measure this after the fact.
2. **The distance encodes the right notion of similarity.** $\lambda$, the standardisation of the
   numeric columns, and treating every categorical mismatch as equally bad are choices. Different
   weights give different graphs. The value of $\lambda$ was set once and not tuned.
3. **A fixed $k$ suits every region of the data.** Dense regions and sparse regions get the same
   number of neighbours; hub rows appear (hubness is reported).
4. **Evaluation rows can be attached to the training pool.** This is an inductive protocol, and it
   assumes new rows come from the same distribution as the training pool.
5. **Only the visible cells decide the neighbours.** Section 4.4: this holds by construction for
   training, at the price of noisier graphs (30 % of the cells are random fill), which is a
   train/test mismatch: the embedding step at evaluation time uses the clean graph.
6. **A shared categorical value is a relation** (record-value graph). This ignores that some
   values are extremely common (a value shared by 90 % of records connects everything and tells
   little), and that numeric columns take no part in the structure.
7. **Duplicates are not signal.** Exact duplicate rows are removed before splitting. Without that,
   identical rows land in training and test and kNN links them trivially.

## 7. What would count against the hypotheses

Fixed before the five-seed run, using the metrics above (probe AUC, kNN AUC, AUC with 50 labels):

* H1 is met for a construction only if, on each of the three metrics, the mean paired difference to
  `mlp_ae` is larger than the standard deviation of the paired differences across seeds and the
  construction is ahead in at least four of the five seeds. Otherwise H1 is not supported for it.
  (This is the wording I use for the verdicts. The first version said "not clearly larger" and "on all
  three metrics" without defining either; section 9 says what changed and what depends on it.)
* H2 is not supported if `sage_rand` or `sage_selfloop` are within that same noise of `sage_knn`.
* H3 is not supported if the gap at 50 labels is not larger than the gap with all labels.
* H4 is not supported if the kNN graph has adjusted homophily near zero and still gains, or has
  clearly positive adjusted homophily and does not.
* The pipeline itself is not trustworthy if either of the following fails: `sage_selfloop` differs
  from `mlp_ae` by more than the seed noise (it has no relational information, only more
  parameters), or the clean-neighbour control does not show the shortcut (much better imputation of
  hidden cells than every other encoder). Label leakage into the representations and dependence of
  training outputs on evaluation rows are ruled out by construction and by tests, not by the
  experiment: the training function has no label argument, and modifying test rows leaves the
  training and validation outputs bit for bit unchanged.

A null result is a legitimate outcome here. It would say that for a low-dimensional table with
informative columns, a graph built from the same columns adds nothing that a same-size MLP does
not already have; it would not say that graphs are useless for heterogeneous data in general.

## 8. Limitations

* One dataset, low-dimensional (12 columns), 42k rows after removing duplicates. The
  conclusions may differ on wider tables, on tables with many rare categories, or where the columns
  are individually weak.
* Hyperparameters were not tuned: code size, hidden size, learning rate, step budget, mask rate,
  $k$ and $\lambda$ were fixed beforehand or chosen from the validation reconstruction loss of one
  seed. A tuned baseline, or a longer schedule, could change the ranking. The reconstruction
  objective is also not the downstream task, and a better pretext loss does not have to give a
  better probe.
* The step budget (2,000 full-batch steps) is equal for all methods and is not a convergence
  guarantee: in a 3,000-step development run on seed 0 the validation loss of the MLP was still
  falling (0.524 at step 2,000, 0.514 at step 3,000), and the curves in
  `results/full/figures/fig_training_curves.png` show the same at the end of every run. The budget
  was chosen from validation reconstruction loss on seed 0, which is also one of the five evaluation
  seeds; no probe metric was used for any choice, but seed 0 is not a clean hold-out.
* The number of corruption views is finite (eight), so training sees a limited set of corruption
  patterns for each row; the train/validation loss gap shows how much this matters.
* Five seeds on one dataset. The standard deviations describe sensitivity to the split and the
  initialisation, and do not support significance claims.
* Full-batch training on the CPU. Larger graphs would need neighbour sampling, which changes the
  method. The kNN search is brute force and quadratic in the number of training rows.
* The GPU path is implemented (`train.device`) but was only run on the CPU.
* Adult contains sensitive attributes (sex, race, native country) and is 1994 census data. They
  are ordinary input columns here. No fairness analysis was done, and nothing in the repository
  should be read as a statement about the people in the data.
* Only GraphSAGE-type encoders. Attention, deeper networks, learned graphs and other
  aggregators are not compared.
* The label-efficiency probe uses one fixed regularisation strength (C = 1) for every method. A
  strong penalty suits a 16-dimensional code fitted on 50 rows better than the 53-dimensional raw
  features, so the low-label comparison between `raw` and the 16-dimensional methods is partly a
  comparison of widths.
* In the record-value graph a training record belongs to the value groups it sends to, while an
  evaluation record does not (only training records send). For common values the difference is
  negligible; for a rare value a training record sees a group mean that includes its own numeric
  features.
* The record-value encoder differs from the MLP in more than the graph (learned embeddings per
  categorical value, 14,736 parameters against 4,816). `sage_bip_nopool` removes the graph and keeps
  the rest, so it answers the question about the graph and leaves open which of the other differences
  produces the small edge over the MLP; a wider MLP with embedding tables would be the control for that.

## 9. Status of the evidence

All five seeds of all 19 methods were trained and evaluated; the numbers below are from
`results/full/summary.md` and `results/full/metrics.csv` (mean ± standard deviation of the per-seed
paired difference to `mlp_ae`, seeds in which the construction is ahead in brackets).

| Criterion of section 7 | kNN graph (`sage_knn`) | record-value graph (`sage_bip`) |
|---|---|---|
| H1: ahead of `mlp_ae` on probe AUC, retrieval AUC and 50-label AUC | **not supported.** -0.001 ± 0.009 (3/5), +0.001 ± 0.002 (4/5), +0.004 ± 0.019 (3/5): fails on all three | **met.** +0.006 ± 0.003 (5/5), +0.001 ± 0.001 (4/5), +0.011 ± 0.010 (4/5): mean over standard deviation 1.8, 1.1 and 1.2 |
| H2: the specific neighbourhoods matter | **not supported.** `sage_selfloop` and `sage_rand` are within noise of it (probe AUC, `sage_knn` minus each: -0.004 ± 0.010 and -0.002 ± 0.010) | **not supported.** `sage_bip_nopool`, with no messages between records, is ahead of `mlp_ae` by the same amount (+0.006 ± 0.002, 5/5); `sage_bip` minus `sage_bip_nopool` is +0.0005 ± 0.003 |
| H3: the gain is larger with few labels than with all | **not informative.** Gap +0.004 at 50 labels, -0.001 with all labels; both inside the noise | **not informative.** Gap +0.011 at 50 labels, +0.006 with all labels, inside the noise at 50; `sage_bip_nopool` shows the same (+0.012, +0.006) |
| H4: the gain follows homophily | **not supported.** Adjusted homophily 0.43 on complete rows and 0.28 in the training views: clearly positive, and there is no gain | not applicable (label agreement inside value groups 0.661 against 0.629 by chance) |
| Pipeline: `sage_selfloop` within the seed noise of `mlp_ae` | passed: probe AUC +0.003 ± 0.005 (mean over standard deviation 0.6), retrieval -0.000 ± 0.002, 50 labels +0.004 ± 0.016 | |
| Pipeline: the clean-neighbour control shows the shortcut | passed: hidden numeric MSE 0.038 against 0.83 to 0.84 for the other encoders trained with corruption, hidden categorical accuracy 0.959 against 0.70 | |

The H3 criterion as I wrote it compares two means and ignores the noise, so it can be satisfied by
chance; its "not informative" is my reading of the numbers, not a verdict of the criterion.

What the evidence supports, on this dataset and under this protocol:

* **The kNN construction gave no benefit.** The encoder on the kNN graph is not distinguishable from
  the MLP, nor from the self-loop and random-graph controls; k = 5, 10, 20 show no trend; and it is
  the least stable across seeds (standard deviation of the probe AUC 0.007, against 0.003 for the
  MLP; I did not investigate why). This holds although the graph is clearly homophilous, so the
  assumption "records that look alike share the label" is true here in the sense of homophily and
  does not matter for the representation, presumably because the features already contain it.
* **The record-value construction has a small, consistent edge that does not come from the graph.**
  The control that removes the record-to-value messages has the same edge. What the two share and the
  MLP lacks is a different parametrisation (learned embedding tables per categorical value, three
  times the parameters); which part matters is not tested here (a capacity-matched MLP would).
* **The self-loop control is not the same model as the MLP.** A self-loop GraphSAGE is a
  reparametrised MLP whose weights move about twice as fast under Adam, so at a fixed step budget it is
  a different optimiser. The learning-rate control shows no effect beyond the noise, so the difference
  between the self-loop encoder and the MLP (+0.003) is neither confirmed nor excluded as an
  optimisation effect.
* **The denoising objective changes what the model can do and not what the probes see.** Without
  corruption the encoders cannot impute hidden cells (categorical accuracy 0.43 to 0.51, below the mode
  baseline of 0.574) and are level with the corrupted ones in probe AUC; at 50 labels they are nominally
  ahead in 16 of 20 seed-method pairs, by amounts of about one standard deviation.
* **The pretext task is not the downstream task.** The clean-neighbour encoder reconstructs hidden
  cells almost perfectly and is no better downstream.

What it does not support:

* A statement that graphs cannot help. Differences below about 0.005 in probe AUC (0.01 to 0.02 at
  50 labels) are not resolved, and there is no positive control for the downstream metrics, that is,
  no case in which a graph is known to help and the probes are shown to notice it.
* Anything about other datasets. Twelve informative low-cardinality columns leave little for a
  neighbourhood to add, which is the regime in which I expected a null.
* A ranking of the learned encoders among themselves. The clear separations are the raw features
  (0.895) and the gradient-boosting reference (0.927) above everything else, and PCA (0.886) above
  the MLP autoencoder (0.881, five of five seeds); the learned encoders lie between 0.878 and 0.887.

### Decisions made after results were seen

The list is complete as far as I can tell; everything else (methods, seeds, metrics, hyper-parameters)
was fixed before the five-seed run and was not changed or dropped afterwards.

* The neighbour-selection shortcut (section 4.4) was found in a development run on one seed
  (validation loss 0.07 against 0.50) and fixed before any reported run. The clean-neighbour control
  exists because of it.
* Hidden width (64), the number of corruption views (8) and the step budget (2,000) were set from the
  compute budget and from the validation reconstruction loss of seed 0 (section 8). No probe metric
  entered.
* After seed 0 had been trained and evaluated I added three controls, because its results showed
  that the comparisons had gaps: `sage_selfloop_nomask` and `sage_rand_nomask` (the no-corruption
  kNN encoder had no control of its own) and `mlp_ae_lr2x` (a self-loop GraphSAGE layer differs from
  the MLP only by the way its weights are parametrised, and I wanted to know whether that alone
  changes the result). I also added a second paired table, against `sage_selfloop`, to the summary.
* A fourth control, `sage_bip_nopool`, was added after four of the five seeds had been evaluated,
  when `sage_bip` was ahead of the MLP in probe AUC in each of them and I had no control that removes
  the graph part of that encoder while keeping its parametrisation. So it is a follow-up to a
  result I had already seen, and should be read that way.
* All five seeds were trained for the added methods in the same way as for the others.
* The wording of the H1 criterion in section 7 was made unambiguous after the results were in. The
  first version said the construction fails if the difference is "not clearly larger" than the standard
  deviation, or if it is ahead in at most three seeds, "on all three metrics". I had not defined
  "clearly", and the last phrase could mean "fails on all three" or "has to hold on all three". The
  version now in section 7 uses "larger than one standard deviation" and requires all three metrics.
  It changes nothing for the kNN graph, which fails on every metric under any reading. For the
  record-value graph, the verdict is the same under both readings of the last phrase (it meets the
  criterion on all three), but it depends on the meaning of "clearly": with a bar of two standard
  deviations instead of one it would fail all three (mean over standard deviation 1.8, 1.1, 1.2).
  Either way the control `sage_bip_nopool` decides the interpretation, not this criterion.

## References

The full list with sources is in the README.
