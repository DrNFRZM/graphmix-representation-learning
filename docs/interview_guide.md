# Interview guide

Notes for explaining this repository to someone who knows the field: a supervisor, a PhD committee
member, an interviewer. Every number refers to [`results/full/summary.md`](../results/full/summary.md)
(five seeds, mean ± standard deviation over seeds). If a number here disagrees with that file, the
file is right.

## Numbers to have in your head

| | |
|---|---|
| Data | UCI Adult, 48,842 rows, 42,468 after removing exact duplicates; 5 numeric and 7 categorical columns; 24.6 % positive |
| Split | stratified 60/20/20 per seed: 25,480 / 8,494 / 8,494 rows |
| Encoders | 16-dimensional code, hidden width 64; encoder parameters: MLP 4,816, GraphSAGE 9,552, record-value 14,736 |
| Training | 2,000 full-batch Adam steps, learning rate 3e-3, 30 % of cells hidden, 8 fixed corruption views, best-validation-loss checkpoint |
| Graphs | kNN (k = 10, Gower-style distance), degree-matched random, self-loop, record-value |
| Seeds | 5 (split, corruption views, probe subsets and initialisation all change with the seed) |
| Headline | No graph construction beat its matched no-neighbour control. Probe AUC: kNN GraphSAGE 0.880 ± 0.007, MLP 0.881 ± 0.003, self-loop control 0.884 ± 0.006; raw features 0.895, gradient boosting 0.927 |

## 1. The 30-second version

My thesis pipeline turns heterogeneous data into a graph, learns representations with a GNN, and
later puts a generative model on top. Before building on that, I wanted to know whether the graph
step earns its place at all. So I took one public table, UCI Adult, and compared an MLP autoencoder
with GraphSAGE autoencoders on graphs built from the same table (kNN, and a record-value graph),
with controls: a random graph of the same degree and a self-loop graph. The graph carries no
information the table does not, so any gain has to come from aggregating over neighbours.
There was no such gain: the kNN encoder is indistinguishable from the MLP and from the no-neighbour
controls, and the one encoder that came out ahead (the record-value graph, +0.006 AUC in all five
seeds) is matched by the same network with record-to-value pooling removed, so the control receives
no information from other records. The most useful things I took away are about method:
a kNN graph built from clean rows lets the model copy the masked cells from its neighbours, and
homophily that a graph inherits from its features is not evidence that aggregation adds anything.

## 2. The two-minute version

**The question.** If I build a graph from heterogeneous records, do GNN representations beat
representations of the same records treated as independent vectors, and what did the construction
have to assume? Most GNN evaluations use datasets that come with a graph, where the edges carry
information the node does not. I wanted the opposite: a graph that is a deterministic function of the
same table, so that the only thing it can add is an inductive bias.

**The design.** One dataset, one label-free pretext task (denoising reconstruction with 30 % of the
cells hidden), one decoder, one optimiser and one budget for every encoder. The encoders differ only
in what they may look at besides the record: nothing (MLP), the record itself through a GNN layer
(self-loop), random other records, its k nearest records, or the records that share its categorical
values. Representations are 16-dimensional and are judged frozen: a linear probe, a probe with 50 to
1,000 labels, and nearest-neighbour retrieval. The protocol is inductive: the encoder trains on the
training subgraph only, and validation and test records are attached afterwards as receivers.

**The trap.** My first kNN encoder looked spectacular on the pretext task (validation loss 0.07 against
0.50 for the MLP) and gave nothing downstream. Neighbours are chosen because they look like the
record in all its cells, including the hidden ones, so the encoder was reading the answer off its
neighbours. The fix is to build each corrupted view's graph from the visible cells only. I kept the
leaky version as a positive control: in the final runs (settings slightly different from that first
one) it reaches a validation loss of 0.10 (0.53 for the MLP) and imputes hidden numeric cells with a
mean squared error of 0.038, against 0.83 for the MLP and 0.84 for the fixed graph encoder, and it is
no better downstream.

**The result.** Probe AUC is 0.880 ± 0.007 for GraphSAGE on the kNN graph and 0.881 ± 0.003 for the
MLP (paired difference -0.001 ± 0.009, ahead in 3 of 5 seeds); retrieval AUC and the 50-label probe
show the same picture, and the self-loop and random-graph controls are at least as good (0.884 and
0.882). The record-value graph is ahead of the MLP by +0.006 ± 0.003 in all five seeds, but
`sage_bip_nopool`, the same network with no messages between records, is ahead by the same amount, so
that edge is not from the graph. The kNN graph has an adjusted homophily of 0.43 and its encoder gains
nothing; every 16-dimensional code is behind the raw features (0.895) and far behind gradient
boosting (0.927).

**What I would and would not claim.** On this dataset, under this protocol, a kNN or record-value graph
built from the same columns did not give a better representation than a matched encoder without
neighbours; the assumption that similar records share the label held (homophily 0.43) and did not
matter. I would not claim anything about graphs on tabular data in general: one low-dimensional
dataset, no tuning, five overlapping splits. The controls and the leakage tests are what I trust most.

**Next.** A planted-structure control to show that the probe can detect a gain when one exists,
wider and noisier tables, tuned baselines, a contrastive pretext, then the generative stage.

## 3. The detailed explanation

1. **Why Adult.** It is the only dataset I found that was simultaneously public without registration,
   automatically downloadable, mixed numeric and categorical, small enough for a laptop CPU, and
   licensed clearly (CC BY 4.0, DOI on the dataset page). Its weaknesses are real: 1994 data,
   twelve usable columns, sensitive attributes, heavily used. The conclusion is limited accordingly.
2. **Cleaning.** `fnlwgt` (a survey weight) and `education` (a copy of `education-num`) are dropped;
   `?` becomes an explicit `Unknown` level; exact duplicates are removed before splitting (6,374
   rows), because otherwise identical rows land in training and test and a kNN graph links them at
   distance zero. Levels rarer than 0.5 % of the training rows share one `__rare__` level.
3. **Splits and preprocessing.** One stratified 60/20/20 split per seed; z-scores, rare-level merging,
   PCA and all network weights are fitted on training rows only.
4. **Graphs.** kNN uses a Gower-style distance (squared Euclidean on z-scored numerics plus a fixed
   price per categorical mismatch, implemented through one-hot coordinates); the search is exact,
   float64, with distances rounded and ties broken by index so that results do not depend on BLAS
   rounding. Random graphs have the same in-degree. Self-loop graphs contain only the record itself.
   The record-value graph has one node per (column, level) and an edge for every categorical cell.
5. **Inductive protocol.** Validation and test records receive messages from the training pool and
   never send. A test replaces all test records by noise and asserts that training and validation
   outputs do not change.
6. **Pretext.** Every cell of every training row is hidden with probability 0.3 (numeric cells zeroed
   and flagged, categorical cells lose their one-hot block); the decoder reconstructs the *clean* row
   and the loss covers visible and hidden cells, so the code is a representation of the row and not
   only an imputation.
7. **The shortcut and its fix.** See section 2 and `src/graphmix/corruption.py`. Training cycles
   through eight fixed views, each with its own kNN graph built from the visible cells (hidden cells
   are refilled with random draws from the visible values of their column). All encoders see the same
   views. Cost: each row is seen with eight corruption patterns only.
8. **Encoders.** MLP; two-layer GraphSAGE with mean aggregation and a separate root weight (so a
   useless neighbourhood can be ignored); a record-value GraphSAGE with learned value embeddings. The
   mean is computed as a sum over a row-normalised CSR matrix, which is equivalent (tested) and about
   four times faster on two CPU cores.
9. **Evaluation.** Frozen embeddings; probe C chosen on validation AUC; low-label probes with the same
   subsets for all methods; cosine kNN retrieval; reconstruction and imputation scores with a
   column-mean/mode reference; graph diagnostics (edge and adjusted homophily, per-column label
   agreement for the record-value graph, hubness).
10. **Statistics.** Five seeds, paired per-seed differences to the MLP with win counts, no p-values.
11. **Results and limits.** See section 2 and the README.

## 4. What was implemented, and where

| Module | Contents |
|---|---|
| `data.py` | download and verification of the UCI files, parsing, cleaning, splits, the training-fitted `Preprocessor` |
| `graphs.py` | Gower-style features, exact brute-force kNN, kNN / random / self-loop / record-value graphs, parameter-free propagation, homophily and hubness diagnostics (numpy and scipy only) |
| `corruption.py` | masks, random fill, the per-view kNN tables that remove the shortcut, the `Pretext` bundle (no torch) |
| `models.py` | MLP, GraphSAGE and record-value encoders, the shared decoder, the loss |
| `train.py` | one training loop for every encoder, checkpoint selection, reconstruction and imputation scores |
| `baselines.py` | PCA and propagated features |
| `evaluate.py` | probes, retrieval, gradient-boosting reference, tables |
| `pipeline.py` | the two stages: `run_training` (embeddings to `runs/`) and `run_evaluation` (tables and figures to `results/`) |
| `viz.py`, `config.py` | figures; strict YAML configuration |
| `tests/` | leakage properties (bit-for-bit invariance to test rows, neighbour tables invariant to hidden cells, sensitivity of the test itself), equivalence of the fast mean aggregator, determinism of every encoder, config validation, end-to-end runs on a synthetic table |

## 5. Data flow

```
adult.data + adult.test                                   48,842 rows
  clean(): drop fnlwgt/education, '?' -> Unknown, drop 6,374 duplicates
                                                           42,468 rows (5 numeric, 7 categorical, label)
    make_splits(seed): stratified 60 / 20 / 20             25,480 / 8,494 / 8,494
      Preprocessor.fit(train): log1p on capital-*, z-score, rare levels
                                                           x_num (n, 5)   x_cat (n, 7) codes, cardinalities 8,7,15,6,5,2,5
        flat features (n, 53)  ->  raw, PCA(16)
        GraphFactory: kNN table (k up to 20), random, self-loop, record-value graph
        Corruptor: 8 training views (masks + one kNN table per view), validation view, test view
        train_autoencoder(kind, x, Pretext)                -> embeddings z (n, 16), from clean inputs, best-validation checkpoint
          evaluate_embedding(z, y): probe, low-label probe, retrieval -> metrics.csv -> summary.md, figures
        graph_diagnostics(): uses labels, runs after all embeddings exist
```

## 6. Why graph learning, and why it might not help

A record-only encoder computes `z_i = f(x_i)`. A graph encoder computes `z_i = f(x_i, {x_j : j in N(i)})`.
If the neighbours are chosen from the same columns, `N(i)` is a function of the data, so the pair
`(x_i, {x_j})` contains no information about the label that the full table does not; the
question is whether this particular function class is a better place to look for structure. Three
mechanisms could make it so:

* **Denoising through redundancy.** If columns are noisy or missing and neighbours are redundant
  measurements of the same underlying entity, averaging over them helps.
* **Local statistics.** The neighbourhood tells the encoder about its local region (local class
  balance, local value ranges), which an MLP has to store in its weights.
* **Shared value nodes.** In the record-value graph rare categorical values borrow statistical
  strength from the records that share them, like an embedding table with smoothing.

Each mechanism needs something to hold: noisy features, a local structure that varies across the
space, or high-cardinality categories with rare values. Adult has twelve fairly clean columns, low
cardinalities (at most 15 levels after merging) and a label largely determined by a handful of them,
so I expected any gain to be small, and I wrote that down before the final runs.

## 7. Why these baselines

| Alternative explanation of a gain | Control that tests it |
|---|---|
| "Any encoder trained this way reaches this level" | `raw` and `pca` (no training), `mlp_ae` (flat, trained) |
| "A GNN has more parameters than an MLP" | `sage_selfloop`: same layers, same parameter count, no neighbours |
| "The GNN just trains faster than the MLP at a fixed step budget" | `mlp_ae_lr2x`: the MLP with twice the learning rate |
| "The record-value encoder is just a differently parametrised flat model" | `sage_bip_nopool`: same weights, value nodes pool nothing from other records |
| "Smoothing over any random records helps" | `sage_rand`: same degree, random neighbours |
| "The graph has useful smoothing, but a network is not needed" | `prop_knn`, `prop_rand`, `prop_bip`: [x, Âx, Â²x] with PCA, no weights |
| "The denoising objective is doing the work" | `mlp_ae_nomask`, `sage_selfloop_nomask`, `sage_rand_nomask`, `sage_knn_nomask` |
| "The neighbours give the answer away" | `sage_knn_cleannbr`: neighbours from clean rows, the positive control for the shortcut |
| "A supervised model would do much better anyway" | `hgb_ref`: gradient boosting on the raw columns, for scale only |
| "The neighbourhood size matters" | `sage_knn_k5`, `sage_knn`, `sage_knn_k20` |

## 8. What could go wrong

* **Leakage.** Checked in four ways: by construction (no label argument in training; evaluation
  rows are receivers only; the fill of hidden cells draws from visible values), by tests (bitwise
  invariance, neighbour tables invariant to hidden cells), by a positive control that shows the
  shortcut when it is present, and by keeping validation-selected choices (checkpoint, probe C)
  separate from the test rows. Not covered: my choice of step budget looked at seed 0's validation
  loss, and seed 0 is also an evaluation seed.
* **Train/inference mismatch.** Training sees corrupted inputs and noisy graphs; the embeddings that
  are evaluated use clean inputs and the clean graph. The MLP has the same mismatch in its inputs;
  the graph encoders have it in their neighbourhoods too. Not measured separately.
* **Statistics.** Five overlapping splits of one dataset. Differences below the seed-to-seed spread
  are not results.
* **Probe sensitivity.** A linear probe may not separate methods that a non-linear probe would; the
  retrieval metric is the only non-parametric check. There is no positive control for the
  downstream metrics (a case where a graph is known to help).
* **Tuning.** Nothing was tuned, for any method. A tuned baseline can change a ranking of small
  differences.
* **Scale.** Full-batch training and an O(n²) kNN search stop being options at a much larger scale;
  neighbour sampling and approximate search would change the method.
* **Sensitive attributes.** Adult contains sex, race and native country and is 1994 data; no
  fairness analysis was done.

## 9. With three more months

1. **A planted-structure control.** Generate data in which a graph *does* carry label information the
   row lacks, and check that the pipeline detects the gain. Without it, a null result on real data
   leaves open that the protocol is insensitive.
2. **When does it help?** Sweep the informativeness of the columns (drop columns, inject noise or
   missingness) and the cardinality of the categorical columns, and plot the graph encoders' gain
   over the MLP against them.
3. **Tuned baselines and more datasets.** A small equal-budget search over width, learning rate and
   steps for every method, on a handful of tabular datasets of different width and cardinality, with
   a paired comparison across datasets. This includes an MLP with learned embedding tables and the
   capacity of the record-value encoder, to see where its small edge over the flat MLP comes from.
4. **Other pretext tasks.** A contrastive objective with two independent corruptions of a record as
   the positive pair, on the same graphs.
5. **The graph itself.** Per-column distance weights or a learned graph (SLAPS-like), and a sweep of the
   categorical mismatch price, which is the assumption I trust least.
6. **The generative stage.** Fit a diffusion model to the codes of the best encoder, decode, and
   compare samples with real rows (marginals, correlations, utility of synthetic data, memorisation).

## 10. Fifteen hard questions

**1. The graph is computed from the same columns the MLP sees. It cannot contain more information about
the label. Why did you expect a gain, and what would a gain have meant?**
You are right that no information is added; a gain could only come from the inductive bias,
that is, from a different function class. A GNN layer averages over similar records, which is a learned
low-pass filter over feature space plus a skip connection, and that can help if the features are
noisy or missing and neighbours are redundant, or if local statistics matter. I expected it to help
little on Adult, because the columns are clean and the label is largely determined by a few of them,
and I wrote the hypotheses so that they could fail. A gain would have meant that this bias suits
this regime, not that the graph "knows" more. What happened is that it helped by nothing measurable
(question 6), which is close to what I had expected: a small effect at most.

**2. You found a leak. How do you know no leak is left?**
Four layers. By construction: the training function has no label argument, evaluation rows only
receive, and hidden cells are refilled with draws from the visible values of their column, so the
graph of a view is a function of visible cells and a seed. By tests: replacing test rows by noise
does not move training or validation outputs, and changing exactly the hidden cells does not move
a view's neighbour table (while changing them *does* move the clean table, so the test can fail).
By a positive control: with clean neighbours the model imputes hidden numeric cells with an error of
0.038 against 0.83 for the MLP and 0.84 for the fixed graph encoder (the column mean gives 0.992),
so the diagnostic is sensitive, and the fixed encoder shows nothing of that size. What I cannot rule
out with these tools is a leak much smaller than the gap between the MLP and the clean control; the
disclosed exception is that my step budget looked at seed 0's validation loss.

**3. Why denoising reconstruction, and not a contrastive or supervised objective?**
It is label-free, applies to numeric and categorical columns alike, needs no augmentation library,
and is the simplest objective in which a graph could plausibly help (imputing a hidden cell from
similar records). A supervised objective would turn the question into supervised graph learning, and
a contrastive one would be the natural next comparison; it is in the roadmap. Reconstruction is
not the downstream task: the 16-dimensional code reconstructs clean test rows almost perfectly
(categorical accuracy 0.995), so the codes keep nearly all the information, and the probe results
say what is linearly accessible, not what is retained.

**4. What does a linear probe on frozen embeddings measure, and what does it not?**
It measures how linearly accessible the label is in the coordinates the encoder happens to use,
after per-dimension standardisation. It does not measure what a fine-tuned model or a non-linear
probe could extract, and a non-linear probe could reorder methods whose differences are small. I
added a few-label probe and a non-parametric retrieval metric to look at the same question from other
sides, but I did not run a non-linear probe.

**5. Five seeds: what can you claim statistically?**
The seeds change the split, the corruption views, the probe subsets and the initialisation, so
the spread describes sensitivity to those on this dataset. The five test sets come from the
same 42k rows and overlap, so the runs are not independent and the spread understates uncertainty
about other data. I report paired differences and win counts and no p-values, and I read a difference
as real only if it is clearly larger than the spread and consistent in sign across seeds. A claim
about tabular data in general would need many datasets and a paired test across datasets.

**6. The differences between the encoders are small. What is the conclusion of the study?**
That on this dataset, under this protocol, none of the graph constructions gives frozen
representations that are better than a matched encoder without neighbours by more than the
seed-to-seed variation. The kNN encoder is level with the MLP (-0.001 ± 0.009 in probe AUC) and with the
self-loop and random-graph controls (0.884 and 0.882 against 0.880). The record-value encoder is ahead
of the MLP in all five seeds (+0.006 ± 0.003), which is more than the noise, but the same network with
the record-to-value messages removed is ahead by the same amount (+0.006 ± 0.002), so the edge is not
evidence for cross-record message passing. The two variants share learned value embeddings, greater
capacity, and other parametrisation differences; I did not isolate which of those explains it. The
controls otherwise behave as they should, and the homophily a feature-derived graph shows (0.43) does
not translate into a gain. The study cannot say that the protocol would detect a small gain: the
clean-neighbour control shows sensitivity on the pretext task, but there is no positive control for
the downstream metrics. That is the first thing I would add.

**7. Why do the raw features (53 dimensions) beat every 16-dimensional embedding in probe AUC, and what are the representations for then?**
A lossy unsupervised compression cannot beat its own input for a linear probe with enough labels: the
raw features reach 0.895 and the codes 0.878 to 0.887, and a supervised tree model on the raw columns
reaches 0.927, so the codes are far from what the columns can give. In this experiment the
representations do not pay off. Where they look better than the raw features is at 200 labels (0.850 to
0.861 against 0.835), but PCA to 16 dimensions is as good there (0.856), so what helps is the
reduction in width under a fixed regularisation strength (C = 1), which favours narrow inputs when
labels are few; that is a confound in the comparison and not evidence that anything was learned. At 50
labels the seed spread (up to 0.017) is larger than the differences between methods.

**8. Inductive or transductive: what is your protocol and why?**
Inductive. The encoder is trained on the training subgraph only, and validation and test rows are
attached afterwards as receivers. Transductive node-classification benchmarks build one graph
over all nodes and train with all features visible; with a self-supervised objective that lets
test features shape training. The inductive protocol matches deployment (new records arrive
after training) and is easy to test, at the price that evaluation rows see only the training pool
as neighbours.

**9. The categorical mismatch price is arbitrary. How sensitive are your results to it?**
I do not know; I did not vary it. It sets the balance between numeric and categorical columns in the
distance and therefore which records become neighbours and how homophilous the graph is. I chose
one standard deviation of a numeric column per mismatch, in the spirit of Gower. The experiment I
would run is a sweep over the price, plotting adjusted homophily and probe AUC against it; a better
answer is per-column weights or a learned graph. The hubness diagnostics (in-degree skew of the
kNN graph) show that even at this price some rows are chosen as neighbours far more than others.

**10. Why adjusted homophily, what does a value of about 0.4 mean, and does high homophily imply a GNN gain?**
Edge homophily is the share of edges between same-label nodes. With 24.6 % positives a random graph
already scores 0.63, so I also report the class-imbalance-adjusted version of Platonov et al., which
is zero for a random graph. The kNN graph on complete rows has an adjusted homophily of 0.43 (edge
homophily 0.785) and the graphs built from visible cells, which are what the encoder trains on, 0.28.
It is positive but inherited: neighbours share a label because they
share the features that predict it, and the row carries those features already. So homophily of a
feature-derived graph is not evidence that aggregation adds information; it would matter for
label propagation, where neighbours' *labels* are used, which is a different setting.

**11. Why GraphSAGE with a mean aggregator and two layers, and not GCN or GAT?**
The separate root weight lets the encoder ignore a useless neighbourhood and reduce to an MLP, which
keeps the comparison clean; GCN's symmetric normalisation with added self-loops mixes self and
neighbourhood and assumes an undirected graph, while the kNN graph is directed (a record picks its
neighbours; nobody picks a record back). Two layers give the same receptive field as the
propagation baselines, and deeper stacks smooth features more (Li et al., 2018). Attention would need
per-neighbour weights on ten neighbours of a low-dimensional record, which I did not explore.

**12. Why is the record-value graph "natural", and what does it lose?**
It mirrors the relational structure of a table without a distance: records that share a categorical
value are connected through that value. It loses the numeric columns as structure (they are only
node features), it has hub values (`sex` has two levels, so each value node pools a third or two
thirds of all records and its mean is nearly constant), and a hidden categorical cell is a missing
edge, which changes the degree of the record. In the results it is the only encoder that is ahead of
the MLP by more than the noise (+0.006 ± 0.003 probe AUC, five of five seeds), and that is where the
control matters: `sage_bip_nopool` removes the messages from records to value nodes and is ahead by
the same amount (+0.006 ± 0.002). The edge therefore cannot be attributed to records sharing values;
learned value embeddings and the larger encoder (three times the parameters of the MLP) are two
remaining explanations, but I did not run a capacity-matched MLP to distinguish them. I added that
control after seeing the edge in four seeds, and I
say so in the research note.

**13. Training uses corrupted inputs and noisy graphs, and evaluation uses clean ones. Is that a problem?**
It is a mismatch, and it is deliberate for the MLP too (denoising autoencoders are applied to clean
inputs). For graph encoders it also affects the neighbourhoods: during training the neighbours are
chosen from rows in which 30 % of the cells are random fill and their own features are 30 % hidden,
while at inference both the choice and the features are clean. I did not measure the size of this
effect. The `_nomask` variants
remove the corruption altogether and bound how much the objective matters: probe AUC is within noise of
the corrupted versions (paired differences -0.001 to +0.003, standard deviations 0.005 to 0.012), and
at 50 labels the uncorrupted encoders are nominally ahead in 16 of 20 seed-method pairs by about one
standard deviation, which I read as suggestive at most. What corruption changes is the ability to
impute hidden cells (categorical accuracy 0.70 against 0.43 to 0.51 without it).

**14. Where does the generative stage come in, and what does this repository tell you about it?**
The plan is to put a generative model (diffusion) on top of the representations, or on the
graph-structured records. This repository tests one prerequisite, whether a graph-based
representation is at least as good as a flat one for downstream use, and finds that it is level, not
better: the kNN encoder equals the MLP and the no-neighbour controls, and the record-value edge
persists when cross-record pooling is removed. It does not test
anything about generation: the decoder was trained to reconstruct only, and a latent
diffusion model needs a smooth, well-covered latent space and an evaluation of samples (marginals,
correlations, utility, memorisation) that I have not built. If graph encoders give no advantage
here, the graph might still matter for generation where relations between records, and not the
records themselves, are what has to be generated.

**15. What would have falsified your hypotheses, and did you tune or change anything after seeing results?**
The criteria are in section 7 of the research note, written before the five-seed run. H1 needs a paired
difference to the MLP that is larger than its standard deviation across seeds, with the construction
ahead in at least four of five seeds, on probe AUC, retrieval AUC and 50-label AUC. H2 fails if the
random or self-loop graph is within the noise of kNN. H3 fails if the gap is not larger with 50
labels. H4 fails if a graph with clearly positive adjusted homophily does not help. Applied to the
results: H1 is not supported for the kNN graph (it fails on all three metrics) and is met, narrowly, for
the record-value graph; H2 is not supported for either (for kNN the controls are within noise of it, for
the record-value graph `sage_bip_nopool` matches it); the H3 criterion is not triggered but says nothing,
because it compares two means and ignores the noise; H4 is not supported for the kNN graph (adjusted
homophily 0.43 and no gain).

What I changed after seeing results is listed in section 9 of the research note: I made the wording
of the H1 criterion explicit (I had not defined "clearly" or "on all three metrics"; the only verdict
that depends on it is the record-value one, and only if "clearly" means two standard deviations), and I
added four controls, three after seed 0 and one after four seeds. Tuning: none of the learning rate,
mask rate, code size, k or price was tuned. Hidden width, number of views and step count were set with
compute in mind and the step count from validation reconstruction loss on seed 0; that is disclosed in
the config and the research note.
