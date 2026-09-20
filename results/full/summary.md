Run `full`; random seeds: 5. Cells are mean ± standard deviation across seeds (the standard deviation is omitted for a single seed). The splits overlap, so these spreads are not confidence intervals.

### Frozen-embedding evaluation (test rows)

| method | dim | probe AUC | probe AP | probe acc | kNN AUC | kNN pos. P@10 |
|---|---:|---:|---:|---:|---:|---:|
| raw | 53 | 0.895 ± 0.002 | 0.737 ± 0.006 | 0.838 ± 0.003 | 0.891 ± 0.003 | 0.582 ± 0.004 |
| pca | 16 | 0.886 ± 0.002 | 0.718 ± 0.006 | 0.829 ± 0.004 | 0.887 ± 0.002 | 0.568 ± 0.005 |
| prop_knn | 16 | 0.886 ± 0.002 | 0.719 ± 0.007 | 0.829 ± 0.004 | 0.885 ± 0.002 | 0.569 ± 0.005 |
| prop_rand | 16 | 0.886 ± 0.002 | 0.718 ± 0.006 | 0.830 ± 0.004 | 0.885 ± 0.001 | 0.566 ± 0.005 |
| prop_bip | 16 | 0.887 ± 0.002 | 0.720 ± 0.006 | 0.830 ± 0.004 | 0.887 ± 0.002 | 0.567 ± 0.005 |
| mlp_ae | 16 | 0.881 ± 0.003 | 0.708 ± 0.015 | 0.831 ± 0.004 | 0.884 ± 0.003 | 0.568 ± 0.006 |
| sage_selfloop | 16 | 0.884 ± 0.006 | 0.713 ± 0.020 | 0.831 ± 0.008 | 0.884 ± 0.003 | 0.570 ± 0.003 |
| sage_rand | 16 | 0.882 ± 0.005 | 0.709 ± 0.020 | 0.831 ± 0.006 | 0.885 ± 0.003 | 0.569 ± 0.005 |
| sage_knn | 16 | 0.880 ± 0.007 | 0.702 ± 0.015 | 0.829 ± 0.006 | 0.885 ± 0.003 | 0.570 ± 0.008 |
| sage_bip | 16 | 0.887 ± 0.002 | 0.724 ± 0.006 | 0.833 ± 0.004 | 0.885 ± 0.003 | 0.573 ± 0.005 |
| sage_bip_nopool | 16 | 0.887 ± 0.003 | 0.721 ± 0.009 | 0.833 ± 0.003 | 0.885 ± 0.003 | 0.573 ± 0.007 |
| sage_knn_k5 | 16 | 0.878 ± 0.004 | 0.706 ± 0.007 | 0.830 ± 0.004 | 0.884 ± 0.003 | 0.569 ± 0.006 |
| sage_knn_k20 | 16 | 0.881 ± 0.004 | 0.710 ± 0.012 | 0.830 ± 0.005 | 0.884 ± 0.003 | 0.569 ± 0.005 |
| mlp_ae_nomask | 16 | 0.882 ± 0.003 | 0.709 ± 0.006 | 0.829 ± 0.002 | 0.885 ± 0.004 | 0.572 ± 0.006 |
| sage_selfloop_nomask | 16 | 0.883 ± 0.004 | 0.715 ± 0.013 | 0.830 ± 0.004 | 0.886 ± 0.002 | 0.572 ± 0.002 |
| sage_rand_nomask | 16 | 0.880 ± 0.009 | 0.706 ± 0.024 | 0.830 ± 0.006 | 0.886 ± 0.004 | 0.573 ± 0.004 |
| sage_knn_nomask | 16 | 0.883 ± 0.006 | 0.710 ± 0.017 | 0.830 ± 0.006 | 0.887 ± 0.002 | 0.573 ± 0.003 |
| mlp_ae_lr2x | 16 | 0.882 ± 0.003 | 0.707 ± 0.010 | 0.831 ± 0.003 | 0.886 ± 0.003 | 0.572 ± 0.005 |
| sage_knn_cleannbr | 16 | 0.881 ± 0.006 | 0.708 ± 0.014 | 0.829 ± 0.006 | 0.884 ± 0.004 | 0.568 ± 0.004 |
| hgb_ref | n/a | 0.927 ± 0.002 | 0.830 ± 0.005 | 0.867 ± 0.003 | n/a | n/a |

`hgb_ref` is a supervised gradient-boosting reference on the flat features, not a representation.

### Label efficiency (test AUC of a probe fitted on n labelled training rows)

| method | n=50 | n=200 | n=1000 |
|---|---:|---:|---:|
| raw | 0.801 ± 0.014 | 0.835 ± 0.001 | 0.881 ± 0.002 |
| pca | 0.801 ± 0.011 | 0.856 ± 0.004 | 0.881 ± 0.002 |
| prop_knn | 0.803 ± 0.011 | 0.859 ± 0.002 | 0.881 ± 0.002 |
| prop_rand | 0.798 ± 0.009 | 0.856 ± 0.003 | 0.881 ± 0.001 |
| prop_bip | 0.803 ± 0.011 | 0.857 ± 0.004 | 0.882 ± 0.002 |
| mlp_ae | 0.808 ± 0.017 | 0.852 ± 0.004 | 0.876 ± 0.003 |
| sage_selfloop | 0.812 ± 0.014 | 0.857 ± 0.006 | 0.879 ± 0.006 |
| sage_rand | 0.806 ± 0.014 | 0.854 ± 0.004 | 0.877 ± 0.006 |
| sage_knn | 0.812 ± 0.007 | 0.854 ± 0.008 | 0.874 ± 0.006 |
| sage_bip | 0.819 ± 0.009 | 0.861 ± 0.002 | 0.882 ± 0.002 |
| sage_bip_nopool | 0.820 ± 0.015 | 0.861 ± 0.003 | 0.882 ± 0.003 |
| sage_knn_k5 | 0.803 ± 0.012 | 0.850 ± 0.006 | 0.874 ± 0.004 |
| sage_knn_k20 | 0.809 ± 0.005 | 0.854 ± 0.005 | 0.877 ± 0.005 |
| mlp_ae_nomask | 0.818 ± 0.015 | 0.856 ± 0.004 | 0.877 ± 0.004 |
| sage_selfloop_nomask | 0.824 ± 0.014 | 0.859 ± 0.004 | 0.878 ± 0.004 |
| sage_rand_nomask | 0.821 ± 0.016 | 0.855 ± 0.010 | 0.875 ± 0.010 |
| sage_knn_nomask | 0.815 ± 0.016 | 0.857 ± 0.009 | 0.878 ± 0.006 |
| mlp_ae_lr2x | 0.815 ± 0.015 | 0.854 ± 0.004 | 0.877 ± 0.003 |
| sage_knn_cleannbr | 0.811 ± 0.010 | 0.854 ± 0.005 | 0.876 ± 0.006 |
| hgb_ref | n/a | n/a | n/a |

### Learned encoders: reconstruction of test rows

`clean` = all cells visible. `hidden` = only the 30 % hidden cells are scored, with the rest of the row (and, for graph encoders, a graph built from the visible cells only) as input.

| method | clean: numeric MSE | clean: categorical acc | hidden: numeric MSE | hidden: categorical acc | encoder params | best step |
|---|---:|---:|---:|---:|---:|---:|
| mlp_ae | 0.009 ± 0.003 | 0.995 ± 0.001 | 0.829 ± 0.022 | 0.704 ± 0.003 | 4816 | 1970 ± 27 |
| sage_selfloop | 0.008 ± 0.001 | 0.996 ± 0.001 | 0.827 ± 0.024 | 0.705 ± 0.003 | 9552 | 1980 ± 45 |
| sage_rand | 0.011 ± 0.003 | 0.994 ± 0.002 | 0.844 ± 0.021 | 0.701 ± 0.003 | 9552 | 1950 ± 87 |
| sage_knn | 0.010 ± 0.003 | 0.995 ± 0.002 | 0.838 ± 0.024 | 0.701 ± 0.005 | 9552 | 1940 ± 65 |
| sage_bip | 0.006 ± 0.001 | 0.994 ± 0.002 | 0.828 ± 0.022 | 0.700 ± 0.004 | 14736 | 1970 ± 45 |
| sage_bip_nopool | 0.006 ± 0.001 | 0.994 ± 0.002 | 0.827 ± 0.024 | 0.702 ± 0.001 | 14736 | 1930 ± 57 |
| sage_knn_k5 | 0.009 ± 0.002 | 0.995 ± 0.001 | 0.835 ± 0.024 | 0.701 ± 0.005 | 9552 | 1950 ± 35 |
| sage_knn_k20 | 0.010 ± 0.001 | 0.995 ± 0.002 | 0.840 ± 0.027 | 0.702 ± 0.003 | 9552 | 1970 ± 45 |
| mlp_ae_nomask | 0.001 ± 0.000 | 0.999 ± 0.000 | 1.026 ± 0.032 | 0.438 ± 0.030 | 4816 | 1960 ± 42 |
| sage_selfloop_nomask | 0.001 ± 0.000 | 0.999 ± 0.000 | 1.013 ± 0.036 | 0.465 ± 0.026 | 9552 | 2000 |
| sage_rand_nomask | 0.001 ± 0.000 | 0.999 ± 0.000 | 1.020 ± 0.032 | 0.433 ± 0.042 | 9552 | 1940 ± 42 |
| sage_knn_nomask | 0.001 ± 0.000 | 0.999 ± 0.000 | 1.092 ± 0.031 | 0.507 ± 0.021 | 9552 | 1970 ± 45 |
| mlp_ae_lr2x | 0.009 ± 0.002 | 0.998 ± 0.000 | 0.826 ± 0.024 | 0.707 ± 0.003 | 4816 | 1920 ± 104 |
| sage_knn_cleannbr | 0.035 ± 0.004 | 0.992 ± 0.001 | 0.038 ± 0.003 | 0.959 ± 0.002 | 9552 | 1980 ± 45 |

For scale, predicting every hidden test cell by the training column mean (numeric) or mode (categorical) gives numeric MSE 0.992 and categorical accuracy 0.574.

Training time: wall-clock seconds of the full-batch steps on the CPU, without graph construction. Median and range over seeds. The times depend on what else the machine was doing, so only large ratios mean anything.

| method | median s | range s |
|---|---:|---:|
| mlp_ae | 101 | 99 to 363 |
| sage_selfloop | 131 | 126 to 232 |
| sage_rand | 175 | 159 to 179 |
| sage_knn | 188 | 175 to 667 |
| sage_bip | 175 | 174 to 217 |
| sage_bip_nopool | 165 | 160 to 171 |
| sage_knn_k5 | 153 | 146 to 156 |
| sage_knn_k20 | 214 | 211 to 226 |
| mlp_ae_nomask | 108 | 105 to 118 |
| sage_selfloop_nomask | 149 | 137 to 160 |
| sage_rand_nomask | 185 | 179 to 205 |
| sage_knn_nomask | 177 | 176 to 203 |
| mlp_ae_lr2x | 103 | 96 to 123 |
| sage_knn_cleannbr | 174 | 170 to 184 |

### Paired differences vs `mlp_ae` (per seed, same split)

Mean ± standard deviation across seeds of the per-seed difference; wins = seeds in which the method is ahead.

| method | Δ probe AUC | wins | Δ kNN AUC | wins | Δ lowlabel_auc@50 | wins |
|---|---:|---:|---:|---:|---:|---:|
| raw | +0.015 ± 0.003 | 5/5 | +0.007 ± 0.001 | 5/5 | -0.007 ± 0.015 | 2/5 |
| pca | +0.005 ± 0.003 | 5/5 | +0.003 ± 0.002 | 5/5 | -0.007 ± 0.013 | 1/5 |
| prop_knn | +0.005 ± 0.003 | 5/5 | +0.001 ± 0.002 | 3/5 | -0.005 ± 0.013 | 2/5 |
| prop_rand | +0.005 ± 0.003 | 5/5 | +0.002 ± 0.002 | 4/5 | -0.010 ± 0.012 | 1/5 |
| prop_bip | +0.006 ± 0.003 | 5/5 | +0.003 ± 0.002 | 5/5 | -0.005 ± 0.012 | 2/5 |
| sage_selfloop | +0.003 ± 0.005 | 3/5 | -0.000 ± 0.002 | 3/5 | +0.004 ± 0.016 | 3/5 |
| sage_rand | +0.001 ± 0.004 | 3/5 | +0.002 ± 0.002 | 4/5 | -0.002 ± 0.018 | 2/5 |
| sage_knn | -0.001 ± 0.009 | 3/5 | +0.001 ± 0.002 | 4/5 | +0.004 ± 0.019 | 3/5 |
| sage_bip | +0.006 ± 0.003 | 5/5 | +0.001 ± 0.001 | 4/5 | +0.011 ± 0.010 | 4/5 |
| sage_bip_nopool | +0.006 ± 0.002 | 5/5 | +0.001 ± 0.001 | 5/5 | +0.012 ± 0.010 | 4/5 |
| sage_knn_k5 | -0.002 ± 0.005 | 3/5 | -0.000 ± 0.002 | 3/5 | -0.005 ± 0.018 | 3/5 |
| sage_knn_k20 | +0.001 ± 0.004 | 2/5 | -0.000 ± 0.002 | 3/5 | +0.001 ± 0.022 | 2/5 |
| mlp_ae_nomask | +0.001 ± 0.005 | 3/5 | +0.001 ± 0.002 | 4/5 | +0.010 ± 0.017 | 4/5 |
| sage_selfloop_nomask | +0.002 ± 0.002 | 5/5 | +0.002 ± 0.002 | 4/5 | +0.016 ± 0.015 | 4/5 |
| sage_rand_nomask | -0.000 ± 0.008 | 3/5 | +0.002 ± 0.002 | 4/5 | +0.013 ± 0.009 | 4/5 |
| sage_knn_nomask | +0.002 ± 0.004 | 3/5 | +0.003 ± 0.002 | 4/5 | +0.007 ± 0.020 | 3/5 |
| mlp_ae_lr2x | +0.001 ± 0.004 | 3/5 | +0.002 ± 0.002 | 5/5 | +0.007 ± 0.017 | 4/5 |
| sage_knn_cleannbr | -0.000 ± 0.008 | 3/5 | +0.000 ± 0.001 | 2/5 | +0.003 ± 0.018 | 2/5 |
| hgb_ref | +0.046 ± 0.004 | 5/5 | n/a | n/a | n/a | n/a |

### Paired differences vs `sage_selfloop` (same layers and parameters, no neighbours)

Mean ± standard deviation across seeds of the per-seed difference; wins = seeds in which the method is ahead.

| method | Δ probe AUC | wins | Δ kNN AUC | wins | Δ lowlabel_auc@50 | wins |
|---|---:|---:|---:|---:|---:|---:|
| mlp_ae | -0.003 ± 0.005 | 2/5 | +0.000 ± 0.002 | 2/5 | -0.004 ± 0.016 | 2/5 |
| sage_rand | -0.002 ± 0.004 | 2/5 | +0.002 ± 0.001 | 4/5 | -0.006 ± 0.009 | 1/5 |
| sage_knn | -0.004 ± 0.010 | 2/5 | +0.001 ± 0.001 | 4/5 | -0.000 ± 0.013 | 2/5 |
| sage_bip | +0.003 ± 0.007 | 4/5 | +0.001 ± 0.002 | 4/5 | +0.007 ± 0.011 | 3/5 |
| sage_bip_nopool | +0.003 ± 0.004 | 4/5 | +0.001 ± 0.001 | 4/5 | +0.008 ± 0.010 | 4/5 |
| sage_knn_k5 | -0.005 ± 0.007 | 1/5 | +0.000 ± 0.001 | 3/5 | -0.009 ± 0.011 | 1/5 |
| sage_knn_k20 | -0.002 ± 0.006 | 2/5 | -0.000 ± 0.001 | 2/5 | -0.003 ± 0.015 | 2/5 |
| mlp_ae_nomask | -0.002 ± 0.009 | 2/5 | +0.001 ± 0.002 | 3/5 | +0.006 ± 0.023 | 3/5 |
| sage_selfloop_nomask | -0.001 ± 0.005 | 2/5 | +0.002 ± 0.002 | 4/5 | +0.012 ± 0.010 | 5/5 |
| sage_rand_nomask | -0.003 ± 0.010 | 2/5 | +0.002 ± 0.001 | 4/5 | +0.009 ± 0.011 | 4/5 |
| sage_knn_nomask | -0.001 ± 0.002 | 1/5 | +0.003 ± 0.002 | 4/5 | +0.003 ± 0.010 | 3/5 |
| mlp_ae_lr2x | -0.002 ± 0.004 | 1/5 | +0.002 ± 0.001 | 5/5 | +0.003 ± 0.007 | 3/5 |
| sage_knn_cleannbr | -0.003 ± 0.009 | 3/5 | +0.000 ± 0.002 | 2/5 | -0.001 ± 0.006 | 2/5 |

### Graph diagnostics (training graph, mean over seeds)

| graph | edge homophily | adjusted homophily | never chosen | max times chosen | skew of times chosen | build s |
|---|---:|---:|---:|---:|---:|---:|
| knn_k10 | 0.785 | 0.428 | 0.016 | 35.6 | 0.43 | 35.6 |
| knn_k20 | 0.782 | 0.423 | 0.004 | 66.4 | 0.46 | 35.6 |
| knn_k5 | 0.787 | 0.430 | 0.048 | 20.4 | 0.49 | 35.6 |
| knn_visible_k10_p0.3 | 0.727 | 0.280 | 0.018 | 42.3 | 0.66 | 164.3 |
| knn_visible_k20_p0.3 | 0.724 | 0.273 | 0.004 | 80.8 | 0.71 | 164.3 |
| knn_visible_k5_p0.3 | 0.731 | 0.286 | 0.057 | 22.5 | 0.66 | 164.3 |
| random_k10 | 0.630 | 0.001 | 0.000 | 24.8 | 0.34 | n/a |

Adjusted homophily is 0 for a graph that ignores the label and 1 for one that only links equal labels (Platonov et al., 2023). `knn_k*` graphs are built from complete rows; `knn_visible_*` graphs are those of the corrupted views on which the kNN encoders train (mean over the views). One neighbour search serves every k, so its build time is shared. *never chosen* is the share of training rows that no row picks as a neighbour.

Record-value graph: averaged over columns and training rows, 0.661 of the other records that share a categorical value with a record have the same label, against 0.629 expected by chance (per column: `graph_diagnostics.csv`).
