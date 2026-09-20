# Data

Nothing in this folder is committed except this file. `make data` (or
`python scripts/download_data.py`) downloads the raw files into `data/raw/adult/`, which is
listed in `.gitignore`.

## UCI Adult (Census Income)

| | |
|---|---|
| Source | UCI Machine Learning Repository, dataset 2, <https://archive.ics.uci.edu/dataset/2/adult> |
| Downloaded from | <https://archive.ics.uci.edu/static/public/2/adult.zip> (the "Download" link of the dataset page) |
| Citation (as given on the dataset page) | Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5XW20> |
| Licence | CC BY 4.0 (stated on the dataset page) |
| Access | Public, no registration, no API key |
| Size | 48,842 rows in two files (`adult.data`: 32,561, `adult.test`: 16,281), 14 attributes + label; the zip is about 620 KB |
| Task in the original data | predict whether yearly income exceeds 50K USD from 1994 US Census attributes |

The zip also contains `adult.names` (the attribute description), `old.adult.names` and a tiny
`Index` file; only `adult.data` and `adult.test` are used (`adult.names` is kept next to them). `adult.test` starts with a `|1x3 Cross validator`
line and its labels end in a period (`>50K.`); both quirks are handled in
`graphmix.data.read_adult_file`. The original train/test split is **not** used: the two files
are concatenated and every seed draws its own stratified 60/20/20 split.

`download_adult` checks the row counts (32,561 and 16,281) and prints SHA-256 digests. These are
the digests recorded for the files used by this project; they are convenience identifiers, not
official UCI checksums.

```
adult.data   5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d
adult.test   a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05
```

## What the cleaning step does

Numbers below are from `python scripts/download_data.py --summary`.

* `fnlwgt` (a survey sampling weight) and `education` (a string copy of `education-num`) are
  dropped. That leaves 5 numeric columns (`age`, `education-num`, `capital-gain`,
  `capital-loss`, `hours-per-week`) and 7 categorical columns (`workclass`, `marital-status`,
  `occupation`, `relationship`, `race`, `sex`, `native-country`).
* `?` (missing) in `workclass`, `occupation` and `native-country` becomes an explicit
  `Unknown` level (2,799 / 2,809 / 857 rows). This is a modelling choice, not an imputation;
  see `docs/design_decisions.md`.
* Exact duplicate rows are removed **before** any split: 6,374 of the 48,842 rows are
  duplicates once `fnlwgt` is gone, leaving 42,468 rows with 24.6 % positives. Without this,
  identical rows could land in both the training and the test split and a nearest-neighbour graph
  would connect them trivially.
* For each seeded partition, categorical levels rarer than 0.5 % of its *training* rows are merged
  into one `__rare__` level; that fitted mapping is then applied unchanged to validation and test.

## Sensitive attributes

`sex`, `race` and `native-country` are attributes of people, and the label is a proxy for
income. They are used as ordinary input columns because the experiment is about representation
learning on mixed-type records, not about deploying an income model. Nothing here should be read
as a statement about the people in the data, and no fairness analysis is attempted (a limitation,
listed in the README). The data is from 1994 and is not representative of any current
population.
