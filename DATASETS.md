# Dataset format

Reuse an existing shared dataset root; do not duplicate datasets for each run.

## Census-income

Download `census-income.data` and `census-income.test` from [UCI](https://archive.ics.uci.edu/dataset/117/census+income+kdd). Use [MPT-Rec's preprocessing script](https://github.com/BAI-LAB/MPT-Rec/blob/two_task/multitaskrec/data_preprocessing/CensusIncome_process.py) to produce `train.gz` and `test.gz`. UCI raw files cannot be substituted directly. Required labels: `label_income`, `label_marital`, integer `education`, binary `sex`; Education is `education == 9`. Keep the encoding used by MPT-Rec and its vocabulary. The published experiments use the existing encoded files identified below; re-encoding creates a distinct data version if hashes differ.

Expected SHA256 of the experiment input files:

```text
e2f2d57925a8020d23fcd05802284b3bc1d91fcdd451f1eaaefd683e76d1b2dc  Census-income/train.gz
8de690375028ac531de37fcbb117bd6527c5f3542d3bfead8975226e45100cec  Census-income/test.gz
```

The original test file is split equally into validation/test using split seed 20260929. All four target fields are excluded from features.

## Ali-CCP

Official source: https://tianchi.aliyun.com/dataset/408 . Download the training/test `sample_skeleton_*.csv` and `common_features_*.csv` files. Follow [MPT-Rec's AliCCP_process.py](https://github.com/BAI-LAB/MPT-Rec/blob/two_task/multitaskrec/data_preprocessing/AliCCP_process.py) for `ctr_cvr.train`, `ctr_cvr.dev`, `ctr_cvr.test`. Columns must retain its encoded layout: the first two are CTR/CVR; feature columns are `columns[3:-1]`; the final column is 301 and BSI is `301 != 2`. Feature 101 is excluded. Categorical cardinalities are defined in `sska_rec/vocabulary.py`.

The released protocol reads the first 10M training, 1M validation and 10M test rows. Raw source files with another schema are not directly compatible.

## KuaiRand-Pure

Official instructions: https://kuairand.com/ . Download: https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz . Archive MD5: `0820331067a3784d9691136f772b35a7`.

Required files: `log_standard_4_08_to_4_21_pure.csv`, `log_standard_4_22_to_5_08_pure.csv`, `log_random_4_22_to_5_08_pure.csv`. SHA256 checks are in `configs/kuairand.json`.

Features: user_id, video_id, tab, hour (hourmin // 100), duration_ms. Fit categorical encoding on training only. Split by complete days derived from time_ms in Asia/Shanghai, preserving all exposures. The raw date column is audited but does not determine the split. Do not filter clicks, rebalance negatives, or feed behavioral labels into features.
