# SSKA-Rec

Self-supervised auxiliary knowledge and selective adaptation for expandable multi-task recommendation.

## Environment

Python 3.10+, PyTorch 2.1+, NumPy <2. The cloud experiments used Python 3.11.5, Linux, PyTorch 2.1.1, CUDA 11.8 and NVIDIA RTX 4090 D (12 GB visible GPU memory); use about 32 GB host RAM. CPU is supported for tests. A matching Conda setup is provided in `environment.yml`.

```bash
python -m venv .venv
source .venv/bin/activate
# Install the PyTorch build matching your CUDA environment, then:
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Data

Set `--dataset-root` to an existing shared dataset directory. Raw data and outputs are not included.

| Dataset | Source | Required files under dataset root |
|---|---|---|
| Census-income | [UCI original](https://archive.ics.uci.edu/dataset/117/census+income+kdd), [MPT-Rec preprocessing](https://github.com/BAI-LAB/MPT-Rec/blob/two_task/multitaskrec/data_preprocessing/CensusIncome_process.py) | `Census-income/train.gz`, `test.gz` (encoded CSV with header) |
| Ali-CCP | [Official data](https://tianchi.aliyun.com/dataset/408), [MPT-Rec preprocessing](https://github.com/BAI-LAB/MPT-Rec/blob/two_task/multitaskrec/data_preprocessing/AliCCP_process.py) | `AliCCP/ctr_cvr.train`, `ctr_cvr.dev`, `ctr_cvr.test` (encoded CSV with header) |
| KuaiRand-Pure | [Official page](https://kuairand.com/), [Download](https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz) | Three `log_*_pure.csv` files in `KuaiRand-Pure/data/` |

Download and extract KuaiRand once:

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
echo "0820331067a3784d9691136f772b35a7  KuaiRand-Pure.tar.gz" | md5sum -c -
tar -xzf KuaiRand-Pure.tar.gz -C /path/to/shared/dataset
```

Census and Ali-CCP require the MPT-Rec encoded format, not the raw source files. Dataset details and expected Census hashes: [DATASETS.md](DATASETS.md).

## Training

```bash
python train.py --config configs/census.json --dataset-root /path/to/shared/dataset --output outputs/census
python train.py --config configs/aliccp.json --dataset-root /path/to/shared/dataset --output outputs/aliccp
python train.py --config configs/kuairand.json --dataset-root /path/to/shared/dataset --cache-root /path/to/shared/cache --output outputs/kuairand
```

Default: five paired seeds, MPT-Rec (`no_u`) and SSKA-Rec (`sska`). Optional: `--models no_u random_u fixed_fusion sska single_task`, `--seeds 1685480945`. Validation AUC selects checkpoints, with patience 5. KuaiRand Single Task uses the corrected dimension-normalized embedding L2.

Census: Income/Marital → Education → Sex. KuaiRand: Click/Like → Follow → Forward. Ali-CCP: CTR/CVR → BSI. All four Census target fields are removed from inputs from the start. KuaiRand uses all exposures and splits whole local days 70/15/15; labels and post-exposure feedback are excluded from inputs.

Each seed shares one frozen backbone and one frozen SSL source. New tasks receive private trainable U copies; historical task copies stay fixed. Outputs include checkpoint weights, validation curves, raw predictions, metrics, source/data identity and costs. Re-running the same command reuses committed stages; an interrupted stage restarts from its seed.

## Experiments

- [Census原任务U接入方式](experiments/u-access-census/PLAN.md)
- [Census二阶段融合与U更新范围](experiments/u-stage2-census/README.md)

## Source

Backbone and base training adapted from [MPT-Rec](https://github.com/BAI-LAB/MPT-Rec), with local reproduction fixes. SSKA modules are in `sska_rec/expert.py` and `sska_rec/core.py`; the training pipeline is in `sska_rec/training.py`. This is a standalone source release, not a redistribution of experimental artifacts. Upstream attribution: [NOTICE](NOTICE).
