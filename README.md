# LEDR

Research code for **LEDR: LLM-assisted Context Enhancement with Domain-Aware Regularization for Multi-Domain Fake News Detection**.

This repository is a cleaned release of the supplied experimental code snapshot. The
main training/evaluation pipeline is retained with minimal runtime-safe cleanup. See
[`REPRODUCIBILITY_NOTES.md`](REPRODUCIBILITY_NOTES.md) for the exact mapping and items
that should be verified against the final experiment source before claiming full
end-to-end reproduction of every manuscript setting.

## Repository layout

```text
LEDR/
├── train.py                  # convenience entry point
├── mainCKD.py                # original experiment entry point
├── Combined_KD_m.py          # training / validation / testing loop
├── models/                   # LEDR snapshot + baseline models
├── utils/                    # data loading, metrics, contrastive utility
├── data/README.md            # expected dataset layout and columns
├── pretrained_model/README.md
├── prompts/README.md         # LLM-generation release note
├── legacy/SNE.py             # non-main legacy script; not runnable as supplied
├── requirements.txt
└── REPRODUCIBILITY_NOTES.md
```

## Installation

Python 3.9+ is recommended. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

A CUDA-capable PyTorch installation is expected by the original training configuration.
Install the PyTorch build appropriate for your CUDA environment if needed.

## Pretrained encoders

Place the pretrained encoders at:

```text
pretrained_model/chinese-bert-wwm-ext/
pretrained_model/roberta-base/
```

See [`pretrained_model/README.md`](pretrained_model/README.md).

## Data

Raw datasets are not redistributed. Prepare the processed split files as described in
[`data/README.md`](data/README.md). The loader expects `train.pkl`, `val.pkl`, and
`test.pkl`, including an `expert_comment` column containing pre-generated LLM background
knowledge.

## Run

Chinese benchmark:

```bash
python train.py --dataset ch1 --domain_num 9 --gpu 0
```

English benchmark:

```bash
python train.py --dataset en --domain_num 3 --gpu 0
```

The original defaults include batch size 64, 50 epochs, learning rate `1e-4`, and seed
2023. Check `mainCKD.py` for all command-line options and configuration values.

## Evaluation

`utils/utils.py::metrics` reports overall macro-F1 together with domain-wise FNR/FPR and
FNED/FPED. Model selection in the supplied trainer uses validation F1 via `Recorder`.

## Outputs

Training checkpoints are written under `param_model/`; test-time model snapshots are
written under `recodertestpkl/`. These generated files are ignored by Git.

## Code/data availability

The repository is intended to host source code and reproduction materials. Third-party
raw datasets and pretrained model weights are not included; obtain them from their
original providers and follow the applicable licenses and access conditions.

## Security

Do not commit API keys, passwords, private dataset credentials, or local tokens. If a
DeepSeek generation script is later added, read the API key from an environment variable
(e.g. `DEEPSEEK_API_KEY`) rather than hard-coding it.
