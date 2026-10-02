# LEDR

Research code for **LLM-assisted Context Enhancement with Domain-Aware Regularization for Multi-Domain Fake News Detection (LEDR)**.

This repository contains the supplied experimental code snapshot together with documentation for data preparation, LLM-assisted context generation, training, and evaluation. Third-party raw datasets, pretrained model weights, and API credentials are not redistributed.

> **Reproducibility note.** The public code snapshot is preserved rather than silently rewritten to make it look identical to settings that are not present in the supplied source. See [`REPRODUCIBILITY_NOTES.md`](REPRODUCIBILITY_NOTES.md) before claiming exact reproduction of every manuscript setting.

## 1. Repository structure

```text
LEDR/
├── train.py                  # convenience entry point
├── mainCKD.py                # experiment configuration / data loading
├── Combined_KD_m.py          # training, validation and testing loop
├── models/                   # LEDR snapshot and baseline model components
├── utils/                    # dataloader, metrics and contrastive-learning utility
├── data/README.md            # dataset sources, expected files and preprocessing
├── prompts/README.md         # documented LLM-generation workflow
├── pretrained_model/README.md
├── requirements.txt
├── REPRODUCIBILITY_NOTES.md
└── CHANGELOG.md
```

## 2. Environment

The base software environment follows the public implementation of **DTDBD (ICDE 2024)**:

- Python 3.8
- PyTorch > 1.0
- pandas
- NumPy
- tqdm
- Transformers

The supplied LEDR snapshot additionally imports `scikit-learn` and `matplotlib`, so they are included in `requirements.txt`.

Create an environment and install the dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

The original experiment configuration is CUDA-oriented and expects a CUDA-capable PyTorch installation.

DTDBD environment reference: https://github.com/ningljy/DTDBD

## 3. Pretrained text encoders

Place the pretrained encoders under:

```text
pretrained_model/chinese-bert-wwm-ext/
pretrained_model/roberta-base/
```

The code loads:

- Chinese benchmark: `hfl/chinese-bert-wwm-ext`
- English benchmark: `roberta-base`

See [`pretrained_model/README.md`](pretrained_model/README.md).

## 4. Datasets

The training code expects already prepared split files and **does not download or redistribute the third-party raw datasets**.

Expected paths:

```text
data/
├── ch1/
│   ├── train.pkl
│   ├── val.pkl
│   └── test.pkl
└── en/
    ├── train.pkl
    ├── val.pkl
    └── test.pkl
```

The manuscript uses an 8:1:1 train/validation/test protocol. The current code consumes the split files directly; it does not create the split at runtime. For exact numerical reproduction, use the same split files as the reported experiments.

Dataset sources and the expected dataframe schema are documented in [`data/README.md`](data/README.md).

## 5. LLM-assisted context generation

Each sample is expected to contain a pre-generated LLM field named `expert_comment`.

The documented preparation workflow is:

1. Start from the original processed sample and retain its original split membership.
2. Build the generation prompt using the sample **content** and its **domain/category**.
3. Ask the LLM to generate the requested background/context text in the prompt-defined format. The ground-truth fake/real label is not supplied to the LLM.
4. Save the generated response as `expert_comment` for that same sample.
5. Preserve the remaining fields and export the augmented dataframe in the same `train.pkl` / `val.pkl` / `test.pkl` structure expected by the loader.
6. Complete and freeze LLM generation before model training.

The supplied code archive did not contain the exact original prompt-generation script or exact prompt wording; therefore this repository does not invent one. The available workflow and manuscript-reported generation settings are documented in [`prompts/README.md`](prompts/README.md).

## 6. Expected dataframe fields

Each split is expected to contain:

- `content`: post/news text
- `comments`: associated comment/context text
- `expert_comment`: LLM-generated auxiliary/background text
- `content_emotion`: numeric feature vector
- `comments_emotion`: numeric feature vector
- `emotion_gap`: numeric feature vector
- `style_feature`: numeric feature vector
- `label`: binary authenticity label
- `category`: domain label

See [`data/README.md`](data/README.md) for details.

## 7. Preprocessing performed by this code

For the prepared split files, `utils/dataloader.py` performs the following runtime processing:

- filters samples to the configured domain/category set;
- maps domain strings to integer domain IDs;
- tokenizes `content`, `comments`, and `expert_comment`;
- uses a maximum sequence length of 170 by default;
- applies truncation and max-length padding;
- converts emotion/style feature arrays to `float32` tensors;
- converts labels and domain IDs to tensors;
- shuffles the training loader but not validation/test loaders.

Any earlier collection, merging, cleaning, feature extraction, or construction of the released benchmark files is not implemented in this code snapshot and should be taken from the original dataset/benchmark provider or the exact experiment preparation records.

## 8. Run

Chinese benchmark (9 domains):

```bash
python train.py --dataset ch1 --domain_num 9 --gpu 0
```

English benchmark (3 domains):

```bash
python train.py --dataset en --domain_num 3 --gpu 0
```

`train.py` forwards arguments to `mainCKD.py`. Important defaults in the supplied snapshot include:

- batch size: 64
- epochs: 50
- learning rate: `1e-4`
- seed: 2023
- maximum sequence length: 170
- early stopping patience: 5

Check `mainCKD.py` for the full configuration used by this code snapshot.

## 9. Evaluation

`utils/utils.py::metrics` reports:

- overall macro-F1;
- overall AUC;
- domain-wise F1/AUC;
- domain-wise FNR/FPR;
- FNED;
- FPED.

The supplied trainer selects checkpoints using validation macro-F1 and then evaluates the selected checkpoint on the test split.

Generated checkpoints/results are written under `param_model/`, `recodertestpkl/`, and local result files; these are ignored by Git.

## 10. Third-party data sources

### Chinese benchmark: Weibo21

Official MDFEND/Weibo21 repository (the source associated with the cited MDFEND paper):

https://github.com/kennqiang/MDFEND-Weibo21

The official repository provides split data and states that access to the original Weibo21 dataset requires an application.

### English three-domain benchmark

The official DITFEND repository associated with the cited COLING 2022 work states that its English three-domain experimental dataset is constructed from **FakeNewsNet** and **MM-COVID**:

https://github.com/ICTMCG/DITFEND

Upstream sources:

- FakeNewsNet: https://github.com/KaiDMML/FakeNewsNet
- MM-COVID: https://github.com/bigheiniu/MM-COVID

Users should follow the original providers' licenses, access conditions, and platform policies. This repository does not re-host restricted third-party raw data.

## 11. Citation / related resources

If using the datasets or benchmark construction, cite the corresponding original papers and repositories, including MDFEND/Weibo21, FakeNewsNet, MM-COVID, and DITFEND as applicable.

The environment description above follows the public DTDBD implementation:

Jiayang Li, Xuan Feng, Tianlong Gu, and Liang Chang, *Dual-Teacher De-Biasing Distillation Framework for Multi-Domain Fake News Detection*, ICDE 2024.

## 12. Security and credentials

Do **not** commit API keys, access tokens, passwords, private dataset credentials, or pretrained weights. If an LLM-generation script is added later, credentials should be read from environment variables rather than hard-coded into source files.
