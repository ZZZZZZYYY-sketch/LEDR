# Data preparation

The code expects preprocessed `train.pkl`, `val.pkl`, and `test.pkl` files. Raw third-party datasets are **not redistributed** in this repository. Obtain data from the original providers and comply with their licenses, access conditions, copyright restrictions, and platform policies.

## 1. Expected locations

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

The experiments use an 8:1:1 train/validation/test protocol. The training code reads the prepared split files directly and does not create a new split at runtime. Use the same provider/released benchmark splits when reproducing the reported experiments.

## 2. Original / upstream sources

### Chinese benchmark: Weibo21

Paper used in the manuscript:

- Q. Nan et al., "MDFEND: Multi-domain Fake News Detection," CIKM 2021.

Official MDFEND/Weibo21 repository:

- https://github.com/kennqiang/MDFEND-Weibo21

The official repository states that the split dataset (`train`, `val`, `test`) is provided in its `data` directory and that access to the original Weibo21 dataset requires submission of an application for use.

### English benchmark: GossipCop, PolitiFact, COVID

The manuscript follows the English three-domain benchmark used by the cited multi-domain studies. The official repository associated with Nan et al. (COLING 2022, DITFEND) states that its English three-domain experimental dataset is constructed from **FakeNewsNet** and **MM-COVID**:

- DITFEND (official repository): https://github.com/ICTMCG/DITFEND
- FakeNewsNet (GossipCop and PolitiFact upstream source): https://github.com/KaiDMML/FakeNewsNet
- MM-COVID (COVID upstream source): https://github.com/bigheiniu/MM-COVID

The FakeNewsNet repository explains that a complete copy cannot be redistributed because of platform privacy policies and publisher copyright restrictions; it provides a minimal dataset and collection tooling. MM-COVID likewise provides its data through its documented repository/data links and notes platform restrictions for social-media content. For these reasons this repository points users to the original providers rather than repackaging the raw third-party data.

## 3. Expected dataframe columns

`utils/dataloader.py` expects each split to contain:

| Column | Meaning |
|---|---|
| `content` | post/news text |
| `comments` | associated comment/context text |
| `expert_comment` | pre-generated LLM auxiliary/background text |
| `content_emotion` | numeric feature vector |
| `comments_emotion` | numeric feature vector |
| `emotion_gap` | numeric feature vector |
| `style_feature` | numeric feature vector |
| `label` | binary authenticity label |
| `category` | domain label string |

## 4. Domain labels used by the loader

Chinese (`ch1`, 9 domains):

`科技`, `军事`, `教育考试`, `灾难事故`, `政治`, `医药健康`, `财经商业`, `文体娱乐`, `社会生活`

English (`en`, 3 domains):

`gossipcop`, `politifact`, `COVID`

## 5. Runtime preprocessing in this repository

For already prepared split files, the loader:

1. keeps only rows whose `category` is in the configured domain dictionary;
2. maps the category string to an integer domain ID;
3. tokenizes `content`, `comments`, and `expert_comment`;
4. uses BERT tokenization for `ch1` and RoBERTa tokenization for `en`;
5. applies truncation and max-length padding (`max_len=170` by default);
6. stacks the emotion/style fields and converts them to `float32` tensors;
7. converts `label` and category IDs to tensors;
8. shuffles training batches and leaves validation/test batches unshuffled.

The supplied code snapshot does **not** implement the upstream raw-data collection or benchmark-construction pipeline. Those materials should be obtained from the original repositories listed above.

## 6. Adding the LLM-generated field

Before training, each sample must contain `expert_comment`. The documented workflow is:

1. keep each sample in its original split;
2. provide the sample `content` and `category` to the LLM using the experiment prompt;
3. do not provide the ground-truth `label` to the LLM;
4. save the returned text to `expert_comment`;
5. keep the remaining sample fields and split membership unchanged;
6. write the augmented dataframe back to the corresponding `.pkl` split.

See `../prompts/README.md` for the generation workflow and disclosure notes.
