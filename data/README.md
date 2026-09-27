# Data preparation

The training loader expects preprocessed `train.pkl`, `val.pkl`, and `test.pkl` files.
Raw third-party datasets are **not redistributed** in this repository. Please obtain the
source datasets from their original providers and comply with their licenses/access terms.

## Expected locations

- Chinese benchmark: `data/ch1/train.pkl`, `data/ch1/val.pkl`, `data/ch1/test.pkl`
- English benchmark: `data/en/train.pkl`, `data/en/val.pkl`, `data/en/test.pkl`

## Expected dataframe columns

The supplied loader (`utils/dataloader.py`) expects each split to contain:

- `content`: post/news text
- `comments`: associated comment/context text
- `expert_comment`: pre-generated LLM auxiliary/background text
- `content_emotion`: numeric feature vector
- `comments_emotion`: numeric feature vector
- `emotion_gap`: numeric feature vector
- `style_feature`: numeric feature vector
- `label`: binary authenticity label
- `category`: domain label string

The manuscript uses an 8:1:1 train/validation/test split. Use the exact split files used
for the reported experiments when reproducing published numbers.

## Domain labels

Chinese (`ch1`): 科技, 军事, 教育考试, 灾难事故, 政治, 医药健康, 财经商业, 文体娱乐, 社会生活.

English (`en`): `gossipcop`, `politifact`, `COVID`.
