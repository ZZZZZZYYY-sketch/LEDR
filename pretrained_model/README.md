# Pretrained text encoders

The supplied code loads pretrained encoders from local directories:

- Chinese: `pretrained_model/chinese-bert-wwm-ext/`
- English: `pretrained_model/roberta-base/`

Download the corresponding Hugging Face model files before training, or edit the
`from_pretrained(...)` calls in `models/dblk.py` and `utils/dataloader.py` to use a
Hugging Face model identifier directly.

Pretrained weights are intentionally not committed to this repository.
