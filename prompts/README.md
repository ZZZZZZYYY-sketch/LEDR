# LLM-assisted context generation

The model loader expects a pre-generated text field named `expert_comment` for every sample.

## Documented workflow

For each sample in the already defined train/validation/test split:

1. Read the sample's `content` and `category` (domain).
2. Insert those fields into the experiment prompt.
3. Ask the LLM to generate the requested domain-aware background/context text in the format specified by that prompt.
4. Do **not** provide the fake/real ground-truth `label` to the LLM.
5. Save the generated response as the sample's `expert_comment` value.
6. Preserve the original sample fields and split membership.
7. Export the augmented data using the same dataframe schema described in `../data/README.md`.
8. Finish generation before model training so that the generated context is fixed during training/evaluation.

Conceptually:

```text
(content, category)
        |
        v
  experiment prompt
        |
        v
       LLM
        |
        v
 generated background/context
        |
        v
expert_comment field in the original sample dataframe
```

## Manuscript-reported generation settings

The manuscript describes DeepSeek as the LLM used for background/context generation, with generation completed before model training. The reported generation settings include a low temperature (`0.1`) and a maximum output length of 1000 tokens.

## Important reproducibility note

The supplied `DBLK` code archive does **not** contain the exact original prompt text or the API-generation script that produced `expert_comment`. This repository therefore does not fabricate an exact prompt or API call and does not claim that an invented template reproduces the reported generated text.

If the original experiment prompt or generation script becomes available, place it in this directory and record:

- provider/tool name;
- API/model identifier;
- exact prompt text or prompt-building logic;
- generation parameters;
- date/version if known;
- input fields used;
- output parsing/post-processing rules.

Never commit API keys or tokens. Use environment variables for credentials.
