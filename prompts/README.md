# LLM background-knowledge generation

The training loader expects pre-generated LLM background knowledge in the dataframe
column `expert_comment`.

The original code archive supplied for this release did **not** include the exact
DeepSeek generation script or the exact prompt text used to produce that column.
Therefore, this repository does not fabricate an "exact" prompt. Add the original
prompt/generation script here if it is available from the experiment records.

The manuscript describes the generation input as the post content plus domain name,
without access to the ground-truth label, with generated knowledge fixed before model
training.
