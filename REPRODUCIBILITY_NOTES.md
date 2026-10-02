# Reproducibility notes for the supplied code snapshot

This repository was cleaned from the supplied `DBLK` research-code archive. Documentation and safe output-directory handling were added, but the underlying research model/loss has **not** been silently rewritten to force agreement with manuscript text that is not represented in the supplied source.

The following points should be kept in mind when comparing this code snapshot with the final manuscript:

1. **LLM feature fusion**: `utils/dataloader.py` loads and tokenizes `expert_comment`, but the `expert_feature` encoding/fusion path is commented out in `models/dblk.py` in the supplied snapshot.
2. **Contrastive objective**: the snapshot calls `SCL(temperature=0.1)` from `utils/utils.py`; a separate explicit `eta=3` parameter is not present in the supplied implementation.
3. **Loss weight**: `Combined_KD_m.py` instantiates `CloserModel(..., 0.15)` for both datasets in this snapshot.
4. **Dropout**: `mainCKD.py` sets `dropout=0.2` in this snapshot.
5. **LLM generation assets**: the exact DeepSeek prompt and API-generation script are absent. The data loader assumes generated text is already stored in `expert_comment`.
6. **Raw-data preparation**: the snapshot expects processed split files and does not include the original scripts for collection/merging/feature extraction.

These items are documented rather than auto-corrected because modifying them without the exact experiment source could create code that was not actually used.
