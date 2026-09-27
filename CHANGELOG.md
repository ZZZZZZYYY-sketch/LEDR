# Release cleanup changelog

This cleanup preserves the supplied research-code snapshot and avoids inventing model logic.

Changes made:

- flattened the nested archive into a GitHub-friendly repository;
- added `README.md`, `requirements.txt`, `.gitignore`, data/model setup notes, and release notes;
- moved the non-runnable `SNE.py` script to `legacy/`;
- added empty output directories with `.gitkeep` files;
- changed one hard-coded `.cuda()` domain-index construction to use the model device;
- ensured `recodertestpkl/` is created before saving a test-time model snapshot;
- added `train.py` as a convenience wrapper around the original `mainCKD.py` entry point;
- did not change the model architecture, loss definition, paper hyperparameters, or reported results.
