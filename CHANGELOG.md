# Changelog

## PeerJ AI Application release preparation - 2026-10-03

- Expanded README to match the requested AI Application reproducibility structure.
- Documented the DTDBD-based software environment and additional imports required by this snapshot.
- Added official/upstream dataset sources for Weibo21, DITFEND, FakeNewsNet and MM-COVID.
- Documented the expected dataframe schema and runtime preprocessing performed by the loader.
- Documented the LLM-assisted context-generation workflow without inventing the missing exact prompt/API script.\n- Recorded the confirmed experiment model version as DeepSeek-V3.1 (`deepseek-chat`).
- Retained explicit reproducibility notes for source/manuscript differences.
- Removed the non-runnable legacy `SNE.py` folder from the public release package.

## Previous cleanup

- Added `.gitignore`, data/pretrained-model placeholders, and documentation.
- Added safe creation of local output directories where needed.
- Preserved the supplied research model/loss rather than fabricating manuscript-aligned code.
