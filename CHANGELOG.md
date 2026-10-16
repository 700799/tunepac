# Changelog

All notable changes to Tunepac are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `python -m tunepac` entry point, which also makes
  `accelerate launch -m tunepac train --config …` work for multi-GPU runs.
- Python 3.13 in the test matrix and classifiers.
- `CONTRIBUTING.md`, `SECURITY.md`, issue and pull-request templates, and
  Dependabot for GitHub Actions and pip.

### Changed
- The two README flow diagrams are now pre-rendered SVGs (light and dark), so
  they display on PyPI as well as GitHub. Sources live in `docs/diagrams/` and
  `scripts/render_diagrams.py` regenerates them.
- The package version is read from `tunepac/__init__.py` alone; `pyproject.toml`
  no longer carries a copy.

## [0.1.0] — 2026-10-12

First release on PyPI.

### Added
- One YAML file per run, validated by a strict Pydantic v2 schema:
  `base_model`, `dataset`, `training`, `peft`, `hardware`. Unknown keys are
  rejected and cross-field rules (QLoRA ⇔ quantization, QLoRA requires bf16 or
  fp16, `eval_steps` requires `validation_split`, bf16/fp16 exclusivity) fail
  before anything heavy is imported.
- `tunepac validate`, `tunepac train`, `tunepac export` CLI. `validate` and
  `export` run on a base install with no CUDA; `train` needs the `[train]`
  extra and says so if it is missing.
- LoRA and QLoRA through PEFT, with bitsandbytes NF4/FP4 quantization.
- Training on TRL `SFTTrainer` with sequence packing, gradient checkpointing,
  flash attention, checkpoint resume, and W&B / TensorBoard reporting.
- Dataset formats `alpaca`, `sharegpt`, `openai`, `completion`, from local
  JSON/JSONL/CSV/Parquet files or the Hugging Face Hub.
- Chat templates `tokenizer_default`, `chatml`, `llama3`, `mistral`, `zephyr`.
- Out-of-memory errors reported with concrete settings to try instead of a
  traceback.
- `tunepac export` merges an adapter into an unquantized copy of the base
  model, so the result loads without PEFT.
- A CPU smoke test (`scripts/run_test.py`) and an integration suite that
  trains a tiny locally-built Llama checkpoint on every CI run.
- Release workflow that re-runs every gate, refuses a tag that does not match
  `__version__`, and publishes through PyPI Trusted Publishing.

[Unreleased]: https://github.com/700799/tunepac/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/700799/tunepac/releases/tag/v0.1.0
