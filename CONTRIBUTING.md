# Contributing

## Set up

```bash
git clone https://github.com/700799/tunepac && cd tunepac
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # base install: config, CLI, 80+ unit tests
pip install -e ".[train,dev]"      # add this to run the training integration tests
```

Every check CI runs, in the order it runs them:

```bash
ruff check .
mypy
pytest -q                          # integration tests skip without [train]
```

## The three test tiers

| Tier | Command | Needs | Time |
|---|---|---|---|
| Unit | `pytest -q` | base install | seconds |
| Integration | `pytest -q tests/test_integration.py` | `[train]`, CPU is fine | ~1 min |
| Smoke | `python scripts/run_test.py` | `[train]` + network for SmolLM2 | ~30 s on CPU |

The integration tests build a ~100k-parameter Llama checkpoint on the fly, so
they need no network and no GPU. They exercise the real `run_training` and
`merge_adapter` code paths.

## Conventions

- `mypy --strict` is enforced on `tunepac/`. Untyped ML libraries are listed in
  `[tool.mypy.overrides]`; add to that list rather than sprinkling `type: ignore`.
- The schema is the contract. Every invariant is checked in
  `tunepac/config/schema.py`; the trainer never re-validates what the schema
  guarantees. If you add a knob, add its validation and a test in
  `tests/test_schema.py` in the same change.
- Heavy imports (`torch`, `transformers`, `trl`, `peft`) stay inside functions,
  so `tunepac validate` keeps working on a machine without CUDA.
- The TRL upper bound (`trl<1.14`) exists because TRL 1.14 made a Triton kernel
  unconditional in `SFTTrainer.compute_loss`, which breaks CPU training. Lift
  it only once `python scripts/run_test.py` passes on CPU against the newer
  release.

## Diagrams

The README's flow diagrams are rendered SVGs so they show on PyPI, which does
not render Mermaid. Edit the sources in `docs/diagrams/*.mmd`, then:

```bash
pip install playwright && playwright install chromium   # once
npm install --no-save mermaid                            # once
python scripts/render_diagrams.py
```

Commit the regenerated `docs/diagrams/*-light.svg` and `*-dark.svg`.

## Releasing

A release is one version bump, one merge, one click.

1. Edit `__version__` in `tunepac/__init__.py` (the only place the version
   lives) and move the `Unreleased` section of `CHANGELOG.md` under the new
   version. Open a PR, let CI pass, merge.
2. On GitHub → **Releases → Draft a new release**, create tag `vX.Y.Z` on
   `main` and publish.
3. `.github/workflows/release.yml` runs on the tag: ruff, mypy, pytest, a check
   that the tag matches `__version__`, build, `twine check`, then upload via
   PyPI Trusted Publishing. No token is involved.

A PyPI upload cannot be undone and a version number can never be reused, so
the workflow refuses a tag that does not match the packaged version.
