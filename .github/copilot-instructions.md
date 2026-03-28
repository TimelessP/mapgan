# Copilot Instructions

This repository is a compact, reproducible research toy. Optimize for clarity, determinism, and small moving parts.

## Project Shape

- Keep the core algorithm centered in `mapgan.py` unless there is a strong reason to split it.
- Prefer straightforward math and search code over abstract frameworks.
- Treat `standalone.html` as a tiny, dependency-free artifact. Keep it self-contained and hard-coded.

## Reproducibility

- Preserve deterministic behavior whenever possible.
- If solver defaults or public reference metrics change, update `reproduce_leaderboard.sh` in the same change.
- If the canonical target or reference outputs change intentionally, update the checked-in artifacts together:
  - `data/countries.geo.json`
  - `out/target_256x128.png`
  - `out/best_overall.json`
  - `out/best_overall.png`
  - `out/best_overall_diff.png`

## Documentation Expectations

- Keep `README.md` concise and showcase-oriented.
- Keep `IMPLEMENTATION.md` as the deeper explanation of how and why the algorithm works.
- If metrics or artifacts change, update the numbers and examples in both docs when needed.

## Dependency Policy

- Prefer the current lightweight stack: Python stdlib, `numpy`, and `Pillow`.
- Do not add heavy dependencies or build tooling unless they materially improve the core map approximation workflow.

## Output And Git Hygiene

- Do not start tracking generated files broadly under `out/`.
- Only the reference artifacts intentionally exposed in `.gitignore` should stay checked in.
- Keep binary or image additions deliberate and minimal.

## Editing Style

- Favor compact code, but not at the expense of readability.
- Keep comments rare and useful.
- When changing the search or scoring logic, prefer root-cause improvements over parameter churn alone.