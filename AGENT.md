# AGENT.md

This repository contains a PyTorch implementation of a 3PG-RNN single-point model.

## Repository layout
- Core model code is in `pytorch_3pg_rnn_single_point/`.
- Training and configuration data live in `pytorch_3pg_rnn_single_point/data_files/`.
- Project overview and usage examples are in `README.md` and the included notebooks.

## Development guidelines
- Keep changes focused and minimal.
- Prefer clear, descriptive function and variable names.
- Preserve existing module boundaries and file organization unless a refactor is explicitly requested.
- Add or update documentation when behavior changes.

## Validation
- For Python edits, run lightweight syntax checks when possible:
  - `python -m compileall pytorch_3pg_rnn_single_point`
- If dependencies are available, run any relevant project scripts or notebooks to sanity check behavior.

## Notes for automated agents
- Avoid introducing unrelated dependencies.
- Do not commit large generated artifacts.
- Keep file paths relative and portable.
