# Numbast 0.2.0 (Nov 2024)

## New Features
- Support static binding generation via CLI command `python -m numbast --cfg-path /path/to/cfg --output-dir /path/to/output`

## Dependency Changes
- Numbast now depends on `numba-cuda` package
- Numbast now requires `cuda-version >=12.5`

# Numbast 0.1.0 (Feb 28 2024)

## New Features

- Initial Release
- Numbast: Launching supoprt for bfloat16, fp16, cuRAND
- AST_Canopy: Support extracting top-level decls from struct/classes, functions, enums, etc.
