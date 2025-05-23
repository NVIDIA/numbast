# Numbast 0.3.0 (May 2025)

Numbast version 0.3.0 delivers the bfloat16 static bindings to Numba-CUDA. Multiple
improvements on the repository made this delivery possible.

## Improvements
The majority of the change is centralized in #106
- The linkable code object that contains the shim functions are added to active_linking_library in python.
Streamed Shim Function Writes. The shim functions are written to a string stream at lowering time using a special class named KeyedStringIO that prevents double write of the same shim function.
- Ruff formatting generated code (with import sort): Add Ruff Linter run to generated bindings
- Generation metadata added to binding: Add Information to Make Binding Generation More Reproducible
- Allow user to specify entry point and retain file list in config file using relative path d1e7aa2
- Allow user to add additional imports via Additional Import config item 8f7bd22
- Allow user to override the include line in shim function via Shim Include Override config item 8f7bd22
- Allow user to specify whether to include check in the binding to assert existence of pynvjitlink bad0c69
- Allow user to specify a custom macro, which dictates both how clangTooling parses the header file, as well as how the shim function is compiled with NVRTC. 0f0ee9c

## Dependency Changes
- Depends on `numba-cuda >=0.10.0`
- Pins to `clangdev ==18`

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
