# Numbast 0.5.0 (Aug 2025)

Numbast 0.5.0 focuses on build-system integration and cleanup, simplifies
dependencies, and adds configurability across static binding generation and
parsing. Highlights include removing RAPIDS and pynvjitlink requirements,
adopting GNU extension parsing, new typing/target registry hooks, expanded
config options (skip/prefix rules, GPU arch, module setup/teardown callbacks),
clearer error surfacing, and numerous fixes and CI improvements.

## Improvements
- bump to version 0.5.0
- Explicitly Pass Cmake Config Options to `cmake` Command in Build Script (#169)
- remove dependency on RAPIDS conda channels (#167)
- Completely Remove Pynvjitlink Requirements (#159)
- Use GNU Extension Standard for Default Clang Parse (#165)
- Allow creation of custom typing/target registry (#161)
- Exclude typedefs for excluded structs (#160)
- Support "Skip Prefix" in numbast config (#156)
- Allow Numbast to Parse Bfloat16 (#155)
- Explicitly Raise ClangAST Parsing Error (#150)
- Remove Unused CI Files (#157)
- Integration to build-tools PR 2 (#154)
- Configure static binding CC via `GPU Arch` Entry (#152)
- Integration PR as part of Build Systems (#149)
- Add Config Item to Allow Module Setup/Teardown Callback Functions (#148)
- fix benchmarks, add missing tests from CI (#147)
- Fix a Bug where GlobalDecl is constructed mistakenly when performing name mangles (#146)
- Add Prefix Remover for Functions in Static Binding Generator (#144)
- Fix CI (#145)

# Numbast 0.4.0 (Jun 2025)

## Improvements
- Use function mangled name to deduplicate forward declared function decls (#142)
- Use Itanium Name Mangler to resolve shim function name conflicts (#139)
- Alter cooperative launch list to regex list (#136)
- Refactor `_static_binding_generator` with Config Input (#135)
- Get contributing section from numba-cuda on making a new release (#134)
- Add config for functions that requires cooperative launch (#133)
- Update Dependency List, Optional run Ruff (#131)
- Explicitly expose symbols via `__all__` (#129)
- Fix the numbast_extension install instruction (#132)
- Allow Custom Binding Name Specified via YAML (#128)
- Add CTPSD matcher, WAR nested `RecordDecl` (#126)
- Add ConstExpr Vardecl Matcher to Extract ConstExpr Value from VarDecl (#112)
- Use Single Upload Job for CondaBuild (#124)
- Build Against Clang20 (#125)

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
