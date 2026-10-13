# Proposal: Numbast-backed Rust device binding generation

- Status: draft research recommendation
- Scope: device-binding portion of internal `cuda-python-private/492`
- Target consumer: CUDA-Oxide device code linked with an existing CUDA LTOIR
  library

## Executive summary

Numbast should grow a first-class Rust device-binding backend instead of each
CUDA library maintaining its own AST Canopy-to-Rust script. The initial backend
should target a deliberately narrow contract:

- The native API consists of externally visible, C-compatible CUDA device
  functions.
- The native library is already compiled to LTOIR for the selected GPU
  architecture.
- Numbast parses and selects declarations, applies its existing configuration
  model, and emits CUDA-Oxide Rust source plus link metadata.
- CUDA-Oxide compiles the Rust caller and links it with the supplied device
  LTOIR.
- There is no generated CUDA C++ shim for the C API rounds. When argument
  intents are added, adaptation is emitted as inline Rust around an exact raw
  `extern "C"` declaration.

Delivery should be organized as complete feature rounds, not as a sequence of
small infrastructure milestones. Round 1 delivers an end-to-end plain C device
backend and uses it to cover the complete selected NVSHMEM device API. Round 2
adds Numbast argument intents. Round 3 adds overloaded and templated free
functions. Round 4 adds record/class APIs and class templates. Each round owns
all refactoring, configuration, documentation, and testing needed to be a
usable deliverable; those are workstreams within a round, not separate release
stages.

This is an appropriate Numbast investment even though the final artifact is
Rust rather than Python. Numbast already owns the CUDA-aware frontend policy:
AST Canopy invocation, declaration selection, naming rules, configuration, and
argument intents. Reimplementing those pieces beside every CUDA library would
create multiple partially compatible binding generators.

The ownership split is straightforward: cybind handles host bindings and
Numbast handles device bindings. The two efforts are complementary and can
share configuration concepts later without blocking each other.

## Context and evidence

The motivating evidence is
[NVSHMEM4Rust's generator](https://github.com/NVIDIA/nvshmem/blob/e937b47514c78118fb5d105274874bf4b5bbce34/contrib/nvshmem4rust/generator/generate_rust_bindings.py).
It demonstrates that AST Canopy can discover CUDA device declarations and that
the declarations can be rendered as CUDA-Oxide `#[device] extern "C"` entries.
It also demonstrates the cost of leaving Rust generation at the per-project
script level. The script contains its own:

- C/CUDA-to-Rust type map;
- pointer depth and `const` reconstruction;
- Rust identifier and keyword handling;
- function selection and name transformation;
- Rust declaration renderer;
- handwritten aliases, ABI types, and constants; and
- build-system integration.

The script uses a Numbast-shaped YAML configuration and AST Canopy, but it
cannot reuse Numbast's selection, type-policy, intent, diagnostics, schema, or
regression tests as one supported pipeline. Each additional CUDA library would
otherwise repeat this pattern and invent its own exceptions.

CUDA-Oxide already supports the other half of the contract: device extern
declarations and LTOIR linking. Its
[device FFI example](https://github.com/NVlabs/cuda-oxide/tree/main/crates/rustc-codegen-cuda/examples/device_ffi_test)
demonstrates Rust kernels calling functions supplied by external CUDA LTOIR.

### Current NVSHMEM4Rust device exception inventory

In the referenced NVSHMEM4Rust generator, the device function renderer itself
is generic: each AST Canopy function is passed through the same Rust declaration
emitter. No device function is handled by a function-name-specific workaround
in that path. The project-specific handling is concentrated around the
function declarations:

- a manually curated `_TYPE_MAP`;
- manual recovery of pointer depth and per-level `const` qualifiers;
- handwritten Rust representations for `__half`, `__nv_bfloat16`, and
  `double2`;
- handwritten team aliases and team, comparison, signal, and shared-memory
  constants; and
- `bypass_parse_error=True`, which permits generation to continue without a
  complete policy for surfacing skipped declarations.

The generator's named special cases for `nvshmemx_cumodule_init`,
`nvshmemx_cumodule_finalize`, and a small set of safe wrappers are in its host
API path, not its device declaration path. They belong to the host/cybind
follow-up rather than this implementation.

This inventory suggests that the first generalizable device gaps are type and
constant representation, qualifier preservation, diagnostics, and semantic
configuration—not per-function code generation. Round 1 makes those gaps part
of NVSHMEM acceptance instead of hiding them in a library-local prelude.

## Why implement this in Numbast?

### The reusable boundary is larger than parsing

AST Canopy provides a CUDA-aware declaration model, but parsing is only the
first stage of a maintained binding generator. A supported backend also needs
stable answers for which declarations are public, what their exported names
are, how aliases and qualifiers are normalized, how configuration overrides
are applied, how unsupported declarations are reported, and how semantic
argument intents change the public API.

Numbast already has implementations or established policy for:

| Capability | Reuse for Rust device bindings |
| --- | --- |
| AST Canopy invocation | Reuse CUDA architecture, include paths, predefined macros, retained-file selection, and parse diagnostics. |
| Configuration | Reuse `Entry Point`, `GPU Arch`, `File List`, `Exclude`, `Clang Include Paths`, `Predefined Macros`, `Skip Prefix`, `API Prefix Removal`, `Output Name`, and `Function Argument Intents`. |
| Declaration selection | Reuse exclusion, retained-file, duplicate, and CUDA execution-space filtering. |
| Naming policy | Reuse API prefix removal and conflict detection; add Rust-specific identifier sanitization. |
| Intent model | Reuse `ArgIntent`, override lookup by name or index, visible-argument ordering, and out-return ordering. Generalize validation from C++ references to C pointers. |
| Type processing | Reuse qualifier normalization concepts, fixed-size array parsing, typedef grouping, and size/alignment metadata. Emit Rust types instead of Numba types. |
| Diagnostics | Reuse the principle of actionable unsupported-type/declaration errors instead of silently producing a questionable ABI. |
| Reproducibility | Reuse generated-file provenance: generator version, AST Canopy version, config path, toolkit version, target architecture, and generation command. |
| Tests | Reuse header fixtures and configuration cases across Numba and Rust renderers where their semantics overlap. |

The architecture should make this division explicit:

```text
CUDA headers
    |
    v
AST Canopy declarations
    |
    v
Numbast target-neutral binding model
    |- selection and naming
    |- normalized C ABI types
    |- aliases and layout metadata
    `- argument intent plan (Round 2+)
          |                         |
          v                         v
   Numba renderer          Rust/CUDA-Oxide renderer
```

The Rust renderer should not import Numba lowering code. As a prerequisite,
the shared configuration and binding-model code should be moved out of modules
whose import initializes Numba-specific state.

### Project ownership is aligned

The proposed ownership split is:

- **AST Canopy** understands CUDA source and reports declarations.
- **Numbast** turns declarations plus human-supplied API semantics into a
  target-neutral binding model and renders Rust bindings.
- **CUDA-Oxide** defines the Rust device-extern syntax, compiles Rust device
  code, and links device artifacts.
- **The CUDA library** owns its public header, LTOIR artifact, and any
  library-specific semantic configuration.

This keeps the compiler and linker out of Numbast while avoiding library-local
copies of the CUDA header frontend and semantic policy.

## Counterarguments and responses

Only three alternatives materially affect the device-bindgen decision:

| Alternative | Response |
| --- | --- |
| Use Rust `bindgen` | `bindgen` remains appropriate for conventional host FFI, but it does not model CUDA execution space, CUDA-Oxide device externs, LTOIR inputs, or Numbast argument intents. Adapting its output would still require a CUDA-specific generator. |
| Keep an AST Canopy-to-Rust script in each library | This works for a POC, as NVSHMEM4Rust demonstrates, but duplicates type conversion, qualifier handling, naming, configuration, diagnostics, and tests. That reusable policy belongs in Numbast. |
| Handwrite the device externs | This is viable for a tiny stable API, but not for NVSHMEM's macro-expanded surface and release-to-release header changes. Generation provides repeatable symbol, qualifier, and ABI-drift checks. |

A new serialized IDL is not required for the first implementation. CUDA
headers remain the source of truth, with AST Canopy declarations plus Numbast
configuration forming the internal language-neutral model.

The generator is a build-time tool; generated Rust has no Python, Numbast, or
Numba runtime dependency. The shared frontend should therefore be importable
without initializing Numba-specific lowering.

## Responsibility boundaries

### Numbast owns

- invoking AST Canopy consistently;
- normalizing declarations into a target-neutral model;
- selection, exclusion, naming, conflict detection, and intent planning;
- C-ABI-to-Rust type rendering and layout checks;
- emitting raw device externs, inline intent adapters, and a manifest; and
- deterministic generation, diagnostics, and binding-level tests.

### CUDA-Oxide owns

- the behavior and stability of `#[device] extern "C"`;
- compiling generated Rust and its callers to device IR;
- consuming or translating link-input metadata;
- libNVVM/nvJitLink orchestration and artifact caching; and
- reporting architecture or device-link failures.

### The bound CUDA library owns

- a C-compatible public device ABI;
- producing and versioning the matching LTOIR artifact;
- intent configuration for semantic pointer parameters;
- supplemental definitions for constructs that do not exist in the AST, such
  as selected preprocessor constants; and
- library-specific end-to-end correctness tests.

### Host/device ownership split

cybind owns host binding generation; Numbast owns device binding generation.
Future configuration alignment is useful, but a unified generator is not a
goal of this proposal.

## Scope and feature delivery rounds

This section is both the product scope and the delivery roadmap. Each round is
a complete usable increment, divided by user-visible Numbast feature families
rather than internal implementation steps. Frontend extraction, rendering,
link metadata, documentation, and testing are work within the relevant round,
not separate deliverables.

All rounds consume the same core inputs: an entry-point header and retained-file
list, Clang/CUDA include paths and macros, a parsing architecture, existing
Numbast selection and naming configuration, and matching external LTOIR link
metadata. CUDA headers remain the syntactic source of truth. Generated output
is an includable Rust module plus a versioned symbol/link manifest and
source-aware diagnostics; generating an entire Cargo crate is not required.

The configuration remains additive to the existing Numbast schema:

```yaml
Entry Point: include/library_device.h
GPU Arch: [sm_90]
File List:
  - include/library_device.h
Predefined Macros:
  - LIBRARY_BUILD_DEVICE_LTOIR
API Prefix Removal:
  Function: [library_]
Output Name: library_device.rs

Backend: cuda-oxide
CUDA Oxide:
  LTOIR Inputs:
    - lib/library_device.ltoir
```

Parser modes are frontend implementation details rather than library policy.
The initial backend should also reject unknown ABI types instead of relying on
an unrestricted `Rust Types` escape hatch; explicit external or opaque type
mappings require documented ABI tests.

With agent-assisted implementation, Round 1 is intended to be completed as one
focused end-to-end push.

| Numbast capability | Rust backend round |
| --- | --- |
| C-linkage free functions and the C data model they require | Round 1 |
| Argument intents (`in`, `inout_ptr`, `out_ptr`, `out_return`) | Round 2 |
| Overloaded free functions, function templates, and free operators | Round 3 |
| Concrete record/class APIs, methods, constructors, conversions, and class templates | Round 4 |

Each round includes implementation, schema changes, documentation, golden and
compile tests, and an executable device test. There is no later "hardening"
round in which these basic completion requirements are deferred.

### Round 1: plain C device API, complete for NVSHMEM

The first deliverable is a usable Numbast Rust backend, not just a toy function
POC. It must generate the complete selected **device-callable** NVSHMEM C API
surface. "Complete" is measured against an explicit retained-header and
preprocessor profile: every declaration in that profile is either generated or
reported as an error/intentional exclusion. Host-only NVSHMEM APIs remain out
of scope.

The native contract is externally visible, non-variadic `extern "C"`
`__device__` or `__host__ __device__` free functions represented in the
supplied LTOIR. Macro-generated declarations are included after preprocessing
when AST Canopy retains them. Supported signatures cover C scalar types,
pointers with their `const` qualifiers, typedefs, enums, opaque handles, and
the concrete CUDA/POD ABI types required by NVSHMEM. Every raw declaration is
`unsafe`.

This round includes all work required to make that result real:

1. Reuse or extract the target-neutral parts of Numbast configuration,
   declaration selection, prefix removal, duplicate detection, typedef
   grouping, source diagnostics, and generated-file provenance. The extraction
   should be only as broad as the Rust backend and existing Numba backend need.
2. Generate deterministic CUDA-Oxide `#[device] unsafe extern "C"`
   declarations with Rust keyword handling, exact primitive mappings, pointer
   depth, and per-level `const` qualification.
3. Cover the C data surface used by NVSHMEM: typedefs, enums, opaque handles,
   required POD records, CUDA scalar/vector types, half/bfloat16 types, and
   required constants. Preprocessor-only constants may use a documented
   declarative supplement, but not a library-local code emitter.
4. Emit a versioned manifest with architecture, expected LTOIR inputs, exported
   symbols, names, provenance, and excluded declarations. Numbast supplies
   metadata; CUDA-Oxide continues to own compilation and linking.
5. Add a maintained end-to-end C-device example that demonstrates direct
   symbol resolution from external LTOIR without a generated CUDA C++ shim.
6. Replace NVSHMEM4Rust's device declaration renderer with this backend and
   delete its duplicated type parser, Rust keyword policy, and function
   renderer. Keep NVSHMEM-specific selection and supplemental ABI facts in
   configuration.

Round 1 is accepted only when:

- no selected NVSHMEM device function requires a handwritten Rust declaration;
- generated and linked symbol inventories agree, with no silent omissions;
- every by-value ABI type has C/CUDA-versus-Rust size, alignment, signedness,
  and relevant field-offset tests;
- parser/golden/compile tests run without a GPU, and an end-to-end test
  generates bindings, compiles a Rust kernel, links the NVSHMEM LTOIR, and
  executes on a supported GPU;
- architecture and unsupported-declaration failures are actionable; and
- the existing Numba backend's shared configuration behavior does not regress.

Argument-intent adapters, templates, and C++ declarations are explicitly not
part of Round 1. Keeping them out is what makes "all NVSHMEM C device APIs" a
crisp, achievable first product boundary.

### Round 2: argument intents and Rust call-shape adapters

Round 2 applies Numbast's existing semantic intent vocabulary to the raw C
externs delivered in Round 1. It generalizes the shared intent planner from
C++ references to C pointers and emits inline Rust adapters for `in`,
`inout_ptr`, `out_ptr`, and `out_return`.

| Intent | Rust adapter behavior |
| --- | --- |
| `in` | Keep the ABI argument visible. Raw pointers remain raw pointers. |
| `inout_ptr` | Keep a writable pointer visible and reject a `const` pointee. |
| `out_ptr` | Keep caller-owned output storage visible and reject a `const` pointee. |
| `out_return` | Hide a single-object output pointer, allocate `MaybeUninit<T>`, and return the initialized value. |

For multiple `out_return` parameters, outputs follow declaration order. A
native return is the first tuple element when present. This does not imply
array, optional-output, or length-coupled-buffer support.

This round includes name/index override precedence, writable-pointee
validation, `MaybeUninit` handling, multiple-output tuple ordering, and native
return-plus-output composition. It also applies intent configuration to the
NVSHMEM APIs for which the library can state the necessary semantics. The raw
`sys` module stays ABI-identical to Round 1, and no CUDA C++ shim or additional
LTOIR artifact is introduced.

Acceptance requires compile and GPU tests for void returns, scalar returns,
single and multiple outputs, in/out pointer combinations, and invalid
intent/type combinations. Generated adapters remain `unsafe`: intent metadata
does not prove pointer validity, aliasing, initialization, or synchronization.

### Round 3: overloaded and templated free functions

Round 3 expands beyond the C-only contract to the free-function C++ features
already modeled by Numbast: overload sets, free operators, function templates,
template argument deduction, and explicit specializations/instantiations. The
Rust API should expose stable, non-colliding call paths while the raw layer
resolves each concrete device symbol.

This round must first select and document a C++ symbol strategy. If
CUDA-Oxide can reliably bind the existing instantiated symbols, the manifest
records their exact mangled names. Otherwise Numbast may generate a small
per-specialization C-linkage LTOIR bridge. Such a bridge belongs only to this
C++ feature layer; it must not return to the Round 1 or Round 2 C path.

Acceptance is parity against Numbast's supported free-function template
fixtures, including overload resolution, multiple arities, explicit template
arguments, inferred type arguments, and clear rejection of a template that has
no concrete device instantiation.

### Round 4: record/class APIs and class templates

Round 4 covers Numbast's remaining structured C++ declaration families:
concrete records/classes, constructors, conversion operators, public fields,
non-mutating methods, operator overloads, nested and fixed-size-array POD
layout, templated methods, and class templates with explicit or deduced
parameters.

This is deliberately separate from free-function templates because Rust
layout, value construction, receiver semantics, and lifetimes require a larger
ABI and safety contract. The C++ symbol/bridge decision from Round 3 is reused,
not redesigned. Acceptance is based on the declaration matrix documented in
Numbast's `supported_declarations` and template fixtures, with layout tests and
device execution for every supported record-passing mode.

## Work outside the four feature rounds

The following work is not required for the C API, intent, template, or class
rounds. It should be added only in response to an audited library requirement:

| Deferred work | Reason to defer | Reconsider when |
| --- | --- | --- |
| Variadic functions | Rust/CUDA calling convention and type safety are unclear. | Only with a concrete required API and dedicated ABI tests. |
| Arbitrary unions, bitfields, callbacks, and function pointers | Layout or call semantics need targeted design. | Individually, driven by audited library APIs. |
| Slices, strings, optional pointers, and length-coupled buffers | Existing intents do not carry enough cardinality or lifetime metadata. | After a richer semantic schema is designed. |
| Automatic conversion of status codes to `Result` | Error domains and output-initialization rules are library-specific. | In a higher-level safe-wrapper project or richer shared semantic model. |
| Claiming generated wrappers are safe Rust | Intent metadata alone cannot establish device pointer validity, aliasing, synchronization, or initialization. | Only after those contracts are modeled and verified. |
| Compiling the native library to LTOIR | The library build owns flags, architecture, versioning, and distribution. | Never by default; optional orchestration may call a library-provided build command. |
| Host API generation and dynamic loading | This is the cybind/host-`bindgen` portion of the parent research issue. | In its own recommendation and implementation plan. |
| A universal serialized IDL | Requirements are not mature enough to freeze one. | After at least two backends and two production library migrations exercise the normalized model. |

## Open decisions for review

The following decisions should be closed as part of Round 1 unless a later
round is named explicitly:

1. Whether the backend is selected by `Backend: cuda-oxide`, a dedicated
   `numbast-cuda-oxide` command, or both over the same implementation.
2. Whether external LTOIR paths live in Numbast config, Cargo metadata, or a
   generated manifest consumed by CUDA-Oxide. The recommendation is a
   generated versioned manifest, with config supplying the source inputs.
3. Whether shared frontend packaging remains inside `numbast` with optional
   Numba dependencies or becomes a small companion package. The implementation
   should first establish a clean module boundary before splitting packages.
4. How preprocessor-only constants and externally defined ABI types are
   supplied without recreating an unrestricted project-specific Rust emitter.
5. In Round 3, whether CUDA-Oxide can bind instantiated C++ symbols directly or
   a generated per-specialization C-linkage LTOIR bridge is required.

## References

- Internal source issue:
  [NVIDIA-dev/cuda-python-private#492](https://github.com/NVIDIA-dev/cuda-python-private/issues/492)
  (NVIDIA access required).
- Public tracker stub:
  [NVIDIA/numbast#382](https://github.com/NVIDIA/numbast/issues/382).
- [NVSHMEM4Rust generator](https://github.com/NVIDIA/nvshmem/blob/e937b47514c78118fb5d105274874bf4b5bbce34/contrib/nvshmem4rust/generator/generate_rust_bindings.py).
- [Rust bindgen user guide](https://rust-lang.github.io/rust-bindgen/).
- [Rust bindgen C++ support and limitations](https://rust-lang.github.io/rust-bindgen/cpp.html).
- [CUDA-Oxide device FFI example](https://github.com/NVlabs/cuda-oxide/tree/main/crates/rustc-codegen-cuda/examples/device_ffi_test).
- [Numbast argument intents](source/argument_intents.rst).
