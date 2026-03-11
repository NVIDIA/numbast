This section is generated directly from:
``numbast/src/numbast/tools/static_binding_generator.schema.yaml``

Required keys
--------------

``Entry Point`` : ``string``
   Path to the input CUDA C/C++ header file.

``GPU Arch`` : ``array``
   Target GPU architecture list. Exactly one architecture is currently supported per run.
   Constraints: min items: 1; max items: 1; item type: ``string``.

``File List`` : ``array``
   Header files to retain while parsing. Declarations from other transitively included files are ignored in generated
   output.
   Constraints: min items: 1; item type: ``string``.

Optional keys
--------------

``Name`` : ``string``
   Optional config metadata field.

``Version`` : ``string | number``
   Optional config metadata field.

``Types`` : ``object``
   Mapping of struct names to Numba type class names. Default: ``{}``.

``Data Models`` : ``object``
   Mapping of struct names to Numba datamodel class names. Default: ``{}``.

``Exclude`` : ``object``
   Declaration names to skip during binding generation. Default: ``{}``.
   Constraints: no unspecified sub-keys.

``Clang Include Paths`` : ``array``
   Additional include search paths for Clang parsing. Default: ``[]``.
   Constraints: item type: ``string``.

``Additional Import`` : ``array``
   Extra Python import statements injected into the generated file. Default: ``[]``.
   Constraints: item type: ``string``.

``Shim Include Override`` : ``string | null``
   Override value used to compose the generated shim include line. If unset, the entry-point path is used. Default:
   ``null``.

``Predefined Macros`` : ``array``
   Macros defined before parsing and prepended in shim generation. Default: ``[]``.
   Constraints: item type: ``string``.

``Output Name`` : ``string | null``
   Output binding filename. Defaults to `<entry-point-basename>.py`. Default: ``null``.

``Cooperative Launch Required Functions Regex`` : ``array``
   Regex patterns. Matching function names are generated with cooperative launch handling. Default: ``[]``.
   Constraints: item type: ``string``.

``API Prefix Removal`` : ``object``
   Prefixes removed from rendered declaration names. Default: ``{}``.
   Constraints: no unspecified sub-keys.

``Module Callbacks`` : ``object``
   Optional module-level shim callback assignments. Default: ``{}``.
   Constraints: no unspecified sub-keys.

``Skip Prefix`` : ``string | null``
   Skip generating bindings for functions whose names start with this prefix. Default: ``null``.

``Use Separate Registry`` : ``boolean``
   Generate separate typing/target registries instead of reusing Numba CUDA's global registries. Default: ``false``.

``Function Argument Intents`` : ``object``
   Per-function argument intent overrides. Function keys map to parameter-name or parameter-index entries. Default:
   ``{}``.

Optional nested keys
^^^^^^^^^^^^^^^^^^^^

.. rubric:: ``Exclude``

``Function`` : ``array``
   Function names to exclude. Default: ``[]``.
   Constraints: item type: ``string``.

``Struct`` : ``array``
   Struct names to exclude. Default: ``[]``.
   Constraints: item type: ``string``.

.. rubric:: ``API Prefix Removal``

``Function`` : ``oneOf(string, array)``
   Prefix(es) removed from generated function names.

``Struct`` : ``oneOf(string, array)``
   Prefix(es) removed from generated struct names.

``Enum`` : ``oneOf(string, array)``
   Prefix(es) removed from generated enum names and values.

.. rubric:: ``Module Callbacks``

``setup`` : ``string``
   Python expression for setup callback.

``teardown`` : ``string``
   Python expression for teardown callback.

Raw schema
----------

.. code-block:: yaml

   # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   # SPDX-License-Identifier: Apache-2.0
   $schema: "https://json-schema.org/draft/2020-12/schema"
   title: Numbast Static Binding Generator Config
   description: >
     Canonical schema describing supported YAML keys and value shapes accepted by the Numbast static binding generator
     configuration loader.

   type: object
   additionalProperties: true
   required:
     - Entry Point
     - GPU Arch
     - File List
   properties:
     Name:
       type: string
       description: Optional config metadata field.
     Version:
       type: [string, number]
       description: Optional config metadata field.
     Entry Point:
       type: string
       description: Path to the input CUDA C/C++ header file.
     GPU Arch:
       type: array
       minItems: 1
       maxItems: 1
       items:
         type: string
         pattern: "^sm_[0-9]+$"
       description: >
         Target GPU architecture list. Exactly one architecture is currently supported per run.

     File List:
       type: array
       minItems: 1
       items:
         type: string
       description: >
         Header files to retain while parsing. Declarations from other transitively included files are ignored in
         generated output.

     Types:
       type: object
       default: {}
       additionalProperties:
         type: string
       description: Mapping of struct names to Numba type class names.
     Data Models:
       type: object
       default: {}
       additionalProperties:
         type: string
       description: Mapping of struct names to Numba datamodel class names.
     Exclude:
       type: object
       default: {}
       additionalProperties: false
       description: Declaration names to skip during binding generation.
       properties:
         Function:
           type: array
           default: []
           items:
             type: string
           description: Function names to exclude.
         Struct:
           type: array
           default: []
           items:
             type: string
           description: Struct names to exclude.
     Clang Include Paths:
       type: array
       default: []
       items:
         type: string
       description: Additional include search paths for Clang parsing.
     Additional Import:
       type: array
       default: []
       items:
         type: string
       description: Extra Python import statements injected into the generated file.
     Shim Include Override:
       type: [string, "null"]
       default: null
       description: >
         Override value used to compose the generated shim include line. If unset, the entry-point path is used.

     Predefined Macros:
       type: array
       default: []
       items:
         type: string
       description: Macros defined before parsing and prepended in shim generation.
     Output Name:
       type: [string, "null"]
       default: null
       description: Output binding filename. Defaults to `<entry-point-basename>.py`.
     Cooperative Launch Required Functions Regex:
       type: array
       default: []
       items:
         type: string
       description: >
         Regex patterns. Matching function names are generated with cooperative launch handling.

     API Prefix Removal:
       type: object
       default: {}
       additionalProperties: false
       description: Prefixes removed from rendered declaration names.
       properties:
         Function:
           oneOf:
             - type: string
             - type: array
               items:
                 type: string
           description: Prefix(es) removed from generated function names.
         Struct:
           oneOf:
             - type: string
             - type: array
               items:
                 type: string
           description: Prefix(es) removed from generated struct names.
         Enum:
           oneOf:
             - type: string
             - type: array
               items:
                 type: string
           description: Prefix(es) removed from generated enum names and values.
     Module Callbacks:
       type: object
       default: {}
       additionalProperties: false
       description: Optional module-level shim callback assignments.
       properties:
         setup:
           type: string
           description: Python expression for setup callback.
         teardown:
           type: string
           description: Python expression for teardown callback.
     Skip Prefix:
       type: [string, "null"]
       default: null
       description: Skip generating bindings for functions whose names start with this prefix.
     Use Separate Registry:
       type: boolean
       default: false
       description: >
         Generate separate typing/target registries instead of reusing Numba CUDA's global registries.

     Function Argument Intents:
       type: object
       default: {}
       description: >
         Per-function argument intent overrides. Function keys map to parameter-name or parameter-index entries.

       additionalProperties:
         type: object
         additionalProperties:
           oneOf:
             - type: string
               enum: ["in", "inout_ptr", "out_ptr", "out_return"]
             - type: object
               properties:
                 intent:
                   type: string
                   enum: ["in", "inout_ptr", "out_ptr", "out_return"]
               required: ["intent"]
