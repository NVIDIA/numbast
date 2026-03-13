This section is generated directly from:
``numbast/src/numbast/tools/static_binding_generator.schema.yaml``

Required keys
--------------

``Entry Point`` : ``string``
   Path to the input CUDA C/C++ header file.

   Example:

   .. code-block:: yaml

      Entry Point: /usr/local/cuda/include/cufp8.hpp


``GPU Arch`` : ``array``
   Target GPU architecture list. Exactly one architecture is currently supported per run.

   Constraints:

   - Min items: 1
   - Max items: 1
   - Item type: ``string``

   Example:

   .. code-block:: yaml

      GPU Arch:
      - sm_80


``File List`` : ``array``
   Header files to retain while parsing. Declarations from other transitively included files are ignored in generated
   output.

   Constraints:

   - Min items: 1
   - Item type: ``string``

   Example:

   .. code-block:: yaml

      File List:
      - /usr/local/cuda/include/cufp8.hpp
      - /usr/local/cuda/include/cufp8_conversions.hpp


Optional keys
--------------

``Name`` : ``string``
   Optional config metadata field.

``Version`` : ``string | number``
   Optional config metadata field.

``Types`` : ``object``
   Mapping of struct names to Numba type class names.

   Default: ``{}``.

   Example:

   .. code-block:: yaml

      Types:
        __nv_fp8_e4m3: FP8e4m3Type


``Data Models`` : ``object``
   Mapping of struct names to Numba datamodel class names.

   Default: ``{}``.

   Example:

   .. code-block:: yaml

      Data Models:
        __nv_fp8_e4m3: FP8e4m3Model


``Exclude`` : ``object``
   Declaration names to skip during binding generation.

   Default: ``{}``.

   Constraints:

   - No unspecified sub-keys

   Example:

   .. code-block:: yaml

      Exclude:
        Function:
        - internal_helper
        Struct:
        - __InternalState


``Clang Include Paths`` : ``array``
   Additional include search paths for Clang parsing.

   Default: ``[]``.

   Constraints:

   - Item type: ``string``

   Example:

   .. code-block:: yaml

      Clang Include Paths:
      - /usr/local/cuda/include
      - /opt/extra/include


``Additional Import`` : ``array``
   Extra Python import statements injected into the generated file.

   Default: ``[]``.

   Constraints:

   - Item type: ``string``

   Example:

   .. code-block:: yaml

      Additional Import:
      - from nvshmem.bindings import module_init


``Shim Include Override`` : ``string | null``
   Override value used to compose the generated shim include line. If unset, the entry-point path is used.

   Default: ``null``.

   Example:

   .. code-block:: yaml

      Shim Include Override: cufp8.hpp


``Predefined Macros`` : ``array``
   Macros defined before parsing and prepended in shim generation.

   Default: ``[]``.

   Constraints:

   - Item type: ``string``

   Example:

   .. code-block:: yaml

      Predefined Macros:
      - SOME_MACRO=1
      - ENABLE_FEATURE


``Output Name`` : ``string | null``
   Output binding filename. Defaults to `<entry-point-basename>.py`.

   Default: ``null``.

   Example:

   .. code-block:: yaml

      Output Name: bindings_my_lib.py


``Cooperative Launch Required Functions Regex`` : ``array``
   Regex patterns. Matching function names are generated with cooperative launch handling.

   Default: ``[]``.

   Constraints:

   - Item type: ``string``

   Example:

   .. code-block:: yaml

      Cooperative Launch Required Functions Regex:
      - ^cg_.*


``API Prefix Removal`` : ``object``
   Prefixes removed from rendered declaration names.

   Default: ``{}``.

   Constraints:

   - No unspecified sub-keys

   Example:

   .. code-block:: yaml

      API Prefix Removal:
        Function:
        - lib_
        Struct:
        - lib_
        Enum:
        - LIB_


``Module Callbacks`` : ``object``
   Optional module-level shim callback assignments.

   Default: ``{}``.

   Constraints:

   - No unspecified sub-keys

   Example:

   .. code-block:: yaml

      Module Callbacks:
        setup: 'lambda mod: print(''loaded'', mod)'
        teardown: 'lambda mod: print(''unloaded'', mod)'


``Skip Prefix`` : ``string | null``
   Skip generating bindings for functions whose names start with this prefix.

   Default: ``null``.

   Example:

   .. code-block:: yaml

      Skip Prefix: __internal_


``Use Separate Registry`` : ``boolean``
   Generate separate typing/target registries instead of reusing Numba CUDA's global registries.

   Default: ``false``.

   Example:

   .. code-block:: yaml

      Use Separate Registry: true


``Function Argument Intents`` : ``object``
   Per-function argument intent overrides. Function keys map to parameter-name or parameter-index entries. See
   :doc:`/argument_intents` for intent semantics and generated signature behavior.

   Default: ``{}``.

   Example:

   .. code-block:: yaml

      Function Argument Intents:
        my_function:
          result: out_ptr
          0: in


Optional nested keys
^^^^^^^^^^^^^^^^^^^^

.. rubric:: ``Exclude``

``Function`` : ``array``
   Function names to exclude.

   Default: ``[]``.

   Constraints:

   - Item type: ``string``

   Example:

   .. code-block:: yaml

      Function:
      - internal_helper
      - deprecated_api


``Struct`` : ``array``
   Struct names to exclude.

   Default: ``[]``.

   Constraints:

   - Item type: ``string``

   Example:

   .. code-block:: yaml

      Struct:
      - __InternalState


.. rubric:: ``API Prefix Removal``

``Function`` : ``oneOf(string, array)``
   Prefix(es) removed from generated function names.

   Example:

   .. code-block:: yaml

      Function:
      - lib_
      - mylib_


``Struct`` : ``oneOf(string, array)``
   Prefix(es) removed from generated struct names.

   Example:

   .. code-block:: yaml

      Struct:
      - lib_


``Enum`` : ``oneOf(string, array)``
   Prefix(es) removed from generated enum names and values.

   Example:

   .. code-block:: yaml

      Enum:
      - LIB_


.. rubric:: ``Module Callbacks``

``setup`` : ``string``
   Python expression for setup callback.

   Example:

   .. code-block:: yaml

      setup: 'lambda mod: print(''loaded'', mod)'


``teardown`` : ``string``
   Python expression for teardown callback.

   Example:

   .. code-block:: yaml

      teardown: 'lambda mod: print(''unloaded'', mod)'


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
       examples:
         - /usr/local/cuda/include/cufp8.hpp
     GPU Arch:
       type: array
       minItems: 1
       maxItems: 1
       items:
         type: string
         pattern: "^sm_[0-9]+$"
       description: >
         Target GPU architecture list. Exactly one architecture is currently supported per run.
       examples:
         - ["sm_80"]

     File List:
       type: array
       minItems: 1
       items:
         type: string
       description: >
         Header files to retain while parsing. Declarations from other transitively included files are ignored in
         generated output.
       examples:
         - - /usr/local/cuda/include/cufp8.hpp
           - /usr/local/cuda/include/cufp8_conversions.hpp

     Types:
       type: object
       default: {}
       additionalProperties:
         type: string
       description: Mapping of struct names to Numba type class names.
       examples:
         - {"__nv_fp8_e4m3": "FP8e4m3Type"}
     Data Models:
       type: object
       default: {}
       additionalProperties:
         type: string
       description: Mapping of struct names to Numba datamodel class names.
       examples:
         - {"__nv_fp8_e4m3": "FP8e4m3Model"}
     Exclude:
       type: object
       default: {}
       additionalProperties: false
       description: Declaration names to skip during binding generation.
       examples:
         - Function: ["internal_helper"]
           Struct: ["__InternalState"]
       properties:
         Function:
           type: array
           default: []
           items:
             type: string
           description: Function names to exclude.
           examples:
             - ["internal_helper", "deprecated_api"]
         Struct:
           type: array
           default: []
           items:
             type: string
           description: Struct names to exclude.
           examples:
             - ["__InternalState"]
     Clang Include Paths:
       type: array
       default: []
       items:
         type: string
       description: Additional include search paths for Clang parsing.
       examples:
         - - /usr/local/cuda/include
           - /opt/extra/include
     Additional Import:
       type: array
       default: []
       items:
         type: string
       description: Extra Python import statements injected into the generated file.
       examples:
         - - "from nvshmem.bindings import module_init"
     Shim Include Override:
       type: [string, "null"]
       default: null
       description: >
         Override value used to compose the generated shim include line. If unset, the entry-point path is used.
       examples:
         - "cufp8.hpp"

     Predefined Macros:
       type: array
       default: []
       items:
         type: string
       description: Macros defined before parsing and prepended in shim generation.
       examples:
         - - SOME_MACRO=1
           - ENABLE_FEATURE
     Output Name:
       type: [string, "null"]
       default: null
       description: Output binding filename. Defaults to `<entry-point-basename>.py`.
       examples:
         - bindings_my_lib.py
     Cooperative Launch Required Functions Regex:
       type: array
       default: []
       items:
         type: string
       description: >
         Regex patterns. Matching function names are generated with cooperative launch handling.
       examples:
         - - "^cg_.*"

     API Prefix Removal:
       type: object
       default: {}
       additionalProperties: false
       description: Prefixes removed from rendered declaration names.
       examples:
         - Function: ["lib_"]
           Struct: ["lib_"]
           Enum: ["LIB_"]
       properties:
         Function:
           oneOf:
             - type: string
             - type: array
               items:
                 type: string
           description: Prefix(es) removed from generated function names.
           examples:
             - ["lib_", "mylib_"]
         Struct:
           oneOf:
             - type: string
             - type: array
               items:
                 type: string
           description: Prefix(es) removed from generated struct names.
           examples:
             - ["lib_"]
         Enum:
           oneOf:
             - type: string
             - type: array
               items:
                 type: string
           description: Prefix(es) removed from generated enum names and values.
           examples:
             - ["LIB_"]
     Module Callbacks:
       type: object
       default: {}
       additionalProperties: false
       description: Optional module-level shim callback assignments.
       examples:
         - setup: "lambda mod: print('loaded', mod)"
           teardown: "lambda mod: print('unloaded', mod)"
       properties:
         setup:
           type: string
           description: Python expression for setup callback.
           examples:
             - "lambda mod: print('loaded', mod)"
         teardown:
           type: string
           description: Python expression for teardown callback.
           examples:
             - "lambda mod: print('unloaded', mod)"
     Skip Prefix:
       type: [string, "null"]
       default: null
       description: Skip generating bindings for functions whose names start with this prefix.
       examples:
         - "__internal_"
     Use Separate Registry:
       type: boolean
       default: false
       description: >
         Generate separate typing/target registries instead of reusing Numba CUDA's global registries.
       examples:
         - true

     Function Argument Intents:
       type: object
       default: {}
       description: >
         Per-function argument intent overrides. Function keys map to parameter-name or parameter-index entries.
         See :doc:`/argument_intents` for intent semantics and generated signature behavior.
       examples:
         - my_function:
             result: out_ptr
             0: in

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
