Static binding generation
=========================

Static binding generation produces a standalone Python module that depends only on Numba CUDA. It lets you distribute
bindings without requiring Numbast at runtime.

When to use
-----------

- You want to ship bindings as a regular Python package/module.
- You want end users to avoid installing ``clang`` tooling or ``numbast``.
- Your target environment is known and compatible with the generation environment.

Requirements
------------

- CUDA Toolkit headers installed and discoverable.
- ``clangdev`` available in your development environment.
- A header entry point and a list of headers to retain.

Configuration file
------------------

The static-binding config contract is defined in
``numbast/src/numbast/tools/static_binding_generator.schema.yaml``.
The docs section below is generated from that schema at build time to avoid
drift between implementation and documentation.
For details on ``Function Argument Intents`` semantics and generated signatures,
see :doc:`/argument_intents`.

Config example:

.. code-block:: yaml

  # file: config.yml

  # --- Required fields ---
  Entry Point: /path/to/library/header.hpp
  File List:
    - /path/to/library/header.hpp
    - /path/to/library/other_deps.hpp
  GPU Arch: ["sm_80"]

  # --- Optional fields ---
  Exclude:
    Function: ["internal_helper", "deprecated_api"]
    Struct: ["__InternalState"]
  Clang Include Paths:
    - /extra/include/dirs
  Additional Import:
    - from numba import types
  Predefined Macros:
    - SOME_MACRO=1
  Output Name: bindings_my_lib.py
  API Prefix Removal:
    Function: ["lib_"]
  Module Callbacks:
    setup: "lambda x: print('setup')"
    teardown: "lambda x: print('teardown')"

Generate the binding
--------------------

Use the CLI to generate a Python file:

.. code-block:: bash

  # from repo root (ensure your development environment is active)
  python -m numbast --cfg-path config.yml --output-dir ./output

This produces a module like ``./output/bindings_my_lib.py`` (or ``<entry_point>.py`` if ``Output Name`` is not set).

Distribute and use
------------------

- Package the generated module into your project, or publish to PyPI.
- Users import it directly without having Numbast installed.

.. code-block:: python

  from bindings_my_lib import my_function, MyStruct

Notes and tips
--------------

- Use the same CUDA version (or backward compatible) between generation and target runtime environments.
- If multiple GPU architectures are needed, run the generator once per architecture.
- If your environment has ``ruff`` formatter installed, Numbast will attempt to run it on generated bindings.

Config Schema Reference
=======================

.. include:: generated/static_binding_schema_reference.rst
