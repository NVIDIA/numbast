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
- ``clangdev`` available (see ``conda/environment-[CUDA_VER].yaml``).
- A header entry point and a list of headers to retain.

Configuration file
------------------

Create a YAML config describing what to generate. Example:

.. code-block:: yaml

  # file: config.yml
  entry_point: /path/to/library/header.hpp
  retain_list:
    - /path/to/library/header.hpp
    - /path/to/library/other_deps.hpp

  gpu_arch: ["sm_80"]

  # Optional controls
  exclude_functions: []
  exclude_structs: []
  clang_includes_paths:
    - /extra/include/dirs
  additional_imports:
    - from numba import types
  predefined_macros:
    - SOME_MACRO=1
  output_name: bindings_my_lib.py
  cooperative_launch_required_functions_regex: []
  api_prefix_removal:
    Function: ["lib_"]
  module_callbacks:
    setup: |
      def _setup():
          pass
    teardown: |
      def _teardown():
          pass
  skip_prefix: null
  separate_registry: false

Generate the binding
--------------------

Use the CLI to generate a Python file:

.. code-block:: bash

  # from repo root (ensure conda env is active)
  python -m numbast --cfg-path config.yml --output-dir ./output

This produces a module like ``./output/bindings_my_lib.py`` (or ``<entry_point>.py`` if ``output_name`` is not set).

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
