ast_canopy.decl
===============

AST Canopy parses a source file and its included headers as one translation
unit. The parsing API returns declarations from the files selected by
``files_to_retain``, grouped in a :class:`~ast_canopy.api.Declarations` object.

Parsing declarations
--------------------

.. autofunction:: ast_canopy.api.parse_declarations_from_source

Declaration collections
-----------------------

.. autoclass:: ast_canopy.api.Declarations

Macro definitions and values
----------------------------

``Declarations.macro_defines`` preserves the replacement text of active
object-like macros defined in retained files. ``macro_constants`` additionally
provides evaluated scalar values and their C++ types through
:class:`~ast_canopy.decl.EvaluatedConstDecl` objects.

For example, given a retained header containing:

.. code-block:: cpp

   struct Config { int value; };
   #define API_MASK (1ULL << 3)
   #define API_VERSION ((2 << 16) + sizeof(Config))
   #define API_FLAG

the parse result exposes both forms:

.. code-block:: python

   decls.macro_defines["API_MASK"]           # "( 1ULL << 3 )"
   decls.macro_constants["API_MASK"].value   # 8
   decls.macro_constants["API_MASK"].type_.name  # "unsigned long long"
   decls.macro_constants["API_VERSION"].value   # 131076
   decls.macro_defines["API_FLAG"]           # ""
   "API_FLAG" in decls.macro_constants      # False

Clang expands references to other macros and evaluates arithmetic, casts,
enum values, constexpr calls, and ``sizeof`` against the parsed declarations
and selected CUDA target. Evaluation uses definitions active at the end of the
translation unit; it does not record historical values before redefinition or
expand each macro at its original definition location. Replacement text is
token spelling normalized with spaces, rather than an exact source slice.

Only integers, booleans, enums, ``float``, ``double``, and ``long double`` have
evaluated entries. Strings, pointers, aggregates, type aliases, empty flags,
runtime expressions, and invalid expressions remain available as text only.
They do not cause strict declaration parsing to fail. Function-like macros
are omitted, but an object-like macro may invoke a function-like macro as
part of an evaluable expression. Definitions from unretained headers can
participate in expansion without being exported themselves. Compiler and
command-line macros are not exported unless redefined in a retained file.

Macro results are separate from ``global_constants`` and enum declarations.
Binding generators must opt in to consuming them; extracting macros does not
automatically add constants to generated bindings.

Declaration objects
-------------------

The objects in this module represent information extracted from the C++ Clang
AST. ``Declarations`` also contains native typedef and enum objects, as
specified in its attribute documentation.

.. automodule:: ast_canopy.decl
   :members:
   :undoc-members:
   :show-inheritance:
