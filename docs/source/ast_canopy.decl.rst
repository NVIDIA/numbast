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

Declaration objects
-------------------

The objects in this module represent information extracted from the C++ Clang
AST. ``Declarations`` also contains native typedef and enum objects, as
specified in its attribute documentation.

.. automodule:: ast_canopy.decl
   :members:
   :undoc-members:
   :show-inheritance:
