Supported declarations
======================

ast_canopy recognizes and serializes a subset of CUDA C++ declarations for downstream consumers.

Concrete structs and classes
----------------------------

- Constructors generate typing and lowering for instantiation
- Conversion operators are mapped to Python conversions
- Public fields are exposed for read access

Functions and operators
-----------------------

- Free functions receive typing and lowering for all overloads
- Operator overloads are mapped to Python operators (e.g., ``operator+`` → ``operator.add``)

Templates
---------

- Class templates (e.g., ``template<class T> struct Vec``) are serialized with template parameters
  and their declaration bodies for downstream specialization/instantiation.
- Function templates (e.g., ``template<typename T> T add(T, T)``) are serialized with parameter lists
  and signatures so consumers can materialize concrete overloads.
- Where present, explicit specializations and explicit instantiations are captured as distinct declarations.

Example mapping
---------------

.. list-table::
   :header-rows: 1

   * - Kind
     - C++ Declaration
     - C++ Usage
     - Numba Usage
     - Notes
   * - Concrete Struct Constructor
     - ``__myfloat16(double val)``
     - ``auto f = __myfloat16(3.14)``
     - ``f = __myfloat16(3.14)``
     - Generates type, data model, typing, lowering
   * - Conversion Operator
     - ``operator float()``
     - ``float x = float(f)``
     - ``x = float(f)``
     - Conversion operator mapping
   * - Public Attribute
     - ``half data``
     - ``auto d = f.data``
     - ``d = f.data``
     - Read-only access
   * - Regular Struct Method
     - ``__myfloat16 neg() const``
     - ``auto r = f.neg()``
     - ``r = f.neg()``
     - Instance method typing/lowering (non-mutative methods only; mutative methods are not supported)
   * - Function
     - ``__myfloat16 hsqrt(__myfloat16 a)``
     - ``auto r = hsqrt(f)``
     - ``r = hsqrt(f)``
     - Typing/lowering for overloads
   * - Operator overload
     - ``__myfloat16 operator+(...)``
     - ``auto twof = f + f``
     - ``twof = f + f``
     - Mapped to appropriate Python operator
   * - Class template
     - ``template<class T> struct Vec { T x; T y; };``
     - ``Vec<float> v; v.x = 1.0f;``
     - N/A (serialization only)
     - Template parameters and body captured for downstream specialization
   * - Function template
     - ``template<typename T> T add(T a, T b);``
     - ``auto r = add(1, 2);``
     - N/A (serialization only)
     - Template parameters/signature captured; specializations/instantiations recorded when present

Qualified names (``qual_name``)
-------------------------------

Many serialized declaration objects expose a ``qual_name`` attribute: the C++
*qualified name* including enclosing scopes (namespaces and record scopes),
using ``::`` as the separator.

This is derived from Clang's ``Decl::getQualifiedNameAsString()`` with small
stability tweaks for anonymous records so downstream consumers always have a
printable identifier.

.. list-table::
   :header-rows: 1

   * - Declaration kind
     - Example ``qual_name``
   * - Function / method
     - ``ns1::ns2::S::m``
   * - Record (struct/class)
     - ``ns1::ns2::S``
   * - Enum
     - ``ns1::ns2::E``
   * - Typedef
     - ``ns1::ns2::Alias``
   * - Function template / class template
     - ``ns1::ns2::tf`` / ``ns1::ns2::Tpl``

Notes and edge cases
^^^^^^^^^^^^^^^^^^^^

- **Global scope**: in the global scope (no namespace), ``qual_name`` is
  typically the unqualified identifier (e.g., ``GlobalS``).

- **Anonymous namespace**: Clang typically renders anonymous namespaces as
  ``(anonymous namespace)`` in qualified names. For example, a declaration
  ``AnonNS_S`` inside ``namespace { ... }`` may have a qualified name like
  ``(anonymous namespace)::AnonNS_S``.

- **Anonymous records in C-style typedefs**: for patterns like

  .. code-block:: cpp

     typedef struct { int a; int b; } CStyleAnon;

  the underlying record has no tag name and Clang may report an empty name.
  ast_canopy falls back to a placeholder ``unnamed<ID>`` for the *record's*
  ``name`` so downstream always has something printable (note: ``<ID>`` is a
  Clang internal decl id and is not stable across runs).

  In this common pattern, Clang often treats the typedef name as the record's
  user-visible qualified name; in that case you may observe:

  - ``Typedef.qual_name == "CStyleAnon"``
  - ``Typedef.underlying_name`` matching ``unnamed<ID>``
  - ``Record.name`` matching ``unnamed<ID>``
  - ``Record.qual_name == "CStyleAnon"``

Supported argument types
------------------------

The following argument types are supported for both standalone functions and
struct/class methods.

.. list-table::
   :header-rows: 1

   * - Type
     - Description
     - Examples
   * - C++ native types
     - Built-in arithmetic and boolean types.
     - ``int``, ``float``, ``double``, ``bool``
   * - Pointers to supported types
     - Raw pointers to the above types (including const-qualified).
     - ``int*``, ``const float*``
   * - Struct types (numbast)
     - Structs generated by numbast bindings.
     - ``my_ns::MyStruct``
   * - Enums (numbast)
     - Enums generated by numbast bindings.
     - ``my_ns::MyEnum``

See also
--------

- :doc:`overview`
- :doc:`static`
- :doc:`dynamic`
