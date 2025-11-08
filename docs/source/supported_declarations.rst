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
     - Instance method typing/lowering
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

See also
--------

- :doc:`overview`
- :doc:`static`
- :doc:`dynamic`
