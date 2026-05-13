Argument intents
================

Argument intents control how C++ parameters are exposed in generated Numba
bindings. You can configure them statically with ``Function Argument Intents``
in binding config files or programmatically with ``arg_intent`` when calling
the template/API.

C++ source of truth
-------------------

.. code-block:: c++

  struct RunningStats {
    int count;
    float sum;
    float sum_sq;
  };

  __device__ void stats_update(RunningStats &state, float x);

  __device__ void stats_get_mean(const RunningStats &state, float &mean_out);

  __device__ bool stats_update_and_get_zscore(
      RunningStats &state, float x, float &zscore_out);

  // Logical 3x4 matrix stored row-major in a flat native buffer.
  __device__ void stats_get_matrix_3x4(float out[12]);

  __device__ void stats_get_vectors(float4 out[3]);

Example config
--------------

.. code-block:: yaml

  Function Argument Intents:
    stats_update:
      state: inout_ptr
    stats_get_mean:
      state: in
      mean_out: out_ptr
    stats_update_and_get_zscore:
      state: inout_ptr
      zscore_out: out_return
    stats_get_matrix_3x4:
      out:
        intent: out_array_return
        dtype: float
        length: 12
    stats_get_vectors:
      out:
        intent: out_array_return
        dtype: float4
        length: 3

Programmatic API
----------------

.. code-block:: python

  from numba.cuda.types import float32
  from numbast import bind_cxx_functions, out_array_return

  bindings = bind_cxx_functions(
      shim_writer,
      funcs,
      arg_intent={
          "stats_get_matrix_3x4": {
              "out": out_array_return(dtype=float32, length=12),
          },
      },
  )

Intent semantics
----------------

``in``
^^^^^^

- Default mode.
- Parameter stays visible in the Python call signature.
- Parameter is typed as the base value type ``T``.
- For C++ references, updates done by C++ are not surfaced as Python-visible outputs.

``inout_ptr``
^^^^^^^^^^^^^

- Parameter stays visible in the Python call signature.
- Parameter is typed as ``CPointer(T)``.
- Use this when C++ mutates referenced state and the caller should pass an addressable pointer.

``out_ptr``
^^^^^^^^^^^

- Parameter stays visible in the Python call signature.
- Parameter is typed as ``CPointer(T)``.
- Use this for explicit output buffers where the caller owns storage and passes a pointer.

``out_return``
^^^^^^^^^^^^^^

- Parameter is removed from the visible Python call arguments.
- Numbast allocates temporary storage, passes it to C++, then returns the value to Python.
- If C++ also returns a non-``void`` value, generated return type is packed as a tuple.

``out_array_return``
^^^^^^^^^^^^^^^^^^^^

- Pointer or fixed-size array output parameter is removed from the visible Python call arguments.
- Numbast allocates fixed-size native stack storage, passes the raw pointer to
  C++ through the shim, loads each element after the call, and returns a fixed
  ``UniTuple``.
- ``dtype`` is the element type and ``length`` is the number of elements to load.
- Multidimensional data is returned as a flat tuple. For example, a logical
  3x4 matrix uses ``length: 12`` and row-major indexing
  ``value[row * 4 + col]``.
- Static configs use C++ or registered type names such as ``float`` or
  ``float4``. Programmatic bindings can use Numba types such as ``float32`` or
  registered C++ type names.

Generated Python signatures
---------------------------

Representative signatures for the example API:

.. code-block:: python

  # in (default):
  signature(void, _type_RunningStats, float32)

  # inout_ptr:
  signature(void, CPointer(_type_RunningStats), float32)

  # out_ptr:
  signature(void, _type_RunningStats, CPointer(float32))

  # out_return with existing non-void C++ return (bool):
  signature(
      types.Tuple((bool_, float32)),
      CPointer(_type_RunningStats),
      float32,
  )

  # out_array_return:
  signature(UniTuple(float32, 12))  # logical 3x4 matrix, flattened
  signature(UniTuple(float32x4, 3))

Notes
-----

- ``inout_ptr``, ``out_ptr``, and ``out_return`` are only supported on C++
  reference parameters (``T&`` / ``T&&``).
- ``out_array_return`` is supported on pointer/array output parameters such as
  ``float *out``, ``float out[12]``, and ``float4 out[3]``.
- ``out_array_return`` returns a one-dimensional ``UniTuple``. For logical
  multidimensional outputs, use the total element count as ``length`` and
  flatten the indexing convention in the binding documentation.
- In ``Function Argument Intents``, parameter overrides can be keyed by
  parameter name or 0-based parameter index.

  .. code-block:: yaml

    0: inout_ptr  # demonstrates 0-based parameter indexing for overrides
- ``out_return`` removes that parameter from visible arguments.
- If C++ has a non-``void`` return and one or more ``out_return`` parameters,
  Numbast returns ``types.Tuple((cxx_return, out1, ...))``.
- If C++ returns ``void`` and there is exactly one ``out_return``, Numbast
  returns that value directly (not a tuple).
- If C++ returns ``void`` and there are multiple ``out_return`` parameters,
  Numbast returns ``types.Tuple((out1, out2, ...))``.
- ``out_array_return`` values participate in the same return packing rules as
  ``out_return``. A single ``void`` function output returns the ``UniTuple``
  directly; multiple outputs or a non-``void`` C++ return are packed in an
  outer ``types.Tuple``.
